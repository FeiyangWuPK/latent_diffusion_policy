from typing import Tuple

import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModel
from diffusers import UNet2DModel, DDPMScheduler
from torchrl.envs import GymEnv

# from torchrl.collectors import SyncDataCollector
# from torchrl.data import TensorDict

from utils.wrapper import MultiStepEnvWrapper


class FiLMLayer(nn.Module):
    """
    A Feature-wise Linear Modulation (FiLM) layer.
    """

    def __init__(self, feature_dim: int, conditioning_dim: int):
        """
        Args:
            feature_dim (int): The dimension of the features to be modulated.
            conditioning_dim (int): The dimension of the conditioning vector.
        """
        super().__init__()
        self.scale = nn.Linear(conditioning_dim, feature_dim)
        self.shift = nn.Linear(conditioning_dim, feature_dim)

    def forward(
        self, features: torch.Tensor, conditioning: torch.Tensor
    ) -> torch.Tensor:
        """
        Modulate the features using the conditioning vector.

        Args:
            features (torch.Tensor): The features to be modulated, shape (batch, feature_dim, ...).
            conditioning (torch.Tensor): The conditioning vector, shape (batch, conditioning_dim).

        Returns:
            torch.Tensor: Modulated features, same shape as input features.
        """
        gamma = self.scale(conditioning)  # (batch, feature_dim)
        beta = self.shift(conditioning)  # (batch, feature_dim)

        # Broadcast gamma and beta to match feature dimensions
        gamma = gamma.view(gamma.size(0), -1, *([1] * (features.dim() - 2)))
        beta = beta.view(beta.size(0), -1, *([1] * (features.dim() - 2)))

        return gamma * features + beta


class LatentDiffusionPolicyWithFiLM(nn.Module):
    """
    Latent Diffusion Policy with FiLM-based conditional modulation.
    """

    def __init__(
        self,
        transformer_config_name: str = "bert-base-uncased",
        action_dim: int = 6,
        latent_dim: int = 128,
        num_inference_steps: int = 50,
    ):
        super().__init__()

        # Transformer backbone to extract conditioning vector
        transformer_config = AutoConfig.from_pretrained(transformer_config_name)
        transformer_config.output_hidden_states = True
        self.transformer = AutoModel.from_config(transformer_config)

        # Map transformer output to a conditioning vector
        self.condition_projection = nn.Linear(
            transformer_config.hidden_size, latent_dim
        )

        # Diffusion model (UNet) with FiLM layers
        self.unet = UNet2DModel(
            sample_size=None,
            in_channels=1,
            out_channels=1,
            layers_per_block=2,
            block_out_channels=(128, 256),
            down_block_types=("DownBlock2D", "DownBlock2D"),
            up_block_types=("UpBlock2D", "UpBlock2D"),
        )

        # FiLM layers for conditioning the UNet
        self.film_layers = nn.ModuleList(
            [
                FiLMLayer(feature_dim, latent_dim)
                for feature_dim in self.unet.config.block_out_channels
            ]
        )

        # Scheduler for the diffusion process
        self.scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="linear",
        )

        # Action head to decode latent outputs to actions
        self.action_head = nn.Linear(latent_dim, action_dim)

        self.num_inference_steps = num_inference_steps

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to produce actions given observations.

        Args:
            observations (torch.Tensor): shape (batch_size, seq_len, obs_dim).

        Returns:
            actions (torch.Tensor): shape (batch_size, seq_len, action_dim).
        """
        batch_size, seq_len, obs_dim = observations.shape

        # 1. Extract conditioning vector from transformer
        transformer_out = self.transformer(inputs_embeds=observations)
        hidden_states = transformer_out.last_hidden_state
        conditioning_vector = hidden_states.mean(dim=1)  # Reduce over sequence
        conditioning_vector = self.condition_projection(conditioning_vector)

        # 2. Diffusion process with FiLM conditioning
        shape = (batch_size, 1, seq_len)
        latents = torch.randn(shape, device=observations.device)
        self.scheduler.set_timesteps(
            self.num_inference_steps, device=observations.device
        )

        for t in self.scheduler.timesteps:
            # Predict noise using the UNet
            model_output = self.unet(latents, t)

            # Apply FiLM layers to modulate activations
            for i, film_layer in enumerate(self.film_layers):
                model_output[i] = film_layer(model_output[i], conditioning_vector)

            # Compute the next latent step
            latents = self.scheduler.step(
                model_output.sample, t, latents, eta=0.0
            ).prev_sample

        # 3. Decode latents into actions
        latents_flat = (
            latents.permute(0, 2, 1).contiguous().view(batch_size * seq_len, -1)
        )
        actions = self.action_head(latents_flat).view(batch_size, seq_len, -1)
        return actions


class ExpertDataset(Dataset):
    """
    A PyTorch dataset for expert demonstrations (state-action pairs).
    """

    def __init__(self, states: torch.Tensor, actions: torch.Tensor):
        """
        Initialize the ExpertDataset.

        Args:
            states (torch.Tensor): Tensor of states (N, T, obs_dim).
            actions (torch.Tensor): Tensor of actions (N, T, action_dim).
        """
        super().__init__()
        self.states = states
        self.actions = actions

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.states[idx], self.actions[idx]


class EnvRunner:
    """
    A class to encapsulate environment interaction, data collection, and training for RL or BC algorithms.
    """

    def __init__(
        self,
        env_name: str,
        policy: LatentDiffusionPolicyWithFiLM,
        num_steps: int = 4,
        lr: float = 1e-3,
        batch_size: int = 64,
        device: str = "cpu",
    ):
        """
        Initialize the EnvRunner.

        Args:
            env_name (str): Name of the Gym environment to use.
            policy (LatentDiffusionPolicyWithFiLM): The policy network to train.
            num_steps (int): Number of steps for multi-step environment.
            lr (float): Learning rate for training the policy.
            batch_size (int): Batch size for training.
            device (str): Device to use ("cpu" or "cuda").
        """
        self.device = torch.device(device)

        # Wrap the Gym environment
        base_env = gym.make(env_name)
        wrapped_env = MultiStepEnvWrapper(base_env, num_steps=num_steps)
        self.env = GymEnv(wrapped_env).to(self.device)

        # Initialize policy
        self.policy = policy.to(self.device)

        # Optimizer for behavior cloning
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        # Loss function for BC
        self.loss_fn = nn.MSELoss()

        # Batch size for training
        self.batch_size = batch_size

        # Expert dataset placeholder
        self.expert_dataset = None
        self.data_loader = None

    def load_expert_data(self, states: torch.Tensor, actions: torch.Tensor) -> None:
        """
        Load expert demonstrations into a dataset and prepare a DataLoader.

        Args:
            states (torch.Tensor): Expert states, shape (N, T, obs_dim).
            actions (torch.Tensor): Expert actions, shape (N, T, action_dim).
        """
        self.expert_dataset = ExpertDataset(states, actions)
        self.data_loader = DataLoader(
            self.expert_dataset, batch_size=self.batch_size, shuffle=True
        )
        print(f"Loaded expert data: {len(self.expert_dataset)} samples.")

    def update_policy(self) -> float:
        """
        Update the policy using behavior cloning (BC) with expert demonstrations.

        Returns:
            float: The average loss value over the dataset.
        """
        if self.data_loader is None:
            raise ValueError(
                "Expert data has not been loaded. Use `load_expert_data` first."
            )

        total_loss = 0.0
        num_batches = 0

        for states, expert_actions in self.data_loader:
            states, expert_actions = states.to(self.device), expert_actions.to(
                self.device
            )

            # Forward pass through the policy
            predicted_actions = self.policy(states)

            # Compute the BC loss
            loss = self.loss_fn(predicted_actions, expert_actions)

            # Backpropagation and optimization step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        return avg_loss

    def train(self, num_epochs: int = 10) -> None:
        """
        Train the policy using behavior cloning.

        Args:
            num_epochs (int): Number of epochs for training.
        """
        for epoch in range(num_epochs):
            avg_loss = self.update_policy()
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

    def evaluate(self, num_episodes: int = 5) -> float:
        """
        Evaluate the policy in the environment.

        Args:
            num_episodes (int): Number of episodes to run for evaluation.

        Returns:
            float: The average reward over the evaluation episodes.
        """
        total_reward = 0.0

        for episode in range(num_episodes):
            obs = self.env.reset()
            episode_reward = 0.0
            done = False

            while not done:
                obs_tensor = (
                    torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
                )
                with torch.no_grad():
                    action = self.policy(obs_tensor)
                action_np = action.squeeze(0).cpu().numpy()
                obs, reward, terminated, truncated, _ = self.env.step(action_np)
                episode_reward += reward
                done = terminated or truncated

            total_reward += episode_reward
            print(f"Episode {episode + 1}/{num_episodes}, Reward: {episode_reward:.2f}")

        avg_reward = total_reward / num_episodes
        print(f"Average Reward over {num_episodes} episodes: {avg_reward:.2f}")
        return avg_reward
