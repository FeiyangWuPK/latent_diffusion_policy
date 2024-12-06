import gymnasium as gym
from skrl.agents.torch.ppo import PPO
from skrl.trainers import Trainer
from skrl.envs.wrappers.torch.gymnasium_envs import GymnasiumWrapper


# Custom environment registration function
def create_env(env_name):
    return gym.make(env_name)


# train a agent in gymnasium-robotics environment
def train_pick_and_place(
    env_name="FetchPickAndPlace-v1",
    num_episodes=1000,
    buffer_size=100000,
    batch_size=64,
    gamma=0.99,
    tau=0.005,
    lr=3e-4,
):
    # Create the environment
    env = GymnasiumWrapper(create_env(env_name))

    # Configuration for the PPO agent
    config = {
        "buffer_size": buffer_size,
        "batch_size": batch_size,
        "gamma": gamma,
        "tau": tau,
        "lr": lr,
    }

    # Create the PPO agent
    agent = PPO(env, config=config)

    # Create the trainer
    trainer = Trainer(agent)

    # Train the agent
    for _ in range(num_episodes):
        result = trainer.train()
        print(f"Episode reward: {result['episode_reward_mean']}")

    trainer.stop()


# Example usage
if __name__ == "__main__":
    train_pick_and_place()
