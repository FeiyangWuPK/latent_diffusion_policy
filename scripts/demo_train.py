from latent_diffusion_policy import EnvRunner, LatentDiffusionPolicyWithFiLM
import hydra
from omegaconf import DictConfig


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    print(cfg)

    return
    policy = LatentDiffusionPolicyWithFiLM(
        transformer_config_name=cfg.transformer_config_name,
        action_dim=cfg.action_dim,
        latent_dim=cfg.latent_dim,
        num_inference_steps=cfg.num_inference_steps,
    )
    # Create a latent diffusion policy
    runner = EnvRunner(
        cfg.task_name,
        policy=policy,
        num_steps=cfg.num_steps,
        num_envs=cfg.num_envs,
        num_sequences=cfg.num_sequences,
        num_workers=cfg.num_workers,
        device=cfg.device,
    )

    # Train the policy
    runner.train()


if __name__ == "__main__":
    main()
