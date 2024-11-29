import pytest
import gymnasium as gym
import gymnasium_robotics

from latent_diffusion_policy.utils.wrapper import MultiStepEnvWrapper

gym.register_envs(gymnasium_robotics)


def test_visualize_pick_and_place():
    env = gym.make("FetchPickAndPlace-v3", render_mode="human", max_episode_steps=100)
    env.reset()
    for _ in range(100):
        action = env.action_space.sample()
        env.step(action)

    env.close()


def test_visualize_multi_step_env_wrapper():
    env = gym.make("FetchPickAndPlace-v3", render_mode="human", max_episode_steps=100)
    env = MultiStepEnvWrapper(env, num_steps=3)
    env.reset()
    for _ in range(100):
        action_sequence = [env.action_space.sample() for _ in range(3)]
        env.step(action_sequence)

    env.close()


if __name__ == "__main__":
    pytest.main([__file__])
