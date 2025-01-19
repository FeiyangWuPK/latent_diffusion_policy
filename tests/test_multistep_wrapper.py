import pytest
import gymnasium as gym
import gymnasium_robotics
import itertools
from latent_diffusion_policy.utils import MultiStepEnvWrapper

gym.register_envs(gymnasium_robotics)


def test_visualize_multi_step_env_wrapper():
    env = gym.make("FetchPickAndPlace-v3", render_mode="human", max_episode_steps=100)
    env = MultiStepEnvWrapper(env, num_steps=3)
    env.reset()
    for step in itertools.count():
        action_sequence = [env.action_space.sample() for _ in range(3)]
        next_obs, reward, done, truncated, info = env.step(action_sequence)
        if done or truncated or step > 1000:
            break

    env.close()


if __name__ == "__main__":
    pytest.main([__file__])
