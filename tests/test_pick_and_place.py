import pytest
import gymnasium as gym
import gymnasium_robotics

gym.register_envs(gymnasium_robotics)


def test_visualize_pick_and_place():
    env = gym.make("FetchPickAndPlace-v3", render_mode="human", max_episode_steps=100)
    env.reset()
    for _ in range(1000):
        action = env.action_space.sample()
        env.step(action)

    env.close()


if __name__ == "__main__":
    pytest.main([__file__])
