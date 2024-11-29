from collections import deque
from typing import List, Tuple, Any

import gymnasium as gym


class MultiStepEnvWrapper(gym.Wrapper):
    """
    A Gym environment wrapper that buffers multiple steps of observations, actions, rewards, terminations, truncations, and infos.
    """

    def __init__(self, env: gym.Env, num_steps: int) -> None:
        """
        Initialize the MultiStepEnvWrapper.

        Args:
            env (gym.Env): The environment to wrap.
            num_steps (int): The number of steps to buffer.
        """
        super().__init__(env)
        self.num_steps = num_steps
        self.obs_buffer = deque(maxlen=num_steps)
        self.action_buffer = deque(maxlen=num_steps)
        self.reward_buffer = deque(maxlen=num_steps)
        self.terminated_buffer = deque(maxlen=num_steps)
        self.truncated_buffer = deque(maxlen=num_steps)
        self.info_buffer = deque(maxlen=num_steps)

    def reset(self, **kwargs) -> Any:
        """
        Reset the environment and clear all buffers.

        Args:
            **kwargs: Additional arguments for the environment reset.

        Returns:
            Any: The initial observation from the environment.
        """
        obs = self.env.reset(**kwargs)
        self._clear_buffers()
        self.obs_buffer.append(obs)
        return obs

    def step(self, action_sequence: List[Any]) -> Tuple[Any, float, bool, bool, Any]:
        """
        Execute a sequence of actions in the environment.

        Args:
            action_sequence (List[Any]): A list of actions to execute.

        Returns:
            Tuple[Any, float, bool, bool, Any]: Last observation, total reward, terminated flag, truncated flag, last info.
        """
        total_reward = 0.0
        terminated = False
        truncated = False
        last_info = {}
        for action in action_sequence:
            obs, reward, terminated, truncated, info = self.env.step(action)
            self._update_buffers(obs, action, reward, terminated, truncated, info)
            total_reward += reward
            last_info = info
            if terminated or truncated:
                break
        return self.obs_buffer[-1], total_reward, terminated, truncated, last_info

    def get_buffers(
        self,
    ) -> Tuple[List[Any], List[Any], List[float], List[bool], List[bool], List[Any]]:
        """
        Retrieve the current buffers.

        Returns:
            Tuple[List[Any], List[Any], List[float], List[bool], List[bool], List[Any]]:
                Observations, actions, rewards, terminations, truncations, infos.
        """
        return (
            list(self.obs_buffer),
            list(self.action_buffer),
            list(self.reward_buffer),
            list(self.terminated_buffer),
            list(self.truncated_buffer),
            list(self.info_buffer),
        )

    def _clear_buffers(self) -> None:
        """Clear all buffers."""
        self.obs_buffer.clear()
        self.action_buffer.clear()
        self.reward_buffer.clear()
        self.terminated_buffer.clear()
        self.truncated_buffer.clear()
        self.info_buffer.clear()

    def _update_buffers(
        self,
        obs: Any,
        action: Any,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: Any,
    ) -> None:
        """Update all buffers with new step data."""
        self.obs_buffer.append(obs)
        self.action_buffer.append(action)
        self.reward_buffer.append(reward)
        self.terminated_buffer.append(terminated)
        self.truncated_buffer.append(truncated)
        self.info_buffer.append(info)
