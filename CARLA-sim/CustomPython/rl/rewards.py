"""
Reward functions for the CARLA RL environment.

Designed for highway driving with the following objectives:
1. Progress along the track (primary)
2. Lane keeping
3. Speed maintenance
4. Collision avoidance
5. Smooth control
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

from rl.config import RewardConfig, ScenarioConfig
from rl.observations import EpisodeState


@dataclass
class RewardInfo:
    """Breakdown of reward components for logging/debugging."""
    
    total: float = 0.0
    progress: float = 0.0
    lane_keeping: float = 0.0
    speed: float = 0.0
    collision: float = 0.0
    goal: float = 0.0
    smoothness: float = 0.0
    alive: float = 0.0


def compute_reward(
    prev_state: EpisodeState,
    curr_state: EpisodeState,
    action: np.ndarray,
    prev_action: Optional[np.ndarray],
    collision_occurred: bool,
    goal_reached: bool,
    reward_config: RewardConfig,
    scenario_config: ScenarioConfig
) -> Tuple[float, RewardInfo]:
    """
    Compute the reward for a single step.
    
    Args:
        prev_state: State before the action
        curr_state: State after the action
        action: Action taken [throttle_brake, steering]
        prev_action: Previous action (for smoothness penalty)
        collision_occurred: Whether a collision happened this step
        goal_reached: Whether the goal was reached this step
        reward_config: Reward function weights
        scenario_config: Scenario configuration
        
    Returns:
        Tuple of (total_reward, RewardInfo breakdown)
    """
    info = RewardInfo()
    
    # 1. Progress reward (main objective)
    # Reward for moving forward along the track
    progress = curr_state.ego.x - prev_state.ego.x
    info.progress = reward_config.progress_weight * progress
    
    # 2. Lane keeping penalty
    # Penalize deviation from lane center
    lane_deviation = abs(curr_state.ego.lane_offset)
    info.lane_keeping = -reward_config.lane_deviation_weight * lane_deviation
    
    # 3. Speed maintenance
    # Penalize deviation from target speed
    target_speed = scenario_config.target_speed_mps
    speed_deviation = abs(curr_state.ego.speed - target_speed)
    info.speed = -reward_config.speed_deviation_weight * speed_deviation
    
    # 4. Collision penalty (terminal)
    if collision_occurred:
        info.collision = -reward_config.collision_penalty
    
    # 5. Goal reached bonus (terminal)
    if goal_reached:
        info.goal = reward_config.goal_bonus
    
    # 6. Action smoothness penalty
    # Penalize jerky control inputs
    if prev_action is not None:
        action_diff = np.abs(action - prev_action)
        smoothness_penalty = np.sum(action_diff)
        info.smoothness = -reward_config.action_smoothness_weight * smoothness_penalty
    
    # 7. Small alive bonus to encourage survival
    info.alive = 0.01
    
    # Sum all components
    info.total = (
        info.progress +
        info.lane_keeping +
        info.speed +
        info.collision +
        info.goal +
        info.smoothness +
        info.alive
    )
    
    return info.total, info


def compute_reward_simple(
    prev_x: float,
    curr_x: float,
    lane_offset: float,
    speed: float,
    collision: bool,
    goal_reached: bool,
    target_speed: float = 8.33
) -> float:
    """
    Simplified reward function for quick testing.
    
    Args:
        prev_x: Previous x position
        curr_x: Current x position
        lane_offset: Distance from lane center
        speed: Current speed in m/s
        collision: Whether collision occurred
        goal_reached: Whether goal was reached
        target_speed: Target speed in m/s
        
    Returns:
        Scalar reward value
    """
    reward = 0.0
    
    # Progress (main objective)
    reward += 0.1 * (curr_x - prev_x)
    
    # Lane keeping
    reward -= 0.5 * abs(lane_offset)
    
    # Speed maintenance
    reward -= 0.1 * abs(speed - target_speed)
    
    # Collision penalty
    if collision:
        reward -= 100.0
    
    # Goal bonus
    if goal_reached:
        reward += 50.0
    
    # Small alive bonus
    reward += 0.01
    
    return reward


class RewardShaper:
    """
    Stateful reward shaper that tracks history for potential-based shaping.
    
    This can help with sparse rewards by providing dense shaping signals.
    """
    
    def __init__(
        self,
        reward_config: RewardConfig,
        scenario_config: ScenarioConfig,
        use_potential_shaping: bool = False
    ):
        self.reward_config = reward_config
        self.scenario_config = scenario_config
        self.use_potential_shaping = use_potential_shaping
        
        self._prev_state: Optional[EpisodeState] = None
        self._prev_action: Optional[np.ndarray] = None
        self._prev_potential: float = 0.0
    
    def reset(self) -> None:
        """Reset the shaper state for a new episode."""
        self._prev_state = None
        self._prev_action = None
        self._prev_potential = 0.0
    
    def _compute_potential(self, state: EpisodeState) -> float:
        """
        Compute potential function for potential-based reward shaping.
        
        Using progress as the potential gives a dense reward signal
        while maintaining optimal policy invariance.
        """
        # Potential based on progress and lane keeping
        progress_potential = state.progress * 10.0
        lane_potential = -abs(state.ego.lane_offset) * 0.5
        
        return progress_potential + lane_potential
    
    def compute(
        self,
        state: EpisodeState,
        action: np.ndarray,
        collision: bool,
        goal_reached: bool
    ) -> Tuple[float, RewardInfo]:
        """
        Compute reward for the current step.
        
        Args:
            state: Current episode state
            action: Action taken
            collision: Whether collision occurred
            goal_reached: Whether goal was reached
            
        Returns:
            Tuple of (total_reward, RewardInfo)
        """
        if self._prev_state is None:
            # First step - no previous state to compare
            self._prev_state = state
            self._prev_action = action
            self._prev_potential = self._compute_potential(state)
            return 0.0, RewardInfo()
        
        # Compute base reward
        reward, info = compute_reward(
            prev_state=self._prev_state,
            curr_state=state,
            action=action,
            prev_action=self._prev_action,
            collision_occurred=collision,
            goal_reached=goal_reached,
            reward_config=self.reward_config,
            scenario_config=self.scenario_config
        )
        
        # Apply potential-based shaping if enabled
        if self.use_potential_shaping:
            curr_potential = self._compute_potential(state)
            gamma = 0.99  # Discount factor
            shaping = gamma * curr_potential - self._prev_potential
            reward += shaping
            info.total = reward
            self._prev_potential = curr_potential
        
        # Update history
        self._prev_state = state
        self._prev_action = action.copy()
        
        return reward, info


def get_termination_conditions(
    state: EpisodeState,
    collision: bool,
    step_count: int,
    scenario_config: ScenarioConfig
) -> Tuple[bool, bool, str]:
    """
    Check episode termination conditions.
    
    Args:
        state: Current episode state
        collision: Whether collision occurred
        step_count: Current step count
        scenario_config: Scenario configuration
        
    Returns:
        Tuple of (terminated, truncated, reason)
        - terminated: True if episode ended naturally (collision, goal)
        - truncated: True if episode was cut short (timeout, out of bounds)
        - reason: String describing why episode ended
    """
    # Goal reached (success)
    if state.ego.x >= scenario_config.goal_x:
        return True, False, "goal_reached"
    
    # Collision (failure)
    if collision:
        return True, False, "collision"
    
    # Time limit exceeded (truncation)
    if step_count >= scenario_config.max_episode_steps:
        return False, True, "timeout"
    
    # Out of bounds (off track)
    # Consider it failed if too far from the road
    max_lane_deviation = scenario_config.lane_width * 2.0
    if abs(state.ego.lane_offset) > max_lane_deviation:
        return True, False, "off_road"
    
    # Reversed too far (going backwards)
    if state.ego.x < scenario_config.spawn_x - 10.0:
        return True, False, "reversed"
    
    # Episode continues
    return False, False, ""

