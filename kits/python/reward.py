import numpy as np

def get_reward(global_type: str, single_type: str, obs: dict, unit_state, player: int, last_points: np.array) -> float:
    reward_global = get_global_reward(global_type, unit_state, player, obs, last_points)
    # reward_unit = get_unit_reward()
    return reward_global


def get_global_reward(global_type: str, unit_state, player: int, obs: dict, last_points: np.array) -> float:
    # exploration reward
    exp_reward = reward_exploration(obs)
    # delta points reward
    delta_reward = reward_delta_points(obs, player, last_points) / 16
    # create weights
    dp_weight = (obs["match_steps"] % 101) / 101.0
    exp_weight = 1 - dp_weight
    if global_type == 'delta_points_exploration':
        pass
    elif global_type == 'delta_points_relic_exploration':
        # discovered relics (check channel 4 of unit state)
        discovered_relics = unit_state[4].sum()
        visible_relics = obs["relic_nodes_mask"].sum()
        exp_reward += (visible_relics/6)
        exp_reward -= (discovered_relics/6)
    else:
        raise ValueError(f"Unknown reward type: {global_type}")
    return (dp_weight * delta_reward) + (exp_weight * exp_reward)


def reward_gap_points(obs: dict, player: int) -> float:
    """
    Return (player score) - (opponent score).
    """
    gap = obs["team_points"][player] - obs["team_points"][1 - player]
    return float(gap)

def reward_delta_points(obs:dict, player: int, last_points: np.array) -> float:
    """
    Return (points score) - (previous point score).
    """
    delta = obs["team_points"] - last_points
    return float(delta[player])

def reward_exploration(obs: dict) -> float:
    """
    Return the visible map, according to observation.
    """
    tot_cells = obs["sensor_mask"].shape[0] * obs["sensor_mask"].shape[1]
    visible_cells = obs["sensor_mask"].sum()
    return visible_cells / tot_cells
