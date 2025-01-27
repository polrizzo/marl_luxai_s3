import numpy as np


def get_reward(type_reward: str, obs: dict, player: int, last_points: np.array) -> float:
    if type_reward == "only_points":
        return reward_points(obs, player)
    elif type_reward == "points_exploration":
        return reward_points(obs, player) + reward_exploration(obs)
    elif type_reward == "delta_points_exploration":
        return reward_delta_points(obs, player, last_points) + reward_exploration(obs)
    else:
        raise ValueError(f"Unknown reward type: {type_reward}")




def reward_points(obs: dict, player: int) -> float:
    """
    Return the points, according to observation obs and the player.
    """
    return obs["team_points"][player]

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