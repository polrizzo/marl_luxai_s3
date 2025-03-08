import numpy as np


# def get_global_reward(unit_state, player: int, obs: dict, last_points: np.array) -> float:
#     # exploration reward
#     exp_reward = reward_exploration(obs)
#     # delta points reward
#     delta_reward = reward_delta_points(obs, player, last_points) / 16
#     # create weights
#     dp_weight = (obs["match_steps"] % 101) / 101.0
#     exp_weight = 1 - dp_weight
#     if global_type == 'delta_points_exploration':
#         pass
#     elif global_type == 'delta_points_relic_exploration':
#         # discovered relics (check channel 4 of unit state)
#         discovered_relics = unit_state[4].sum()
#         visible_relics = obs["relic_nodes_mask"].sum()
#         exp_reward += (visible_relics/6)
#         exp_reward -= (discovered_relics/6)
#     else:
#         raise ValueError(f"Unknown reward type: {global_type}")
#     return (dp_weight * delta_reward) + (exp_weight * exp_reward)

def get_global_reward(player: int, obs: dict, last_points: np.array) -> float:
    # exploration reward (based on half-map)
    exp_reward = 2.0 * reward_exploration(obs)
    # delta points reward
    delta_reward = reward_delta_points(obs, player, last_points) / 16
    # create weights
    dp_weight = obs["steps"] / 101.0
    exp_weight = 1 - dp_weight
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


def get_unit_reward(unit_state, action, pos_x, pos_y, relics_mask, relics_position) -> float:
    action_type = action[0]
    if action_type == 5:  # sap action
        opp_x = int(action[2])
        opp_y = int(action[1])
        if unit_state[1, opp_x, opp_y] > 0:
            return float(1)
        else:
            return float(-1)
    else: # all other movement actions
        if np.any(relics_mask): # any relics detected?
            available_relics = np.where(relics_mask)[0]
            closest_relic = None
            closest_relic_distance = 48
            # check closest relic
            for relic in available_relics:
                distance = abs(pos_x - relics_position[relic, 0]) + abs(pos_y - relics_position[relic, 1])
                if distance < closest_relic_distance:
                    closest_relic = relic
                    closest_relic_distance = distance
            next_tile_x = pos_x
            next_tile_y = pos_y
            if action_type == 1:  # up
                next_tile_x -= 1
            elif action_type == 2:  # right
                next_tile_y -= 1
            elif action_type == 3:  # down
                next_tile_x += 1
            elif action_type == 4:  # left
                next_tile_y += 1
            new_distance = max(0, abs(next_tile_x - relics_position[closest_relic, 0] - 2)) + max(0, abs(next_tile_y - relics_position[closest_relic, 1] - 2))
            if new_distance == 0: # already in relic range (2x2)
                return float(1)
            else: # outside relic range
                return float(1 - new_distance/46) # 46 as max distance between two tiles on map
        else:
            next_tile_x = pos_x
            next_tile_y = pos_y
            if action_type == 1: # up
                next_tile_x -= 1
            elif action_type == 2: # right
                next_tile_y -= 1
            elif action_type == 3: # down
                next_tile_x += 1
            elif action_type == 4: # left
                next_tile_y += 1
            else: # stay
              return float(-1)
            if next_tile_x > 23 or next_tile_y > 23: # outside of map
                return float(-1)
            elif unit_state[2, next_tile_x, next_tile_y] > 0 or unit_state[3, next_tile_x, next_tile_y] > 0: # nebula or asteroid
                return float(-1)
            else:
                return float(0)