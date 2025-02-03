import numpy as np

def global_state(obs, relics_mask, relics_position, player: int, foe: int) -> tuple[np.ndarray, np.array, np.ndarray]:
    """
    Return global state representation of current obs.
    """
    state = np.zeros((6,24,24))
    print(obs["units_mask"][player])
    # channel 0: player's units ---------------------------------
    for unit_id in np.where(obs["units_mask"][player])[0]:
        # in obs, x & y are inverted
        y = obs["units"]["position"][player, unit_id, 0]
        x = obs["units"]["position"][player, unit_id, 1]
        # state[0, x, y] += 1
        state[0, x, y] += 1/16
    # channel 1: foe's units ------------------------------------
    for unit_id in np.where(obs["units_mask"][foe])[0]:
        # in obs, x & y are inverted
        y = obs["units"]["position"][foe, unit_id, 0]
        x = obs["units"]["position"][foe, unit_id, 1]
        # state[1, x, y] += 1
        state[1, x, y] += 1/16
    # channel 2-3: nebula & asteroid ----------------------------
    for x_ax in range(obs["map_features"]["tile_type"].shape[0]):
        for y_ax in range(obs["map_features"]["tile_type"].shape[1]):
            if obs["map_features"]["tile_type"][x_ax, y_ax] == 1:
                state[2, y_ax, x_ax] += 1
            elif obs["map_features"]["tile_type"][x_ax, y_ax] == 2:
                state[3, y_ax, x_ax] += 1
            else:
                pass
    # channel 4: relics -----------------------------------------
    for relic_id in np.where(relics_mask)[0]:
        x = relics_position[relic_id, 0]
        y = relics_position[relic_id, 1]
        # state[4, x, y] += 1
        state[4, x, y] += 1/6
    if np.array_equal(relics_mask, np.add(relics_mask, obs['relic_nodes_mask'])):
        pass
    else:
        for relic_id in np.where(obs['relic_nodes_mask'])[0]:
            if relics_mask[relic_id]:
                pass
            else:
                # in obs, x & y are inverted
                y = obs["relic_nodes"][relic_id, 0]
                x = obs["relic_nodes"][relic_id, 1]
                state[4, x, y] += 1/6
                relics_mask[relic_id] = True
                relics_position[relic_id, 0] = y
                relics_position[relic_id, 1] = x
    # channel 5: single unit's energy -----------------------------
    # it will be updated later
    return state, relics_mask, relics_position

def update_single_unit_energy(state_repr, energy, x, y) -> np.ndarray:
    """
    Add single unit's energy.
    """
    state_repr[5, x, y] = energy/400
    return state_repr

