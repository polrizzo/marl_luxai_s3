import random
import numpy as np
import torch
import os

from policy_dqn import DQN
from state_custom import global_state, update_single_unit_energy

class Agent():
    def __init__(self, player: str, env_cfg) -> None:
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        self.team_id = 0 if self.player == "player_0" else 1
        self.opp_team_id = 1 if self.team_id == 0 else 0
        self.env_cfg = env_cfg
        # Custom attributes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state = None
        self.relics_mask = np.zeros((6,), dtype=bool)
        self.relics_position = np.full((6, 2), -1)
        # Load DQN model
        self.action_size = 6
        self.model = DQN(channels=6, hidden_size=128, output_size=self.action_size)
        path_name = "dqn_player_0.pth" if self.team_id == 0 else "dqn_player_1.pth"
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), path_name)
        self.model.load_state_dict(torch.load(path, weights_only=True, map_location=self.device)["target_net"])
        self.model.to(self.device)
        self.model.eval()

    def state_representation(self, obs):
        self.state, self.relics_mask, self.relics_position = global_state(obs, self.relics_mask, self.relics_position,
                                                                          self.team_id, self.opp_team_id)
        return

    def get_single_state(self, state_global, energy, x, y) -> np.ndarray:
        return update_single_unit_energy(state_global, energy, x, y)

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        """implement this function to decide what actions to send to each available unit.

        step is the current timestep number of the game starting from 0 going up to max_steps_in_match * match_count_per_episode - 1.
        """
        self.state_representation(obs)
        # state = self.state_representation(obs)
        actions = np.zeros((self.env_cfg["max_units"], 3), dtype=int)
        available_units = np.where(obs["units_mask"][self.team_id])[0]

        for unit_id in available_units:
            energy_single = obs["units"]["energy"][self.team_id, unit_id]
            # in obs, x & y are inverted
            y_single = obs["units"]["position"][self.team_id, unit_id, 0]
            x_single = obs["units"]["position"][self.team_id, unit_id, 1]
            state_single = self.get_single_state(self.state.copy(), energy_single, x_single, y_single)
            # call greedy policy or epsilon-random action
            with torch.no_grad():
                state_tensor = torch.from_numpy(np.float32(state_single))
                state_tensor = state_tensor.unsqueeze(0)
                state_tensor = state_tensor.to(self.device)
                action_type = self.model(state_tensor).argmax().item()
            # Sap action
            if action_type == 5:
                # check if state[1] (opponent channel) is full of zeros
                sap_range = self.env_cfg["unit_sap_range"]
                north = max(0, x_single - sap_range)
                west = max(0, y_single - sap_range)
                south = min(23, x_single + sap_range)
                east = min(23, y_single + sap_range)
                sap_done = False
                if np.any(state_single[1, north:south + 1, west:east + 1]):
                    # there is at leat one visible opponent
                    for target_x in range(north, south + 1):
                        if sap_done:
                            break
                        for target_y in range(west, east + 1):
                            if state_single[1, target_x, target_y] > 0:
                                # for env, x and y are inverted
                                actions[unit_id] = [5, target_y, target_x]
                                sap_done = True
                                break
                else:
                    # no opponent at sap range --> sap random cell
                    fake_dx = random.randint(0, sap_range + 1)
                    fake_dy = random.randint(0, sap_range + 1)
                    if (y_single + fake_dy) < self.env_cfg["map_width"]:
                        target_y = y_single + fake_dy
                    else:
                        target_y = y_single - fake_dy
                    if (x_single + fake_dx) < self.env_cfg["map_height"]:
                        target_x = x_single + fake_dx
                    else:
                        target_x = x_single - fake_dx
                    # for env, x and y are inverted
                    actions[unit_id] = [5, target_y, target_x]
            else:
                actions[unit_id] = [action_type, 0, 0]
        return actions