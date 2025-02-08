import torch
from torch.nn import MSELoss, HuberLoss
import torch.optim as optim
import numpy as np
import random
import wandb
from datetime import datetime
from policy_dqn import DQN, ReplayBuffer
from state_custom import global_state, update_single_unit_energy


class AgentRl:
    def __init__(self, player: str, env_cfg) -> None:
        """
        Baseline init.
        """
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        self.team_id = 0 if self.player == "player_0" else 1
        self.opp_team_id = 1 if self.team_id == 0 else 0
        self.env_cfg = env_cfg

    def build_model(self, config: dict):
        """
        Build model with config, after baseline initialization.
        """
        # State representation
        self.state = None
        self.relics_mask = np.zeros((6,), dtype=bool)
        self.relics_position = np.full((6, 2), -1)
        if config["resume"]:
            self.load_model(config[self.player]["saved_model"])
            return
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Hyperparameters
        self.batch_size = config["hyper"]["batch_size"]
        self.epsilon = config["hyper"]["epsilon"]
        self.epsilon_decay = config["hyper"]["epsilon_decay"]
        self.epsilon_min = config["hyper"]["epsilon_min"]
        self.gamma = config["hyper"]["gamma"]
        self.lr_rate = config["hyper"]["lr_rate"]
        # Model parameters
        self.action_size = config[self.player]["action_size"]
        self.policy_net = DQN(config[self.player]["channels"], config[self.player]["hidden_size"],
                              config[self.player]["action_size"]).to(self.device)
        self.target_net = DQN(config[self.player]["channels"], config[self.player]["hidden_size"],
                              config[self.player]["action_size"]).to(self.device)
        self.update_target_net()
        self.memory = ReplayBuffer(config[self.player]["buffer"])
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr_rate)
        self.loss = MSELoss() if config[self.player]["loss"] == 'MSE' else HuberLoss()

    def update_env_cfg(self, new_cfg):
        self.env_cfg = new_cfg

    def state_representation(self, obs):
        self.state, self.relics_mask, self.relics_position = global_state(obs, self.relics_mask, self.relics_position,
                                                                          self.team_id, self.opp_team_id)
        return

    def get_global_state(self):
        return self.state.copy()

    def get_relic_mask(self):
        return self.relics_mask.copy()

    def get_relic_position(self):
        return self.relics_position.copy()

    def get_single_state(self, global_state, energy, x, y) -> np.ndarray:
        return update_single_unit_energy(global_state, energy, x, y)

    def act(self, state_single, x, y):
        """
        Get single state obs and output action.
        """
        with torch.no_grad():
            if random.random() < self.epsilon:
                action_type = np.random.choice(self.action_size)
            else:
                action_type = self.policy_net(torch.from_numpy(state_single).to(self.device)).argmax().item()
        # Sap action
        if action_type == 5:
            # check if state[1] (opponent channel) is full of zeros
            sap_range = self.env_cfg["unit_sap_range"]
            north = max(0, x - sap_range)
            west = max(0, y - sap_range)
            south = min(23, x + sap_range)
            east = min(23, y + sap_range)
            if np.any(state_single[1, north:south+1, west:east+1]):
                # there is at leat one visible opponent
                for target_x in range(north, south + 1):
                    for target_y in range(west, east + 1):
                        if state_single[1, target_x, target_y] > 0:
                            # for env, x and y are inverted
                            return [5, target_y, target_x]
            else:
                # no opponent at sap range --> sap random cell
                fake_dx = random.randint(0, sap_range + 1)
                fake_dy = random.randint(0, sap_range + 1)
                if (y + fake_dy) < self.env_cfg["max_width"]:
                    target_y = y + fake_dy
                else:
                    target_y = y - fake_dy
                if (x + fake_dx) < self.env_cfg["max_height"]:
                    target_x = x + fake_dx
                else:
                    target_x = x - fake_dx
                # for env, x and y are inverted
                return [5, target_x, target_y]
        else:
            return [action_type, 0, 0]

    def predict(self, step: int, obs, remainingOverageTime: int = 60):
        """
        Prediction with target net.
        """
        self.state_representation(obs)
        # state = self.state_representation(obs)
        actions = np.zeros((self.env_cfg["max_units"], 3), dtype=int)
        available_units = np.where(obs["units_mask"][self.team_id])[0]
        available_opponents = np.where(obs["units_mask"][self.opp_player])[0]

        for unit_id in available_units:
            energy_single = obs["units"]["energy"][self.team_id, unit_id, 0]
            # in obs, x & y are inverted
            y_single = obs["units"]["position"][self.team_id, unit_id, 0]
            x_single = obs["units"]["position"][self.team_id, unit_id, 1]
            state_single = self.get_single_state(self.state.copy(), energy_single, x_single, y_single)
            # call greedy policy or epsilon-random action
            with torch.no_grad():
                action_type = self.policy_net(torch.from_numpy(state_single).to(self.device)).argmax().item()
            # Sap action
            if action_type == 5:
                if available_opponents:
                    for opp_unit_id in available_opponents:
                        # in obs, x & y are inverted
                        opp_y = obs["units"]["position"][self.opp_team_id, opp_unit_id, 0]
                        opp_x = obs["units"]["position"][self.opp_team_id, opp_unit_id, 1]
                        if abs(opp_x - x_single) <= self.env_cfg["unit_sap_range"] and abs(opp_y - y_single) <= \
                                self.env_cfg["unit_sap_range"]:
                            actions[unit_id] = [5, opp_y, opp_x]
                            break
                        else:
                            continue
                else:
                    fake_delta_y = random.randint(0, self.env_cfg["unit_sap_range"] + 1)
                    fake_delta_x = random.randint(0, self.env_cfg["unit_sap_range"] + 1)
                    if (y_single + fake_delta_y) < self.env_cfg["max_width"]:
                        target_y = y_single + fake_delta_y
                    else:
                        target_y = y_single - fake_delta_y
                    if (x_single + fake_delta_x) < self.env_cfg["max_height"]:
                        target_x = x_single + fake_delta_x
                    else:
                        target_x = x_single - fake_delta_x
                    actions[unit_id] = [5, target_y, target_x]
        return actions

    def learn(self, step, player, training=True):
        if not training or len(self.memory) < self.batch_size:
            return
        batch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.stack(states)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.stack(next_states)
        dones = torch.FloatTensor(dones).to(self.device)

        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_net(next_states).max(1)[0].detach()
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        loss = self.loss(current_q_values.squeeze(), target_q_values)
        loss_name = "loss_0" if player == "player_0" else "loss_1"
        wandb.log({loss_name: loss})

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network and decrease epsilon after every game
        # if step % 504 == 0:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        epsilon_name = "epsilon_0" if player == "player_0" else "epsilon_1"
        wandb.log({epsilon_name: self.epsilon})

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save_model(self):
        now_str = datetime.now().strftime("%Y-%m-%d_%H:%M")
        name_model = "dqn_" + self.player + now_str
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, f'model_pytorch/{name_model}.pth')

    def load_model(self, config: dict):
        try:
            path = config["path_models"] + config["saved_model_0"] if self.player == "player_0" else config["path_models"] + config["saved_model_1"]
            checkpoint = torch.load(path)
            self.policy_net.load_state_dict(checkpoint['policy_net'])
            self.target_net.load_state_dict(checkpoint['target_net'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        except FileNotFoundError:
            raise FileNotFoundError(f"No trained model found for {self.player}")
