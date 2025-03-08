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
        self.tau = config["hyper"]["tau"]
        self.lr_rate = config["hyper"]["lr_rate"]
        # Model parameters
        self.action_size = config[self.player]["action_size"]
        self.policy_net = DQN(config[self.player]["channels"], config[self.player]["hidden_size"],
                              config[self.player]["action_size"]).to(self.device)
        self.target_net = DQN(config[self.player]["channels"], config[self.player]["hidden_size"],
                              config[self.player]["action_size"]).to(self.device)
        self.update_target_net()
        self.target_net.eval()
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

    def get_single_state(self, state_global, energy, x, y) -> np.ndarray:
        return update_single_unit_energy(state_global, energy, x, y)

    def act(self, state_single, x, y):
        """
        Get single state obs and output action.
        """
        with torch.no_grad():
            if random.random() < self.epsilon:
                action_type = np.random.choice(self.action_size)
            else:
                state_tensor = torch.from_numpy(np.float32(state_single))
                state_tensor = state_tensor.to(self.device)
                state_tensor = state_tensor.unsqueeze(0)
                action_type = self.policy_net(state_tensor).argmax().item()
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
                if (y + fake_dy) < self.env_cfg["map_width"]:
                    target_y = y + fake_dy
                else:
                    target_y = y - fake_dy
                if (x + fake_dx) < self.env_cfg["map_height"]:
                    target_x = x + fake_dx
                else:
                    target_x = x - fake_dx
                # for env, x and y are inverted
                return [5, target_y, target_x]
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

        for unit_id in available_units:
            energy_single = obs["units"]["energy"][self.team_id, unit_id]
            # in obs, x & y are inverted
            y_single = obs["units"]["position"][self.team_id, unit_id, 0]
            x_single = obs["units"]["position"][self.team_id, unit_id, 1]
            state_single = self.get_single_state(self.state.copy(), energy_single, x_single, y_single)
            # call greedy policy or epsilon-random action
            with torch.no_grad():
                state_tensor = torch.from_numpy(np.float32(state_single))
                state_tensor = state_tensor.to(self.device)
                state_tensor = state_tensor.unsqueeze(0)
                action_type = self.target_net(state_tensor).argmax().item()
                # action_type = self.target_net(torch.from_numpy(state_single).to(self.device)).argmax().item()
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
                    # there is at least one visible opponent
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

    def learn(self, step, player, training=True):
        if not training or len(self.memory) < self.batch_size:
            return
        batch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.stack(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.stack(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_net(next_states).max(1)[0].detach()
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        loss = self.loss(current_q_values.squeeze(), target_q_values)
        loss_name = "loss_0" if player == "player_0" else "loss_1"
        wandb.log({loss_name: loss})

        self.optimizer.zero_grad()
        loss.backward()
        # clip gradient
        torch.nn.utils.clip_grad_norm_(parameters=self.policy_net.parameters(), max_norm=1)
        # log gradients
        epoch_grad_norms = [param.grad.norm(2).item() for param in self.policy_net.parameters() if param.grad is not None]
        gradient_name = "gradient_0" if self.player == "player_0" else "gradient_1"
        for grad_param in epoch_grad_norms:
            wandb.log({gradient_name: grad_param})

        self.optimizer.step()

        # Update target network and decrease epsilon after every game
        # if step % 504 == 0:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        epsilon_name = "epsilon_0" if player == "player_0" else "epsilon_1"
        wandb.log({epsilon_name: self.epsilon})

    def update_target_net(self):
        if self.tau is None:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        else:
            target_net_state_dict = self.target_net.state_dict()
            policy_net_state_dict = self.policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key] * self.tau + target_net_state_dict[key] * (1 - self.tau)
            self.target_net.load_state_dict(target_net_state_dict)

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
