import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import wandb
from kits.python.lux.utils import direction_to
from policy import DQN, ReplayBuffer


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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = config["hyper"]["batch_size"]
        self.epsilon = config["hyper"]["epsilon"]
        self.epsilon_decay = config["hyper"]["epsilon_decay"]
        self.epsilon_min = config["hyper"]["epsilon_min"]
        self.eval = config["hyper"]["eval"]
        self.gamma = config["hyper"]["gamma"]
        self.lr_rate = config["hyper"]["lr_rate"]
        self.memory = ReplayBuffer(10000)

        self.action_size = config["dqn"]["fc"]["action_size"]
        self.policy_net = DQN(config["dqn"]["fc"]["state_size"], config["dqn"]["fc"]["hidden_size"],
                              config["dqn"]["fc"]["action_size"]).to(self.device)
        self.target_net = DQN(config["dqn"]["fc"]["state_size"], config["dqn"]["fc"]["hidden_size"],
                              config["dqn"]["fc"]["action_size"]).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr_rate)
        if config["resume"]:
            self.load_model(config)

    def _state_representation(self, unit_pos, unit_energy, relic_nodes, step, relic_mask):
        """
        Representation of input state to policy net.
        """
        if not relic_mask.any():
            closest_relic = np.array([-1, -1])
        else:
            visible_relics = relic_nodes[relic_mask]
            distances = np.linalg.norm(visible_relics - unit_pos, axis=1)
            closest_relic = visible_relics[np.argmin(distances)]

        state = np.concatenate([
            unit_pos,
            closest_relic,
            [unit_energy],
            [step/505.0]  # Normalize step
        ])
        return torch.FloatTensor(state).to(self.device)

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        """
        Baseline act.
        """
        unit_mask = np.array(obs["units_mask"][self.team_id])
        unit_positions = np.array(obs["units"]["position"][self.team_id])
        unit_energys = np.array(obs["units"]["energy"][self.team_id])
        relic_nodes = np.array(obs["relic_nodes"])
        relic_mask = np.array(obs["relic_nodes_mask"])
        self.score = np.array(obs["team_points"][self.team_id])
        observed_relic_node_positions = np.array(obs["relic_nodes"]) # shape (max_relic_nodes, 2)
        observed_relic_nodes_mask = np.array(obs["relic_nodes_mask"]) # shape (max_relic_nodes, )

        actions = np.zeros((self.env_cfg["max_units"], 3), dtype=int)
        available_units = np.where(unit_mask)[0]

        for unit_id in available_units:
            state = self._state_representation(
                unit_positions[unit_id],
                unit_energys[unit_id],
                relic_nodes,
                step,
                relic_mask
            )

            # action_type = random.randrange(self.action_size)
            self.unit_explore_locations = dict()
            self.relic_node_positions = []
            self.discovered_relic_nodes_ids = set()

            # visible relic nodes
            visible_relic_node_ids = set(np.where(observed_relic_nodes_mask)[0])
            # save any new relic nodes that we discover for the rest of the game.
            for id in visible_relic_node_ids:
                if id not in self.discovered_relic_nodes_ids:
                    self.discovered_relic_nodes_ids.add(id)
                    self.relic_node_positions.append(observed_relic_node_positions[id])

            with torch.no_grad():
                if random.random() < self.epsilon:
                    action_type = np.random.choice(self.action_size)
                else:
                    action_type = self.policy_net(state).argmax().item()

                if action_type == 5:  # Sap action
                    # Find closest enemy unit
                    opp_positions = obs["units"]["position"][self.opp_team_id]
                    opp_mask = obs["units_mask"][self.opp_team_id]
                    valid_targets = []

                    for opp_id, pos in enumerate(opp_positions):
                        if opp_mask[opp_id] and pos[0] != -1:
                            valid_targets.append(pos)

                    if valid_targets:
                        target_pos = valid_targets[0]  # Choose first valid target
                        actions[unit_id] = [5, target_pos[0], target_pos[1]]
                    else:
                        actions[unit_id] = [0, 0, 0]  # Stay if no valid targets
                else:
                    actions[unit_id] = [action_type, 0, 0]

            # print(f"Q-values: {self.policy_net(state)}\tAction: {actions[unit_id]}") ----------------------------

        return actions

    def learn(self, step, last_obs, actions, obs, rewards, dones, player, training=True):
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

        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        loss_name = "loss_0" if player == "player_0" else "loss_1"
        wandb.log({loss_name: loss})

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if step % 100 == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        if step % 504 == 0:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        epsilon_name = "epsilon_0" if player == "player_0" else "epsilon_1"
        wandb.log({epsilon_name: self.epsilon})

    def save_model(self):
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, f'model_pytorch/dqn_model_{self.player}.pth')

    def load_model(self, config: dict):
        try:
            path = config["path_models"] + config["saved_model_0"] if self.player == "player_0" else config["path_models"] + config["saved_model_1"]
            checkpoint = torch.load(path)
            self.policy_net.load_state_dict(checkpoint['policy_net'])
            self.target_net.load_state_dict(checkpoint['target_net'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        except FileNotFoundError:
            raise FileNotFoundError(f"No trained model found for {self.player}")
