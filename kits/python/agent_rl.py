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
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        self.team_id = 0 if self.player == "player_0" else 1
        self.opp_team_id = 1 if self.team_id == 0 else 0
        self.env_cfg = env_cfg

        # DQN parameters
        self.state_size = 6  # unit_pos(2) + closest_relic(2) + unit_energy(1) + step(1)
        self.action_size = 6  # stay, up, right, down, left, sap
        self.hidden_size = 128
        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon = 1.0  # change epsilon if not training
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.0001

        # Initialize networks
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQN(self.state_size, self.hidden_size, self.action_size).to(self.device)
        self.target_net = DQN(self.state_size, self.hidden_size, self.action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.memory = ReplayBuffer(10000)


    def _state_representation(self, unit_pos, unit_energy, relic_nodes, step, relic_mask):
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
        unit_mask = np.array(obs["units_mask"][self.team_id])
        unit_positions = np.array(obs["units"]["position"][self.team_id])
        unit_energys = np.array(obs["units"]["energy"][self.team_id])
        relic_nodes = np.array(obs["relic_nodes"])
        relic_mask = np.array(obs["relic_nodes_mask"])
        self.score = np.array(obs["team_points"][self.team_id])
        observed_relic_node_positions = np.array(obs["relic_nodes"]) # shape (max_relic_nodes, 2)
        observed_relic_nodes_mask = np.array(obs["relic_nodes_mask"]) # shape (max_relic_nodes, )

       # if step % 500 == 0:
          #print(f"memory:  {len(self.memory)}")

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


            # if random.random() < self.epsilon and self.training:
            #     if len(self.relic_node_positions) > 0:
            #         nearest_relic_node_position = self.relic_node_positions[0]
            #         unit_pos = unit_positions[unit_id]
            #         manhattan_distance = abs(unit_pos[0] - nearest_relic_node_position[0]) + abs(unit_pos[1] - nearest_relic_node_position[1])
            #
            #         # if close to the relic node we want to move randomly around it and hope to gain points
            #         if manhattan_distance <= 4:
            #             random_direction = np.random.randint(0, 5)
            #             actions[unit_id] = [random_direction, 0, 0]
            #         else:
            #             # otherwise we want to move towards the relic node
            #             actions[unit_id] = [direction_to(unit_pos, nearest_relic_node_position), 0, 0]
            #     else:
            #         #pick a random location on the map for the unit to explore
            #         unit_pos = unit_positions[unit_id]
            #         rand_loc = (np.random.randint(0, self.env_cfg["map_width"]), np.random.randint(0, self.env_cfg["map_height"]))
            #         self.unit_explore_locations[unit_id] = rand_loc
            #         # using the direction_to tool we can generate a direction that makes the unit move to the saved location
            #         # note that the first index of each unit's action represents the type of action. See specs for more details
            #         actions[unit_id] = [direction_to(unit_pos, self.unit_explore_locations[unit_id]), 0, 0]
            # else:
            with torch.no_grad():
                q_values = self.policy_net(state)
                if random.uniform(0.0,1.0) < self.epsilon:
                    action_type = np.random.choice(self.action_size)
                else:
                    action_type = q_values.argmax().item()

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

            print(f"Q-values: {q_values}\tAction: {actions[unit_id]}")

        return actions

    def learn(self, step, last_obs, actions, obs, rewards, dones, player, training=True):
        if not training or len(self.memory) < self.batch_size:
            return
        rewards = self.score
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

        #print(f"Loss: {loss.item()} Epsilon: {self.epsilon} Score: {rewards} Step: {step}")
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        epsilon_name = "epsilon_0" if player == "player_0" else "epsilon_1"
        wandb.log({epsilon_name: self.epsilon})

    def save_model(self):
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, f'dqn_model_{self.player}.pth')

    def load_model(self):
        try:
            checkpoint = torch.load(f'dqn_model_{self.player}.pth')
            self.policy_net.load_state_dict(checkpoint['policy_net'])
            self.target_net.load_state_dict(checkpoint['target_net'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        except FileNotFoundError:
            raise FileNotFoundError(f"No trained model found for {self.player}")
