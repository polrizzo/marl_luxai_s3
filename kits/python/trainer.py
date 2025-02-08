import random
from datetime import datetime
import wandb
import yaml
import numpy as np

from kits.python.state_custom import update_single_unit_energy, global_state
from luxai_s3.wrappers import RecordEpisode, LuxAIS3GymEnv
from agent_rl import AgentRl
from reward import get_reward

if __name__ == "__main__":
    # Env settings
    env = LuxAIS3GymEnv(numpy_output=True)
    seed = random.randint(0, 1000000000)
    obs, info = env.reset(seed=seed)
    max_units = info["params"]["max_units"]
    max_step = info["params"]["max_steps_in_match"]
    match_per_episode = info["params"]["match_count_per_episode"]

    # Agents settings
    with open("config_trainer.yaml", "r") as stream:
        config_trainer = yaml.safe_load(stream)
    player_0 = AgentRl("player_0", info["params"])
    player_0.build_model(config_trainer)
    player_1 = AgentRl("player_1", info["params"])
    player_1.build_model(config_trainer)

    now_str = datetime.now().strftime("%Y-%m-%d_%H:%M")
    name_test = "local__" + now_str if config_trainer["local"] == "true" else "remote__" + now_str

    # Wandb settings
    wandb.init(
        entity="polrizzo",
        project="LuxAI_S3",
        dir="./",
        name=name_test,
        config=config_trainer,
        group= config_trainer["type_policy"],
        job_type= "training" if config_trainer["training"] else "testing",
        # id: (str | None) = None, # settings: (Settings | dict[str, Any] | None) = None
        # reinit: (bool | None) = None,
        # resume: (bool | Literal['allow', 'never', 'must', 'auto'] | None) = None,
        # resume_from: (str | None) = None, # fork_from: (str | None) = None, # save_code: (bool | None) = None,
    )
    wandb.define_metric("total_games")
    wandb.define_metric("seed", step_metric="total_games")
    wandb.define_metric("winner", step_metric="total_games")
    wandb.define_metric("step_total")
    wandb.define_metric("epsilon_0", step_metric="step_total")
    wandb.define_metric("epsilon_1", step_metric="step_total")
    wandb.define_metric("loss_0", step_metric="step_total")
    wandb.define_metric("loss_1", step_metric="step_total")
    wandb.define_metric("points_0", step_metric="step_total")
    wandb.define_metric("points_1", step_metric="step_total")
    wandb.define_metric("reward_0", step_metric="step_total")
    wandb.define_metric("reward_1", step_metric="step_total")

    print("Starting Training") if config_trainer["training"] else print("Starting Testing")
    step_total = 0
    # TRAINING -------------------------------------------------
    # for i in range(config_trainer["hyper"]["num_games"]):
    for i in range(1):
        step = 0
        game_done = False
        # Setup obs variables
        last_obs_global = {
            "player_0": None,
            "player_1": None
        }
        last_obs = {
            "player_0": [None]*max_units,
            "player_1": [None]*max_units
        }
        last_active_units = {
            "player_0": np.zeros(max_units, dtype=bool),
            "player_1": np.zeros(max_units, dtype=bool),
        }
        last_actions = {
            "player_0": np.zeros([max_units, 3]),
            "player_1": np.zeros([max_units, 3])
        }
        last_points = np.array([0, 0])
        next_obs_global = {
            "player_0": None,
            "player_1": None
        }
        # Setup env
        seed = random.randint(0, 1000000000)
        obs, info = env.reset(seed=seed)
        player_0.update_env_cfg(info["params"])
        player_1.update_env_cfg(info["params"])
        # Store global state for each agent
        for agent in [player_0, player_1]:
            agent.state_representation(obs[agent.player])
            last_obs_global[agent.player] = agent.get_global_state()

        while not game_done:
            actions = {}

            # Store current state + action
            for agent in [player_0, player_1]:
                # get actions
                actions[agent.player] = agent.act(step=step, obs=obs[agent.player])
                # store signle unit's state
                for unit_id in range(max_units):
                    if obs[agent.player]["units_mask"][agent.team_id][unit_id]:
                        energy = obs[agent.player]["units"]["energy"][agent.team_id, unit_id, 0]
                        y_pos = obs[agent.player]["units"]["position"][agent.team_id, unit_id, 0]
                        x_pos = obs[agent.player]["units"]["position"][agent.team_id, unit_id, 1]
                        # get and store single unit's state
                        single_player_state = update_single_unit_energy(last_obs_global[agent.player].copy(), energy, y_pos, x_pos)
                        last_obs[agent.player][unit_id] = single_player_state.copy()
                        last_actions[agent.player][unit_id, 0] = actions[agent.player][unit_id, 0]
                        last_actions[agent.player][unit_id, 1] = actions[agent.player][unit_id, 1]
                        last_actions[agent.player][unit_id, 2] = actions[agent.player][unit_id, 2]
                        # set if the unit was active
                        last_active_units[agent.player][unit_id] = True
                    else:
                        last_actions[agent.player][unit_id, 0] = 0
                        last_actions[agent.player][unit_id, 1] = 0
                        last_actions[agent.player][unit_id, 2] = 0
                        last_active_units[agent.player][unit_id] = False

            # Environment step + reward log
            obs, rewards, terminated, truncated, info = env.step(actions)
            dones = {k: terminated[k] | truncated[k] for k in terminated}
            rewards = {
                "player_0": get_reward(type_reward="delta_points_exploration", obs=obs["player_0"], player=0, last_points=last_points),
                "player_1": get_reward(type_reward="points_exploration", obs=obs["player_1"], player=1, last_points=last_points)
            }
            wandb.log({"step_total": step_total,
                       "points_0": obs["player_0"]["team_points"][0], "points_1": obs["player_0"]["team_points"][1],
                       "reward_0": rewards["player_0"], "reward_1": rewards["player_1"]})
            # update last points (if first step of match, set [0,0])
            if (step + 2) % 101 == 0:
                last_points = np.array([0, 0])
            else:
                last_points = obs["player_0"]["team_points"]

            # Store experience for each player's unit and learn
            for agent in [player_0, player_1]:
                agent.state_representation(obs[agent.player])
                next_obs_global[agent.player] = agent.get_global_state()
                for unit_id in range(max_units):
                    if last_active_units[agent.player][unit_id]:
                        energy = obs[agent.player]["units"]["energy"][agent.team_id, unit_id, 0]
                        y_pos = obs[agent.player]["units"]["position"][agent.team_id, unit_id, 0]
                        x_pos = obs[agent.player]["units"]["position"][agent.team_id, unit_id, 1]
                        # get and store single unit's state
                        next_single_obs = update_single_unit_energy(next_obs_global[agent.player].copy(), energy,
                                                                        y_pos, x_pos)
                        # push to buffer
                        agent.memory.push(
                            last_obs[agent.player][unit_id],
                            last_actions[agent.player][unit_id][0],
                            rewards[agent.player],
                            next_single_obs,
                            dones[agent.player]
                        )
                last_obs_global[agent.player] = next_obs_global[agent.player].copy()
                next_obs_global[agent.player] = None

            if dones["player_0"] or dones["player_1"]:
                game_done = True
                # Learn from experiences at the end of the game
                player_0.learn(step=step, player="player_0", training=config_trainer["training"])
                player_1.learn(step=step, player="player_1", training=config_trainer["training"])
                # if config_trainer["training"]:
                #     player_0.save_model()
                #     player_1.save_model()

            step += 1
            step_total += 1
        winner = obs["player_0"]["team_wins"][0] - obs["player_0"]["team_wins"][1]
        wandb.log({"total_games": i, "seed": seed,
                   "winner": 1 if winner > 0 else -1})
        # Update target_net every (num_games/update_tn)
        if (i+1) % (config_trainer["hyper"]['num_games'] // config_trainer["hyper"]['update_tn']) == 0:
            player_0.update_target_net()
            player_1.update_target_net()

        # EVALUATION -------------------------------------------------
        # # Eval phase every 1/10 of num_games
        # if (i + 1) % (config_trainer["hyper"]['num_games'] // config_trainer["hyper"]['eval']) == 0:
        # # if i  == 0:
        #     game_done = False
        #     last_obs = None
        #     last_actions = None
        #     last_points = np.array([0, 0])
        #     # Setup env
        #     seed = random.randint(0, 1000000000)
        #     env_eval = RecordEpisode(env=env, save_dir="./replays/", save_on_reset=False, save_on_close=False)
        #     obs, info = env_eval.reset(seed=seed)
        #     player_0.update_env_cfg(info["params"])
        #     player_1.update_env_cfg(info["params"])
        #
        #     while not game_done:
        #         actions = {}
        #
        #         # Store current observation for learning
        #         last_obs = {
        #             "player_0": obs["player_0"].copy(),
        #             "player_1": obs["player_1"].copy()
        #         }
        #
        #         # Get actions + store current actions for learning
        #         for agent in [player_0, player_1]:
        #             actions[agent.player] = agent.predict(step=step, obs=obs[agent.player])
        #         last_actions = actions.copy()
        #
        #         # Environment step + reward log
        #         obs, rewards, terminated, truncated, info = env_eval.step(actions)
        #         dones = {k: terminated[k] | truncated[k] for k in terminated}
        #         rewards = {
        #             "player_0": get_reward(type_reward="delta_points_exploration", obs=obs["player_0"], player=0,
        #                                    last_points=last_points),
        #             "player_1": get_reward(type_reward="points_exploration", obs=obs["player_1"], player=1,
        #                                    last_points=last_points)
        #         }
        #
        #         # update last points (if first step of match, set [0,0])
        #         if (step + 2) % 101 == 0:
        #             last_points = np.array([0, 0])
        #         else:
        #             last_points = obs["player_0"]["team_points"]
        #
        #         if dones["player_0"] or dones["player_1"]:
        #             game_done = True
        #             print(obs["player_0"]["team_wins"])
        #     game_num = (i + 1) // (config_trainer["hyper"]['num_games'] // config_trainer["hyper"]['eval'])
        #     name_eval = "./replays/game_" + str(game_num) + ".json"
        #     env_eval.save_episode(save_path=name_eval)
        #     env_eval.close()

    player_0.save_model()
    player_1.save_model()
    env.close()
