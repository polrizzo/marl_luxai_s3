import random
import os
from datetime import datetime
import wandb
import yaml
import numpy as np

from luxai_s3.wrappers import RecordEpisode, LuxAIS3GymEnv
from agent_rl import AgentRl
from reward import get_reward

if __name__ == "__main__":
    now_str = datetime.now().strftime("%Y-%m-%d_%H:%M")
    name_test = "local__" + now_str

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

    # Wandb settings
    wandb.init(
        entity="polrizzo",
        project="LuxAI_S3",
        dir="./",
        name=name_test,
        config=config_trainer,
        group= config_trainer["type_policy"],
        job_type= "training" if config_trainer["training"] == "true" else "testing",
        # id: (str | None) = None, # settings: (Settings | dict[str, Any] | None) = None
        # reinit: (bool | None) = None,
        # resume: (bool | Literal['allow', 'never', 'must', 'auto'] | None) = None,
        # resume_from: (str | None) = None, # fork_from: (str | None) = None, # save_code: (bool | None) = None,
    )
    wandb.define_metric("total_games")
    wandb.define_metric("seed", step_metric="total_games")
    wandb.define_metric("winner_final", step_metric="total_games")
    wandb.define_metric("step_total")
    wandb.define_metric("step", step_metric="step_total")
    wandb.define_metric("epsilon_0", step_metric="step_total")
    wandb.define_metric("epsilon_1", step_metric="step_total")
    wandb.define_metric("loss_0", step_metric="step_total")
    wandb.define_metric("loss_1", step_metric="step_total")
    wandb.define_metric("points_0", step_metric="step_total")
    wandb.define_metric("points_1", step_metric="step_total")
    wandb.define_metric("reward_0", step_metric="step_total")
    wandb.define_metric("reward_1", step_metric="step_total")

    print("Starting Training") if config_trainer["training"] else print("Starting Testing")
    # for i in range(config_trainer["num_games"]):
    for i in range(1):
        step = 0
        step_total = 0
        game_done = False
        last_obs = None
        last_actions = None
        last_points = np.array([0, 0])
        # Setup env
        seed = random.randint(0, 1000000000)
        obs, info = env.reset(seed=seed)
        player_0.update_env_cfg(info["params"])
        player_1.update_env_cfg(info["params"])

        while not game_done:
            actions = {}

            # Store current observation for learning
            last_obs = {
                "player_0": obs["player_0"].copy(),
                "player_1": obs["player_1"].copy()
            }

            # Get actions + store current actions for learning
            for agent in [player_0, player_1]:
                actions[agent.player] = agent.act(step=step, obs=obs[agent.player])
            last_actions = actions.copy()

            # Environment step + reward log
            obs, rewards, terminated, truncated, info = env.step(actions)
            dones = {k: terminated[k] | truncated[k] for k in terminated}
            rewards = {
                "player_0": get_reward(type_reward="delta_points_exploration", obs=obs["player_0"], player=0, last_points=last_points),
                "player_1": get_reward(type_reward="points_exploration", obs=obs["player_1"], player=1, last_points=last_points)
            }
            wandb.log({"step": step, "step_total": step_total,
                       "points_0": obs["player_0"]["team_points"][0], "points_1": obs["player_0"]["team_points"][1],
                       "reward_0": rewards["player_0"], "reward_1": rewards["player_1"]})
            # update last points (if first step of match, set [0,0])
            if (step + 2) % 101 == 0:
                last_points = np.array([0, 0])
            else:
                last_points = obs["player_0"]["team_points"]

            # Store experience for each player's unit and learn
            for agent in [player_0, player_1]:
                for unit_id in range(max_units):
                    if obs[agent.player]["units_mask"][agent.team_id][unit_id]:
                        current_state = agent._state_representation(
                            last_obs[agent.player]["units"]["position"][agent.team_id][unit_id],
                            last_obs[agent.player]["units"]["energy"][agent.team_id][unit_id],
                            last_obs[agent.player]["relic_nodes"],
                            step,
                            last_obs[agent.player]["relic_nodes_mask"]
                        )

                        next_state = agent._state_representation(
                            obs[agent.player]["units"]["position"][agent.team_id][unit_id],
                            obs[agent.player]["units"]["energy"][agent.team_id][unit_id],
                            obs[agent.player]["relic_nodes"],
                            step + 1,
                            obs[agent.player]["relic_nodes_mask"]
                        )

                        agent.memory.push(
                            current_state,
                            last_actions[agent.player][unit_id][0],
                            rewards[agent.player],
                            next_state,
                            dones[agent.player]
                        )

                # Learn from experiences
                player_0.learn(step, last_obs["player_0"], actions["player_0"],
                               obs["player_0"], rewards["player_0"], dones["player_0"], "player_0", config_trainer["training"])
                player_1.learn(step, last_obs["player_1"], actions["player_1"],
                               obs["player_1"], rewards["player_1"], dones["player_1"], "player_1", config_trainer["training"])

            if dones["player_0"] or dones["player_1"]:
                game_done = True
                # if config_trainer["training"]:
                #     player_0.save_model()
                #     player_1.save_model()

            step += 1
            step_total += 1
        winner = obs["player_0"]["team_wins"][0] - obs["player_0"]["team_wins"][1]
        wandb.log({"total_games": i, "seed": seed,
                   "winner": 1 if winner > 0 else -1})

        # Eval phase every 200 games
        # if (i + 1) % 200 == 0:
        if i  == 0:
            game_done = False
            last_obs = None
            last_actions = None
            last_points = np.array([0, 0])
            # Setup env
            seed = random.randint(0, 1000000000)
            env_eval = RecordEpisode(env=env, save_dir="./replays/", save_on_reset=False, save_on_close=False)
            obs, info = env_eval.reset(seed=seed)
            player_0.update_env_cfg(info["params"])
            player_1.update_env_cfg(info["params"])

            while not game_done:
                actions = {}

                # Store current observation for learning
                last_obs = {
                    "player_0": obs["player_0"].copy(),
                    "player_1": obs["player_1"].copy()
                }

                # Get actions + store current actions for learning
                for agent in [player_0, player_1]:
                    actions[agent.player] = agent.predict(step=step, obs=obs[agent.player])
                last_actions = actions.copy()

                # Environment step + reward log
                obs, rewards, terminated, truncated, info = env_eval.step(actions)
                dones = {k: terminated[k] | truncated[k] for k in terminated}
                rewards = {
                    "player_0": get_reward(type_reward="delta_points_exploration", obs=obs["player_0"], player=0,
                                           last_points=last_points),
                    "player_1": get_reward(type_reward="points_exploration", obs=obs["player_1"], player=1,
                                           last_points=last_points)
                }

                # update last points (if first step of match, set [0,0])
                if (step + 2) % 101 == 0:
                    last_points = np.array([0, 0])
                else:
                    last_points = obs["player_0"]["team_points"]

                if dones["player_0"] or dones["player_1"]:
                    game_done = True
                    print(obs["player_0"]["team_wins"])
            name_eval = "./replays/game_" + str((i + 1) // 200) + ".json"
            env_eval.save_episode(save_path=name_eval)

    env.close()
