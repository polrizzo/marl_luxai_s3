import json
from datetime import datetime
import flax.serialization
import jax
from click import option

import wandb
import random

from luxai_s3 import LuxAIS3GymEnv
from luxai_s3.state import EnvState, serialize_env_actions, serialize_env_states
from luxai_s3.env import LuxAIS3Env
from luxai_s3.wrappers import RecordEpisode, LuxAIS3GymEnv
from params_custom import EnvParamsCustom
from agent_rl import AgentRl
from reward import get_reward

if __name__ == "__main__":

    env = LuxAIS3GymEnv(numpy_output=True)
    env = RecordEpisode(env, save_dir="./replays")
    # seed = random.randint(0, 100000)
    seed = 0
    option = dict(params=EnvParamsCustom())
    obs, info = env.reset(seed=seed, options=option)

    now_str = datetime.now().strftime("%Y-%m-%d_%H:%M")
    name_test = "local__" + now_str

    wandb.init(
        entity="polrizzo",
        project="LuxAI_S3",
        dir="./",
        # id: (str | None) = None,
        name=name_test,
        config= info["full_params"],
        # group: (str | None) = None,
        # job_type: (str | None) = None,
        # reinit: (bool | None) = None,
        # resume: (bool | Literal['allow', 'never', 'must', 'auto'] | None) = None,
        # resume_from: (str | None) = None,
        # fork_from: (str | None) = None,
        # save_code: (bool | None) = None,
        # settings: (Settings | dict[str, Any] | None) = None
    )

    # Initialize Agents
    player_0 = AgentRl("player_0", info["params"])
    player_1 = AgentRl("player_1", info["params"])
    training = True
    print("Starting Training") if training else print("Starting Testing")

    for i in range(info["params"]["match_count_per_episode"]):
        # reset at each match
        seed = 0
        option = {"params": EnvParamsCustom()}
        obs, info = env.reset(seed=seed, options=option)
        env_cfg = info["params"]

        game_done = False
        step = 0
        last_obs = None
        last_actions = None
        while not game_done:
            print(f"Game:{i} Step:{step}")

            actions = {}

            # Store current observation for learning
            if training:
                last_obs = {
                    "player_0": obs["player_0"].copy(),
                    "player_1": obs["player_1"].copy()
                }

            # Get actions
            for agent in [player_0, player_1]:
                print(agent.player)
                actions[agent.player] = agent.act(step=step, obs=obs[agent.player])

            if training:
                last_actions = actions.copy()

            # Environment step
            obs, rewards, terminated, truncated, info = env.step(actions)
            dones = {k: terminated[k] | truncated[k] for k in terminated}
            rewards = {
                "player_0": get_reward(type_reward="only_points", obs=obs["player_0"], player=0),
                "player_1": get_reward(type_reward="points_exploration", obs=obs["player_1"], player=1)
            }
            wandb.log({"reward_0": rewards["player_0"], "reward_1": rewards["player_1"]})
            if step > 70:  # debug purpose
                pass
            print(f"rewards: {rewards}")
            # Store experiences and learn
            if training and last_obs is not None:
                # Store experience for each unit
                for agent in [player_0, player_1]:
                    for unit_id in range(env_cfg["max_units"]):
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
                               obs["player_0"], rewards["player_0"], dones["player_0"], "player_0", training)
                player_1.learn(step, last_obs["player_1"], actions["player_1"],
                               obs["player_1"], rewards["player_1"], dones["player_1"], "player_1", training)

            if dones["player_0"] or dones["player_1"]:
                game_done = True
                if training:
                    player_0.save_model()
                    player_1.save_model()

            step += 1

    env.close()
