import os
import csv

import numpy as np

from stable_baselines3 import PPO

from vizdoom_corridor_env import VizDoomCorridorEnv
from vizdoom import GameVariable



MODELS_DIR = "../models"
OUTPUT_CSV = "data/episodes_ticks_rl.csv"

PERSONAS = ["rusher"]

N_EPISODES_PER_PERSONA = 50
MAX_TICS_PER_EPISODE = 4200


FIELDNAMES = [
    "episode_global",
    "episode_local",
    "persona",
    "tick",
    "reward",
    "health",
    "ammo",
    "killcount",
    "position_x",
    "position_y",
    "position_z",
    "velocity_x",
    "velocity_y",
    "velocity_z",
    "action_index",
    "action_move_left",
    "action_move_right",
    "action_attack",
    "action_move_forward",
    "action_move_backward",
    "action_turn_left",
    "action_turn_right",
    "is_finished",
]


def main():
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    rows = []
    episode_global = 0

    for persona in PERSONAS:
        model_path = os.path.join(MODELS_DIR, f"ppo_deadly_corridor_{persona}.zip")
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file not found for persona '{persona}': {model_path}. "
                f"Train it first with train_rl_personas.py."
            )

        print(f"=== Generating episodes for RL persona: {persona} ===")

        # Для генерації нам не потрібен VecEnv, досить одного env
        env = VizDoomCorridorEnv(persona_type=persona)
        model = PPO.load(model_path, device="auto")

        for ep_local in range(N_EPISODES_PER_PERSONA):
            episode_global += 1
            print(
                f"[{persona}] Episode {ep_local + 1}/{N_EPISODES_PER_PERSONA} "
                f"(global {episode_global})"
            )

            obs = env.reset()
            done = False
            tick = 0

            while (not done) and (tick < MAX_TICS_PER_EPISODE):
                action_idx, _ = model.predict(obs, deterministic=False)
                action_idx = int(action_idx)

                obs, reward, done, info = env.step(action_idx)

                game = env.game

                health = int(game.get_game_variable(GameVariable.HEALTH))
                ammo = 0
                kills = 0

                pos_x = float(game.get_game_variable(GameVariable.POSITION_X))
                pos_y = float(game.get_game_variable(GameVariable.POSITION_Y))
                pos_z = float(game.get_game_variable(GameVariable.POSITION_Z))
                vel_x = float(game.get_game_variable(GameVariable.VELOCITY_X))
                vel_y = float(game.get_game_variable(GameVariable.VELOCITY_Y))
                vel_z = float(game.get_game_variable(GameVariable.VELOCITY_Z))

                action_vector = env.actions[action_idx]
                rows.append(
                    {
                        "episode_global": episode_global,
                        "episode_local": ep_local,
                        "persona": persona,
                        "tick": tick,
                        "reward": float(reward),
                        "health": health,
                        "ammo": ammo,
                        "killcount": kills,
                        "position_x": pos_x,
                        "position_y": pos_y,
                        "position_z": pos_z,
                        "velocity_x": vel_x,
                        "velocity_y": vel_y,
                        "velocity_z": vel_z,
                        "action_index": action_idx,
                        "action_move_left": action_vector[0],
                        "action_move_right": action_vector[1],
                        "action_attack": action_vector[2],
                        "action_move_forward": action_vector[3],
                        "action_move_backward": action_vector[4],
                        "action_turn_left": action_vector[5],
                        "action_turn_right": action_vector[6],
                        "is_finished": 0,
                    }
                )

                tick += 1

            if tick > 0:
                rows[-1]["is_finished"] = 1

        env.close()

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[OK] Saved {len(rows)} rows to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
