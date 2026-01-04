import os
import csv
import random

import vizdoom as vzd
from vizdoom import DoomGame, Mode, GameVariable

N_EPISODES_PER_ARCH = 50

MAX_TICS_PER_EPISODE = 2100

OUTPUT_CSV = "data/episodes_ticks.csv"

SCENARIO_CFG = os.path.join(vzd.scenarios_path, "../configs/deathmatch_vars.cfg")
# SCENARIO_CFG = os.path.join(vzd.scenarios_path, "defend_the_center.cfg")

ARCHETYPES = ["rusher", "survivor", "strafer", "camper"]

game = DoomGame()
game.load_config(SCENARIO_CFG)
game.set_window_visible(False)
game.set_mode(Mode.PLAYER)
game.init()

os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

fieldnames = [
    "episode_global",
    "episode_local",
    "archetype",
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
    "action_move_forward",
    "action_turn_left",
    "action_turn_right",
    "action_attack",
    "is_finished",
]

rows = []

def act_rusher(state, game):
    move_forward = 1
    turn_left = 1 if random.random() < 0.08 else 0
    turn_right = 1 if (turn_left == 0 and random.random() < 0.08) else 0
    attack = 1 if random.random() < 0.8 else 0
    return [move_forward, turn_left, turn_right, attack]


def act_survivor(state, game):
    health = game.get_game_variable(GameVariable.HEALTH)

    if health < 30:
        move_forward = 1 if random.random() < 0.1 else 0
        attack = 1 if random.random() < 0.15 else 0
        turn_left = 1 if random.random() < 0.4 else 0
        turn_right = 1 if (turn_left == 0 and random.random() < 0.4) else 0
    elif health < 60:
        move_forward = 1 if random.random() < 0.4 else 0
        attack = 1 if random.random() < 0.35 else 0
        turn_left = 1 if random.random() < 0.3 else 0
        turn_right = 1 if (turn_left == 0 and random.random() < 0.3) else 0
    else:
        move_forward = 1 if random.random() < 0.6 else 0
        attack = 1 if random.random() < 0.4 else 0
        turn_left = 1 if random.random() < 0.25 else 0
        turn_right = 1 if (turn_left == 0 and random.random() < 0.25) else 0

    return [move_forward, turn_left, turn_right, attack]


def act_strafer(state, game):
    last_reward = game.get_last_reward()

    move_forward = 1 if random.random() < 0.5 else 0

    if last_reward != 0:
        # Якщо щось сталось (хіт, kill, урон) — більше руху й стрільби
        turn_left = 1 if random.random() < 0.5 else 0
        turn_right = 1 if (turn_left == 0 and random.random() < 0.5) else 0
        attack = 1 if random.random() < 0.8 else 0
    else:
        turn_left = 1 if random.random() < 0.35 else 0
        turn_right = 1 if (turn_left == 0 and random.random() < 0.35) else 0
        attack = 1 if random.random() < 0.4 else 0

    return [move_forward, turn_left, turn_right, attack]


def act_camper(state, game):
    tick = game.get_episode_time()

    # Більшість часу стоїть
    move_forward = 1 if random.random() < 0.15 else 0

    # Раз на деякий час — суттєвіший поворот
    if tick % 20 == 0:
        turn_left = 1 if random.random() < 0.5 else 0
        turn_right = 1 if not turn_left else 0
    else:
        turn_left = 1 if random.random() < 0.1 else 0
        turn_right = 1 if (turn_left == 0 and random.random() < 0.1) else 0

    attack = 1 if random.random() < 0.6 else 0

    return [move_forward, turn_left, turn_right, attack]


POLICY_FUNCS = {
    "rusher": act_rusher,
    "survivor": act_survivor,
    "strafer": act_strafer,
    "camper": act_camper,
}

episode_global = 0

for archetype in ARCHETYPES:
    policy_fn = POLICY_FUNCS[archetype]
    print(f"=== Generating for archetype: {archetype} ===")

    for ep_local in range(N_EPISODES_PER_ARCH):
        episode_global += 1
        print(f"[{archetype}] Episode {ep_local + 1}/{N_EPISODES_PER_ARCH} "
              f"(global {episode_global})")

        game.new_episode()
        tick = 0

        while (not game.is_episode_finished()) and (tick < MAX_TICS_PER_EPISODE):
            state = game.get_state()

            action = policy_fn(state, game)
            reward = game.make_action(action)

            health = game.get_game_variable(GameVariable.HEALTH)
            ammo = game.get_game_variable(GameVariable.AMMO2)
            kills = game.get_game_variable(GameVariable.HITCOUNT)

            pos_x = game.get_game_variable(GameVariable.POSITION_X)
            pos_y = game.get_game_variable(GameVariable.POSITION_Y)
            pos_z = game.get_game_variable(GameVariable.POSITION_Z)
            vel_x = game.get_game_variable(GameVariable.VELOCITY_X)
            vel_y = game.get_game_variable(GameVariable.VELOCITY_Y)
            vel_z = game.get_game_variable(GameVariable.VELOCITY_Z)

            rows.append({
                "episode_global": episode_global,
                "episode_local": ep_local,
                "archetype": archetype,
                "tick": tick,
                "reward": float(reward),
                "health": int(health),
                "ammo": int(ammo),
                "killcount": int(kills),
                "position_x": float(pos_x),
                "position_y": float(pos_y),
                "position_z": float(pos_z),
                "velocity_x": float(vel_x),
                "velocity_y": float(vel_y),
                "velocity_z": float(vel_z),
                "action_move_forward": action[0],
                "action_turn_left": action[1],
                "action_turn_right": action[2],
                "action_attack": action[3],
                "is_finished": 0,
            })

            tick += 1

        if tick > 0:
            rows[-1]["is_finished"] = 1

game.close()

with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"Saved {len(rows)} rows to {OUTPUT_CSV}")
print(f"Total episodes: {len(ARCHETYPES) * N_EPISODES_PER_ARCH}")
