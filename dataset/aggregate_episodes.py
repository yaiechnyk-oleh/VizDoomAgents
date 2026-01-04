import os
import numpy as np
import pandas as pd

INPUT_CSV = "data/episodes_ticks_rl.csv"
OUTPUT_CSV = "data/episodes_aggregated_rl.csv"

TICRATE = 35.0


def turn_ratio(sub: pd.DataFrame) -> float:
    turns = (sub["action_turn_left"] > 0) | (sub["action_turn_right"] > 0)
    return float(turns.mean())


def move_ratio(sub: pd.DataFrame) -> float:
    moves = (
        (sub["action_move_forward"] > 0)
        | (sub["action_move_backward"] > 0)
        | (sub["action_move_left"] > 0)
        | (sub["action_move_right"] > 0)
    )
    return float(moves.mean())


def avg_speed(sub: pd.DataFrame) -> float:
    vx = sub["velocity_x"].to_numpy(dtype=float)
    vy = sub["velocity_y"].to_numpy(dtype=float)
    vz = sub["velocity_z"].to_numpy(dtype=float)
    speed = np.sqrt(vx ** 2 + vy ** 2 + vz ** 2)
    return float(speed.mean())


def main():
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

    print(f"Reading tick-level data from {INPUT_CSV} ...")
    df = pd.read_csv(INPUT_CSV)

    group_cols = ["episode_global", "persona"]
    g = df.groupby(group_cols)

    print("Aggregating per-episode statistics ...")

    agg = g.agg(
        duration_ticks=("tick", lambda s: int(s.max() + 1)),
        reward_sum=("reward", "sum"),
        health_start=("health", "first"),
        health_end=("health", "last"),
        shots_fired=("action_attack", "sum"),
        # kills тут поки що 0, якщо не логиш HITCOUNT
        kills=("killcount", "max"),
    )

    agg["duration_sec"] = agg["duration_ticks"] / TICRATE
    agg["turn_ratio"] = g.apply(turn_ratio)
    agg["move_ratio"] = g.apply(move_ratio)
    agg["avg_speed"] = g.apply(avg_speed)

    # "М’який" win: вважаємо успіхом епізоди з reward_sum вище медіани
    median_reward = agg["reward_sum"].median()
    agg["win_soft"] = (agg["reward_sum"] > median_reward).astype(int)

    agg = agg.reset_index()

    print(f"Saving episode-level dataset to {OUTPUT_CSV} ...")
    agg.to_csv(OUTPUT_CSV, index=False)

    print("Done. Preview:")
    print(agg.head())


if __name__ == "__main__":
    main()
