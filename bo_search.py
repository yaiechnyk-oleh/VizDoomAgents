"""
Bayesian Optimization search for optimal rusher reward weights.

Usage:
    python bo_search.py --cfg configs/deathmatch_vars.cfg

Each BO iteration:
    1. BO proposes a set of reward weights
    2. Train a fresh agent for --timesteps steps
    3. Eval for --eval_episodes episodes (deterministic)
    4. Compute rusher_score from game metrics
    5. Report score to BO → repeat

Results are logged to bo_results/ directory.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import numpy as np
from bayes_opt import BayesianOptimization

from sb3_contrib.ppo_recurrent import RecurrentPPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

from env import make_env
from cnn_gru import CnnGruPolicy, CnnStateExtractor  # noqa: F401
from callback import (
    ActionDiagnosticsCallback,
    EntropyAnnealCallback,
    PeriodicEvalSaveCallback,
    InfoStatsCallback,
    RewardHackAlertCallback,
    EpisodeEndDiagnosticsCallback,
    GoalModeStatsCallback,
)


# ── BO Search Space ──────────────────────────────────────────────
SEARCH_SPACE = {
    "enemy_dist":                    (0.03, 0.15),
    "backward_while_enemy_penalty":  (0.0,  0.10),
    "death":                         (8.0,  20.0),
    "weapon_situational_bonus":      (0.05, 0.40),
    "blind_fire_penalty":            (0.0,  0.04),
}


# ── Objective Function ───────────────────────────────────────────
def rusher_score(
    kills_attrib: float,
    damage_dealt: float,
    hits: float,
    hits_taken: float,
    damage_taken: float,
    deaths: float,
    action_top_frac: float,
) -> float:
    """
    Composite score: higher = better rusher.
    Uses ONLY game metrics, NOT shaped reward (avoiding circularity).
    """
    # Hit efficiency: what fraction of total hits are ours
    hit_eff = hits / max(1.0, hits + hits_taken)

    # Damage efficiency: trade ratio
    dmg_eff = damage_dealt / max(1.0, damage_dealt + damage_taken)

    # Action diversity penalty: humans don't press one button 81% of the time
    # If top action > 60%, penalize proportionally
    diversity_penalty = max(0.0, action_top_frac - 0.60) * 30.0

    score = (
        kills_attrib * 15.0       # killing is the goal
        + damage_dealt * 0.05     # raw combat output
        + hits * 0.5             # landing shots
        + hit_eff * 20.0         # accuracy = not spraying
        + dmg_eff * 15.0         # good trade ratio
        - deaths * 5.0           # survive but don't camp
        - diversity_penalty      # don't be a one-action bot
    )
    return score


# ── Train + Eval Pipeline ────────────────────────────────────────
def train_and_eval(
    weight_overrides: Dict[str, float],
    cfg: str,
    persona: str,
    timesteps: int,
    eval_episodes: int,
    n_envs: int,
    seed: int,
    iteration: int,
    out_dir: str,
    resume_path: str = "",
) -> float:
    """Run one full train → eval cycle. Returns rusher_score."""

    run_id = int(time.time())
    print("\n" + "=" * 70)
    print(f"BO ITERATION {iteration}")
    print(f"Weights: {json.dumps(weight_overrides, indent=2)}")
    print("=" * 70)

    # ── Create envs with overridden weights ──
    env_fns = []
    for i in range(n_envs):
        env_fns.append(
            make_env(
                cfg=cfg,
                frame_skip=4,
                max_steps=4200,
                seed=seed + i * 1000,
                render=False,
                persona=persona,
                weight_overrides=weight_overrides,
            )
        )
    train_env = SubprocVecEnv(env_fns)
    train_env = VecMonitor(train_env)

    eval_env = DummyVecEnv([
        make_env(
            cfg=cfg,
            frame_skip=4,
            max_steps=4200,
            seed=seed + 99999,
            render=False,
            persona=persona,
            weight_overrides=weight_overrides,
        )
    ])
    eval_env = VecMonitor(eval_env)

    # ── Create model ──
    best_path = os.path.join(out_dir, f"bo_iter_{iteration}_best.zip")

    if resume_path and os.path.exists(resume_path):
        print(f"Resuming from baseline model: {resume_path}")
        model = RecurrentPPO.load(
            resume_path,
            env=train_env,
            learning_rate=1e-4,
            device="auto",
        )
        model.ent_coef = 0.04
    else:
        policy_kwargs = dict(
            features_extractor_class=CnnStateExtractor,
            features_extractor_kwargs=dict(features_dim=256),
            lstm_hidden_size=256,
            n_lstm_layers=1,
            shared_lstm=True,
            enable_critic_lstm=False,
        )

        model = RecurrentPPO(
            policy=CnnGruPolicy,
            env=train_env,
            learning_rate=1e-4,
            n_steps=128,
            batch_size=256,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.1,
            ent_coef=0.04,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=policy_kwargs,
            device="auto",
            verbose=0,
            seed=seed,
        )

    # ── Train ──
    callbacks = CallbackList([
        EntropyAnnealCallback(ent_start=0.04, ent_end=0.02, total_timesteps=timesteps),
        PeriodicEvalSaveCallback(
            eval_env=eval_env,
            eval_freq=max(timesteps // 5, 10000),
            n_eval_episodes=5,
            best_path=best_path,
            deterministic=False,
            verbose=0,
        ),
    ])

    print(f"Training for {timesteps} steps...")
    t0 = time.time()
    model.learn(total_timesteps=timesteps, callback=callbacks)
    train_time = time.time() - t0
    print(f"Training done in {train_time:.0f}s ({train_time/3600:.1f}h)")

    # Save last model
    last_path = os.path.join(out_dir, f"bo_iter_{iteration}_last.zip")
    model.save(last_path)

    train_env.close()
    eval_env.close()

    # ── Eval using best checkpoint ──
    eval_model_path = best_path if os.path.exists(best_path) else last_path
    print(f"Evaluating {eval_model_path} for {eval_episodes} episodes...")

    eval_env2 = DummyVecEnv([
        make_env(
            cfg=cfg,
            frame_skip=4,
            max_steps=4200,
            seed=seed + 77777,
            render=False,
            persona=persona,
            weight_overrides=weight_overrides,
        )
    ])
    eval_env2 = VecMonitor(eval_env2)

    eval_model = RecurrentPPO.load(eval_model_path, device="auto")

    # Run episodes
    all_kills = []
    all_damage = []
    all_hits = []
    all_hits_taken = []
    all_damage_taken = []
    all_deaths = []
    all_rewards = []
    action_counts: Counter = Counter()
    total_steps = 0

    for ep in range(eval_episodes):
        obs = eval_env2.reset()
        lstm_states = None
        episode_starts = np.ones((1,), dtype=bool)
        ep_reward = 0.0
        done = False
        last_info = {}

        while not done:
            action, lstm_states = eval_model.predict(
                obs, state=lstm_states,
                episode_start=episode_starts,
                deterministic=True,
            )
            obs, reward, dones, infos = eval_env2.step(action)
            ep_reward += float(reward[0])
            action_counts[int(action[0])] += 1
            total_steps += 1
            episode_starts = dones
            last_info = infos[0]
            done = bool(dones[0])

        all_rewards.append(ep_reward)
        all_kills.append(float(last_info.get("ep_monster_kills", 0.0)))
        all_damage.append(float(last_info.get("ep_damage_dealt", last_info.get("damagecount", 0.0))))
        all_hits.append(float(last_info.get("ep_hits", last_info.get("hitcount", 0.0))))
        all_hits_taken.append(float(last_info.get("ep_hits_taken", last_info.get("hits_taken", 0.0))))
        all_damage_taken.append(float(last_info.get("ep_damage_taken", last_info.get("damage_taken", 0.0))))
        all_deaths.append(float(last_info.get("ep_deaths", last_info.get("deathcount", 0.0))))

    eval_env2.close()

    # Compute action diversity
    if total_steps > 0 and action_counts:
        top_action_count = action_counts.most_common(1)[0][1]
        action_top_frac = top_action_count / total_steps
    else:
        action_top_frac = 1.0

    mean = lambda xs: float(np.mean(xs)) if xs else 0.0

    mk = mean(all_kills)
    dmg = mean(all_damage)
    h = mean(all_hits)
    ht = mean(all_hits_taken)
    dt = mean(all_damage_taken)
    d = mean(all_deaths)
    mr = mean(all_rewards)

    score = rusher_score(mk, dmg, h, ht, dt, d, action_top_frac)

    print(f"\n--- BO ITER {iteration} RESULTS ---")
    print(f"mean_reward={mr:.2f}  kills={mk:.2f}  dmg={dmg:.1f}  hits={h:.1f}")
    print(f"hits_taken={ht:.1f}  dmg_taken={dt:.1f}  deaths={d:.2f}")
    print(f"action_top_frac={action_top_frac:.3f}  unique_actions={len(action_counts)}")
    print(f"RUSHER SCORE = {score:.3f}")
    print(f"Weights: {weight_overrides}")
    print("-" * 40)

    # Log to CSV
    csv_path = os.path.join(out_dir, "bo_results.csv")
    row = {
        "iteration": iteration,
        "score": score,
        "mean_reward": mr,
        "kills_attrib": mk,
        "damage_dealt": dmg,
        "hits": h,
        "hits_taken": ht,
        "damage_taken": dt,
        "deaths": d,
        "action_top_frac": action_top_frac,
        "unique_actions": len(action_counts),
        "train_time_s": train_time,
        **weight_overrides,
    }
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)

    return score


# ── Main ─────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description="Bayesian Optimization for reward weights")
    p.add_argument("--cfg", type=str, required=True)
    p.add_argument("--persona", type=str, default="rusher")
    p.add_argument("--timesteps", type=int, default=500000,
                   help="Training steps per BO iteration")
    p.add_argument("--eval_episodes", type=int, default=10)
    p.add_argument("--n_envs", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_init", type=int, default=5,
                   help="Random exploration iterations")
    p.add_argument("--n_iter", type=int, default=10,
                   help="BO exploitation iterations")
    p.add_argument("--out_dir", type=str, default="bo_results")
    p.add_argument("--resume", type=str, default="",
                   help="Path to a baseline model to resume from in each iteration")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    iteration_counter = {"n": 0}

    def objective(**kwargs):
        iteration_counter["n"] += 1
        return train_and_eval(
            weight_overrides=kwargs,
            cfg=args.cfg,
            persona=args.persona,
            timesteps=args.timesteps,
            eval_episodes=args.eval_episodes,
            n_envs=args.n_envs,
            seed=args.seed,
            iteration=iteration_counter["n"],
            out_dir=args.out_dir,
            resume_path=args.resume,
        )

    optimizer = BayesianOptimization(
        f=objective,
        pbounds=SEARCH_SPACE,
        random_state=args.seed,
        verbose=2,
    )


    print("\n" + "=" * 70)
    print("BAYESIAN OPTIMIZATION SEARCH")
    print(f"Search space: {json.dumps({k: list(v) for k, v in SEARCH_SPACE.items()}, indent=2)}")
    print(f"Init points: {args.n_init} | BO iterations: {args.n_iter}")
    print(f"Steps per iteration: {args.timesteps}")
    print(f"Eval episodes: {args.eval_episodes}")
    print(f"Estimated time: {(args.n_init + args.n_iter) * args.timesteps / 35 / 3600:.1f} hours")
    print("=" * 70)

    optimizer.maximize(
        init_points=args.n_init,
        n_iter=args.n_iter,
    )

    # Print results
    print("\n" + "=" * 70)
    print("BO SEARCH COMPLETE")
    print("=" * 70)
    print(f"\nBest score: {optimizer.max['target']:.3f}")
    print(f"Best weights: {json.dumps(optimizer.max['params'], indent=2)}")

    # Save best weights
    best_path = os.path.join(args.out_dir, "best_weights.json")
    with open(best_path, "w") as f:
        json.dump({
            "score": optimizer.max["target"],
            "weights": optimizer.max["params"],
        }, f, indent=2)
    print(f"\nSaved best weights to {best_path}")

    # Print all results sorted by score
    print("\nAll iterations (sorted by score):")
    print(f"{'Iter':>4}  {'Score':>8}  Weights")
    for i, res in enumerate(sorted(optimizer.res, key=lambda x: x['target'], reverse=True)):
        params_str = "  ".join(f"{k}={v:.4f}" for k, v in res['params'].items())
        print(f"{i+1:4d}  {res['target']:8.3f}  {params_str}")


if __name__ == "__main__":
    main()
