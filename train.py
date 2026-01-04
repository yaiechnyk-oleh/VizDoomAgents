from __future__ import annotations

import argparse
import os
import time

from sb3_contrib.ppo_recurrent import RecurrentPPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from env import make_env
from cnn_gru import CnnGruPolicy, CustomCNN
from callback import (
    ActionDiagnosticsCallback,
    EntropyAnnealCallback,
    PeriodicEvalSaveCallback,
    InfoStatsCallback,
    RewardHackAlertCallback,
    EpisodeEndDiagnosticsCallback,
    GoalModeStatsCallback,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cfg", type=str, required=True)

    p.add_argument("--timesteps", type=int, default=800000)
    p.add_argument("--frame_skip", type=int, default=4)
    p.add_argument("--max_steps", type=int, default=4200)

    p.add_argument("--persona", type=str, default="rusher")
    p.add_argument("--n_envs", type=int, default=4)

    p.add_argument("--lr", type=float, default=1e-4)

    p.add_argument("--ent_coef", type=float, default=0.04)
    p.add_argument("--ent_final", type=float, default=0.02)

    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--eval_freq", type=int, default=100000)
    p.add_argument("--eval_episodes", type=int, default=30)

    p.add_argument("--own_kill_user_var", type=int, default=0)

    p.add_argument("--disable_weapon_actions", action="store_true")

    # IMPORTANT: game_reward OFF by default.
    p.add_argument("--use_game_reward", action="store_true", help="Use Doom make_action reward (NOT recommended; noisy)")
    p.add_argument("--disable_game_reward", action="store_true", help="Deprecated")

    args = p.parse_args()

    os.makedirs("models", exist_ok=True)
    run_id = int(time.time())

    best_path = os.path.join("models", f"best_{args.persona}.zip")
    last_path = os.path.join("models", f"{args.persona}_last_model.zip")

    enable_weapon_actions = (not args.disable_weapon_actions)
    use_game_reward = bool(args.use_game_reward) and (not bool(args.disable_game_reward))

    print("=" * 70)
    print(f"Training (RecurrentPPO + CustomCNN + GRU): {args.persona}")
    print(f"cfg={args.cfg} max_steps={args.max_steps} frame_skip={args.frame_skip}")
    print(f"n_envs={args.n_envs} timesteps={args.timesteps} lr={args.lr} ent={args.ent_coef}->{args.ent_final}")
    print(f"own_kill_user_var={args.own_kill_user_var}")
    print(f"enable_weapon_actions={enable_weapon_actions}")
    print(f"use_game_reward={use_game_reward}  (default OFF)")
    print("Episode rule: death = terminated (NO respawn).")
    print("=" * 70)

    # --- VecEnv ---
    env_fns = []
    for i in range(args.n_envs):
        env_fns.append(
            make_env(
                cfg=args.cfg,
                frame_skip=args.frame_skip,
                max_steps=args.max_steps,
                seed=args.seed + i * 1000,
                render=False,
                persona=args.persona,
                own_kill_user_var=args.own_kill_user_var,
                enable_weapon_actions=enable_weapon_actions,
                use_game_reward=use_game_reward,
            )
        )
    train_env = DummyVecEnv(env_fns)
    train_env = VecMonitor(train_env)

    # Separate eval env (single env)
    eval_env = DummyVecEnv([
        make_env(
            cfg=args.cfg,
            frame_skip=args.frame_skip,
            max_steps=args.max_steps,
            seed=args.seed + 99999,
            render=False,
            persona=args.persona,
            own_kill_user_var=args.own_kill_user_var,
            enable_weapon_actions=enable_weapon_actions,
            use_game_reward=use_game_reward,
        )
    ])
    eval_env = VecMonitor(eval_env)

    # --- Policy kwargs ---
    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=256),
        lstm_hidden_size=256,
        n_lstm_layers=1,
        shared_lstm=True,
        enable_critic_lstm=False,
    )

    model = RecurrentPPO(
        policy=CnnGruPolicy,
        env=train_env,
        learning_rate=args.lr,
        n_steps=128,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.1,
        ent_coef=args.ent_coef,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        device="auto",
        verbose=1,
        seed=args.seed,
    )

    # Log rollout means of reward components + attribution signals
    info_keys = [
        "r_total",
        "r_game",
        "game_reward",

        # movement / anti-camp
        "r_move",
        "r_idle",
        "r_inactive",
        "vis_static_steps",
        "stuck_steps",

        # enemy distance shaping
        "enemy_dist",
        "enemy_dist_delta",
        "r_enemy_dist",
        "r_enemy_dist_retreat",
        "r_enemy_close",
        "r_enemy_panic",

        # goal distance shaping
        "goal_dist",
        "goal_dist_delta",
        "r_goal_dist_pickup",
        "r_goal_dist_enemy_goal",

        # NEW: search shaping
        "r_search_move",
        "r_search_turn",
        "r_search_idle",

        # combat
        "r_damage",
        "r_hit",
        "r_damage_taken",
        "r_hits_taken",

        # kills attribution
        "r_monster_kill",
        "monster_kill_raw_step",
        "monster_kill_attrib_step",
        "kill_credit",
        "steps_since_attack",

        # weapon select hygiene
        "r_weapon_switch",
        "r_weapon_switch_noop",
        "r_weapon_switch_unproductive",
        "r_weapon_switch_spam",
        "weapon_select_pressed",
        "weapon_changed",

        # shooting hygiene
        "r_shoot_no_damage",
        "r_shoot_no_hit",
        "r_shoot_no_ammo",
        "r_shoot_waste_ammo",

        # termination reasons / penalties
        "terminated_by_death",
        "terminated_by_game",
        "terminated_by_stuck",
        "r_stuck_end",
    ]

    callbacks = CallbackList([
        ActionDiagnosticsCallback(print_every_rollout=True),
        InfoStatsCallback(keys=info_keys, prefix="rollout", print_every_rollout=False),

        GoalModeStatsCallback(print_every_rollout=False),

        EpisodeEndDiagnosticsCallback(verbose=1),
        RewardHackAlertCallback(verbose=1),

        EntropyAnnealCallback(ent_start=args.ent_coef, ent_end=args.ent_final, total_timesteps=args.timesteps),

        PeriodicEvalSaveCallback(
            eval_env=eval_env,
            eval_freq=args.eval_freq,
            n_eval_episodes=args.eval_episodes,
            best_path=best_path,
            deterministic=False,
            verbose=1,
        ),
        CheckpointCallback(save_freq=20000, save_path="models", name_prefix=f"ckpt_{args.persona}_{run_id}"),
    ])

    model.learn(total_timesteps=args.timesteps, callback=callbacks)
    model.save(last_path)

    print(f"[saved] last model -> {last_path}")
    print(f"[saved] best model -> {best_path}")


if __name__ == "__main__":
    main()
