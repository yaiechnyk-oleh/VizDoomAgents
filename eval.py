from __future__ import annotations

import argparse
import csv
import re
import time
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Optional, Dict, List

import numpy as np
from sb3_contrib.ppo_recurrent import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

# Ensure custom policy + extractor classes are importable for model loading
from cnn_gru import CnnGruPolicy, CnnStateExtractor  # noqa: F401

from env import make_env


@dataclass
class EpisodeResult:
    episode: int
    reward: float

    monster_kills_attrib: float
    monster_kills_raw: float
    suspicious_monster_kills: float

    killcount: float
    fragcount: float

    damage_dealt: float
    damage_taken: float

    hits: float
    hits_taken: float

    deaths: float

    end_hp: float
    end_armor: float
    end_ammo: float
    end_weapon: float
    steps: int


def _unwrap_env(vec_env) -> Any:
    try:
        e = vec_env
        if hasattr(e, "venv"):
            e = e.venv
        if hasattr(e, "envs") and len(e.envs) > 0:
            e0 = e.envs[0]
            while hasattr(e0, "env"):
                e0 = e0.env
            return e0
    except Exception:
        pass
    return None


def _safe_getattr(obj: Any, names: List[str]) -> Any:
    for n in names:
        if hasattr(obj, n):
            return getattr(obj, n)
    return None


def _describe_action(base_env: Any, action_id: int) -> str:
    if base_env is None:
        return f"id={action_id} (base_env=None)"

    actions = _safe_getattr(base_env, ["actions", "_actions", "action_map", "_action_map"])
    buttons = _safe_getattr(
        base_env,
        ["buttons", "_buttons", "available_buttons", "_available_buttons", "button_names"],
    )

    try:
        if actions is None:
            return f"id={action_id} (no actions table exposed on env)"

        a = actions[action_id]
        a_vec = np.asarray(a).astype(int).tolist()

        on_idx = [i for i, v in enumerate(a_vec) if int(v) != 0]
        if buttons is not None:
            b = list(buttons)
            names = []
            for i in on_idx:
                if i < len(b):
                    names.append(str(b[i]))
                else:
                    names.append(f"btn[{i}]")
            return f"id={action_id} vec={a_vec} ON={names}"
        return f"id={action_id} vec={a_vec} ON_idx={on_idx}"
    except Exception as e:
        return f"id={action_id} (decode failed: {e})"


def _extract_step_vars(info: Dict[str, Any]) -> Dict[str, float]:
    def g(*keys: str, default: float = 0.0) -> float:
        for k in keys:
            if k in info:
                try:
                    return float(info.get(k, default))
                except Exception:
                    pass
        return float(default)

    return {
        "health": g("health", "HEALTH"),
        "armor": g("armor", "ARMOR"),
        "ammo_total": g("ammo_total", "AMMO_TOTAL"),
        "selected_weapon": g("selected_weapon", "SELECTED_WEAPON"),
        "selected_weapon_ammo": g("selected_weapon_ammo", "SELECTED_WEAPON_AMMO"),

        "killcount": g("killcount", "KILLCOUNT"),
        "own_killcount": g("own_killcount", "OWN_KILLCOUNT"),
        "fragcount": g("fragcount", "FRAGCOUNT"),

        "damagecount": g("damagecount", "DAMAGECOUNT"),
        "damage_taken": g("damage_taken", "DAMAGE_TAKEN"),
        "hitcount": g("hitcount", "HITCOUNT"),
        "hits_taken": g("hits_taken", "HITS_TAKEN"),
        "deathcount": g("deathcount", "DEATHCOUNT"),

        "mk_raw_step": g("monster_kill_raw_step"),
        "mk_attrib_step": g("monster_kill_attrib_step"),

        "kill_credit": g("kill_credit"),
        "steps_since_attack": g("steps_since_attack"),

        "game_reward": g("game_reward"),

        "weapon_select_pressed": g("weapon_select_pressed"),
        "weapon_changed": g("weapon_changed"),

        "enemy_dist": g("enemy_dist", default=-1.0),
        "enemy_dist_delta": g("enemy_dist_delta"),
        "enemy_visible": g("enemy_visible"),
        "enemy_angle_err": g("enemy_angle_err"),
        "enemy_angle_err_abs": g("enemy_angle_err_abs", default=180.0),

        "r_enemy_dist": g("r_enemy_dist"),
        "r_enemy_dist_retreat": g("r_enemy_dist_retreat"),
        "r_enemy_close": g("r_enemy_close"),
        "r_enemy_panic": g("r_enemy_panic"),

        "goal_dist": g("goal_dist", default=-1.0),
        "goal_dist_delta": g("goal_dist_delta"),
        "r_goal_dist_pickup": g("r_goal_dist_pickup"),
        "r_goal_dist_enemy_goal": g("r_goal_dist_enemy_goal"),

        "r_search_move": g("r_search_move"),
        "r_search_turn": g("r_search_turn"),
        "r_search_idle": g("r_search_idle"),

        "r_aim": g("r_aim"),
        "r_aim_center": g("r_aim_center"),
        "r_attack_on_target": g("r_attack_on_target"),
        "r_no_attack_on_target": g("r_no_attack_on_target"),

        # v2 reward components
        "r_engage_in_range": g("r_engage_in_range"),
        "r_backward_combat": g("r_backward_combat"),
        "r_weapon_situational": g("r_weapon_situational"),

        # v3 reward components
        "r_blind_fire": g("r_blind_fire"),
    }


def _pretty_deltas(prev: Dict[str, float], cur: Dict[str, float], keys: List[str]) -> str:
    parts = []
    for k in keys:
        dp = float(cur.get(k, 0.0) - prev.get(k, 0.0))
        if dp == 0.0:
            continue
        if "dist" in k:
            parts.append(f"Δ{k}={dp:+.2f}")
        elif "err" in k:
            parts.append(f"Δ{k}={dp:+.2f}")
        elif k in ("kill_credit",):
            parts.append(f"Δ{k}={dp:+.3f}")
        else:
            parts.append(f"Δ{k}={dp:+.0f}")
    return " ".join(parts) if parts else "(no deltas)"


def _parse_int_list(spec: str) -> List[int]:
    out: List[int] = []
    for s in re.split(r"[,\s]+", spec.strip()):
        if not s:
            continue
        out.append(int(s))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--persona", type=str, default="rusher")
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--cfg", type=str, required=True)

    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--frame_skip", type=int, default=4)
    p.add_argument("--max_steps", type=int, default=4200)

    p.add_argument("--watch", action="store_true", help="Render window while evaluating")
    p.add_argument("--watch_fps", type=float, default=35.0)

    p.add_argument("--stochastic", action="store_true", help="deterministic=False")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--out", type=str, default="", help="Optional: write metrics to CSV")

    p.add_argument("--own_kill_user_var", type=int, default=0)
    p.add_argument("--disable_weapon_actions", action="store_true")

    p.add_argument("--use_game_reward", action="store_true", help="Use Doom make_action reward (NOT recommended; noisy)")
    p.add_argument("--disable_game_reward", action="store_true", help="Deprecated")

    p.add_argument("--diag", action="store_true", help="Enable step-by-step diagnostics")
    p.add_argument("--diag_every", type=int, default=10)
    p.add_argument("--diag_first", type=int, default=50)
    p.add_argument("--diag_action_ids", type=str, default="11")
    p.add_argument("--diag_print_info_keys", action="store_true")

    p.add_argument("--diag_decode_top_actions", action="store_true")
    p.add_argument("--diag_decode_top_k", type=int, default=5)

    args = p.parse_args()

    use_game_reward = bool(args.use_game_reward) and (not bool(args.disable_game_reward))

    env = DummyVecEnv(
        [
            make_env(
                cfg=args.cfg,
                frame_skip=args.frame_skip,
                max_steps=args.max_steps,
                seed=args.seed,
                render=args.watch,
                persona=args.persona,
                own_kill_user_var=args.own_kill_user_var,
                enable_weapon_actions=(not args.disable_weapon_actions),
                use_game_reward=use_game_reward,
            )
        ]
    )
    env = VecMonitor(env)

    base_env = _unwrap_env(env)

    try:
        vars0 = env.get_attr("_vars")[0]
        print("\n=== RUNTIME GAME VARIABLES (env._vars) ===")
        for v in sorted(list(vars0), key=lambda x: x.name):
            print(" -", v)
        print("Total:", len(vars0))
    except Exception as e:
        print("[debug] cannot read env vars:", e)

    model = RecurrentPPO.load(args.model, device="auto")
    det = not args.stochastic

    if args.diag:
        try:
            ids = _parse_int_list(args.diag_action_ids)
            print("\n=== DIAG: ACTION DECODE (static) ===")
            for aid in ids:
                print(" ", _describe_action(base_env, aid))
            print("=== END ACTION DECODE ===\n")
        except Exception as e:
            print("[diag] cannot decode actions:", e)

    overall_action_counts: Counter[int] = Counter()
    results: List[EpisodeResult] = []

    out_path: Optional[Path] = Path(args.out) if args.out else None
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)

    for ep in range(1, args.episodes + 1):
        obs = env.reset()
        state = None
        episode_start = np.ones((env.num_envs,), dtype=bool)

        done = [False]
        ep_rew = 0.0
        last_info: Optional[Dict[str, Any]] = None
        ep_action_counts: Counter[int] = Counter()
        steps = 0

        prev_vars: Optional[Dict[str, float]] = None
        suspicious_local = 0.0

        action_history: List[int] = []
        stuck_recovery_steps = 0

        while not done[0]:
            force_stochastic = False
            if det:
                if len(action_history) >= 15 and len(set(action_history[-15:])) == 1:
                    stuck_recovery_steps = 3  # Force stochastic for 3 steps to break out
                    action_history.clear()
                
                if stuck_recovery_steps > 0:
                    force_stochastic = True
                    stuck_recovery_steps -= 1

            action, state = model.predict(
                obs,
                state=state,
                episode_start=episode_start,
                deterministic=det and not force_stochastic,
            )

            a0 = int(action[0])
            if det:
                action_history.append(a0)
                if len(action_history) > 20:
                    action_history.pop(0)
                    
            overall_action_counts[a0] += 1
            ep_action_counts[a0] += 1

            obs, reward, done, infos = env.step(action)
            episode_start = np.array(done, dtype=bool)

            ep_rew += float(reward[0])
            last_info = infos[0] if infos and len(infos) > 0 else None
            steps += 1

            if args.diag and last_info is not None:
                cur_vars = _extract_step_vars(last_info)

                if prev_vars is None:
                    prev_vars = cur_vars
                    if args.diag_print_info_keys:
                        print(f"[diag ep={ep} step=1] info keys:", sorted(list(last_info.keys())))

                if cur_vars["mk_raw_step"] > 0 and cur_vars["mk_attrib_step"] <= 0:
                    suspicious_local += cur_vars["mk_raw_step"]

                should_print = (
                    (steps <= args.diag_first)
                    or (args.diag_every > 0 and steps % args.diag_every == 0)
                    or done[0]
                )
                if should_print:
                    deltas = _pretty_deltas(
                        prev_vars,
                        cur_vars,
                        keys=[
                            "health",
                            "armor",
                            "ammo_total",
                            "selected_weapon",
                            "selected_weapon_ammo",
                            "weapon_select_pressed",
                            "weapon_changed",
                            "killcount",
                            "own_killcount",
                            "fragcount",
                            "damagecount",
                            "damage_taken",
                            "hitcount",
                            "hits_taken",
                            "deathcount",
                            "mk_raw_step",
                            "mk_attrib_step",
                            "kill_credit",
                            "steps_since_attack",

                            "enemy_visible",
                            "enemy_dist",
                            "enemy_dist_delta",
                            "enemy_angle_err_abs",
                            "r_enemy_dist",
                            "r_enemy_dist_retreat",
                            "r_enemy_close",
                            "r_enemy_panic",

                            "r_aim",
                            "r_aim_center",
                            "r_attack_on_target",
                            "r_no_attack_on_target",

                            "goal_dist",
                            "goal_dist_delta",
                            "r_goal_dist_pickup",
                            "r_goal_dist_enemy_goal",

                            "r_search_move",
                            "r_search_turn",
                            "r_search_idle",

                            "r_engage_in_range",
                            "r_backward_combat",
                            "r_weapon_situational",

                            "game_reward",
                        ],
                    )
                    tl_trunc = last_info.get("TimeLimit.truncated", None)
                    goal_mode = str(last_info.get("goal_mode", ""))
                    print(
                        f"[diag ep={ep} step={steps} a={a0} r={float(reward[0]):+.3f} done={done[0]} "
                        f"TimeLimit.truncated={tl_trunc} susp_local={suspicious_local:.1f} "
                        f"goal_mode={goal_mode} {deltas}"
                    )
                prev_vars = cur_vars

            if args.watch and args.watch_fps and args.watch_fps > 0:
                time.sleep(1.0 / float(args.watch_fps))

        if last_info is None:
            print(f"Episode {ep}: no info")
            continue

        r = EpisodeResult(
            episode=ep,
            reward=float(ep_rew),

            monster_kills_attrib=float(last_info.get("ep_monster_kills", 0.0)),
            monster_kills_raw=float(last_info.get("ep_monster_kills_raw", 0.0)),
            suspicious_monster_kills=float(last_info.get("ep_suspicious_monster_kills", 0.0)),

            killcount=float(last_info.get("ep_killcount", 0.0)),
            fragcount=float(last_info.get("ep_fragcount", 0.0)),

            damage_dealt=float(last_info.get("ep_damage_dealt", 0.0)),
            damage_taken=float(last_info.get("ep_damage_taken", 0.0)),

            hits=float(last_info.get("ep_hits", 0.0)),
            hits_taken=float(last_info.get("ep_hits_taken", 0.0)),

            deaths=float(last_info.get("ep_deaths", 0.0)),

            end_hp=float(max(0.0, last_info.get("health", 0.0))),
            end_armor=float(max(0.0, last_info.get("armor", 0.0))),
            end_ammo=float(max(0.0, last_info.get("ammo_total", 0.0))),
            end_weapon=float(last_info.get("selected_weapon", 0.0)),
            steps=int(steps),
        )
        results.append(r)

        top_actions_ep = ep_action_counts.most_common(8)
        print(
            f"Episode {ep}: R={r.reward:.2f} | mk_attrib={r.monster_kills_attrib:.2f} mk_raw={r.monster_kills_raw:.2f} "
            f"susp={r.suspicious_monster_kills:.2f} | killcount={r.killcount:.1f} frag={r.fragcount:.1f} "
            f"| dmg+={r.damage_dealt:.1f} dmg-={r.damage_taken:.1f} "
            f"| hit={r.hits:.1f} hit-={r.hits_taken:.1f} "
            f"| deaths={r.deaths:.1f} "
            f"| end_hp={r.end_hp:.1f} armor={r.end_armor:.1f} ammo={r.end_ammo:.1f} wpn={r.end_weapon:.0f} "
            f"| steps={r.steps} | top_actions={top_actions_ep}"
        )

        if args.diag and args.diag_decode_top_actions:
            k = max(1, int(args.diag_decode_top_k))
            topk = ep_action_counts.most_common(k)
            print(f"\n=== DIAG: TOP-{k} ACTIONS (episode {ep}) ===")
            for aid, cnt in topk:
                print(f" count={cnt:6d}  {_describe_action(base_env, int(aid))}")
            print("=== END TOP ACTIONS ===\n")

    if results:
        mean = lambda xs: float(np.mean(xs))
        print("\n=== EVAL SUMMARY ===")
        print(f"model: {args.model}")
        print(f"cfg: {args.cfg}")
        print(f"episodes: {args.episodes} | deterministic: {det} | seed: {args.seed}")
        print(f"mean_reward: {mean([x.reward for x in results]):.3f}")
        print(f"mean_mk_attrib: {mean([x.monster_kills_attrib for x in results]):.3f}")
        print(f"mean_mk_raw: {mean([x.monster_kills_raw for x in results]):.3f}")
        print(f"mean_suspicious: {mean([x.suspicious_monster_kills for x in results]):.3f}")
        print(f"mean_killcount: {mean([x.killcount for x in results]):.3f}")
        print(f"mean_fragcount: {mean([x.fragcount for x in results]):.3f}")
        print(f"mean_damage_dealt: {mean([x.damage_dealt for x in results]):.3f}")
        print(f"mean_damage_taken: {mean([x.damage_taken for x in results]):.3f}")
        print(f"mean_hits: {mean([x.hits for x in results]):.3f}")
        print(f"mean_hits_taken: {mean([x.hits_taken for x in results]):.3f}")
        print(f"mean_deaths: {mean([x.deaths for x in results]):.3f}")

    print("\nTop actions overall:", overall_action_counts.most_common(12))
    print("Total unique actions overall:", len(overall_action_counts))

    if args.diag and args.diag_decode_top_actions:
        k = max(1, int(args.diag_decode_top_k))
        topk = overall_action_counts.most_common(k)
        print(f"\n=== DIAG: TOP-{k} ACTIONS (overall) ===")
        for aid, cnt in topk:
            print(f" count={cnt:6d}  {_describe_action(base_env, int(aid))}")
        print("=== END TOP ACTIONS ===\n")

    if out_path and results:
        with out_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
            w.writeheader()
            for r in results:
                w.writerow(asdict(r))
        print(f"\n[saved] episode metrics -> {out_path.as_posix()}")


if __name__ == "__main__":
    main()
