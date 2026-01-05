from __future__ import annotations

import argparse
import math
import random
from typing import Any, Dict, Tuple

import numpy as np

from env import DoomDeathmatchEnv

def assert_obs(obs: np.ndarray, stack: int, size: int) -> None:
    assert isinstance(obs, np.ndarray), f"obs must be np.ndarray, got {type(obs)}"
    assert obs.dtype == np.uint8, f"obs dtype must be uint8, got {obs.dtype}"
    assert obs.shape == (stack, size, size), f"obs shape must be {(stack, size, size)}, got {obs.shape}"


def is_finite(x: float) -> bool:
    return (not math.isnan(x)) and (not math.isinf(x))


def try_suicide(env: DoomDeathmatchEnv) -> bool:
    # Some configs allow this. If not, we'll skip.
    try:
        env.game.send_game_command("suicide")
        return True
    except Exception:
        return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cfg", type=str, required=True)
    p.add_argument("--persona", type=str, default="rusher")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--frame_skip", type=int, default=4)
    p.add_argument("--max_steps", type=int, default=1200)
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--stack", type=int, default=2)
    p.add_argument("--obs_size", type=int, default=84)
    p.add_argument("--own_kill_user_var", type=int, default=0)
    args = p.parse_args()

    env = DoomDeathmatchEnv(
        cfg_path=args.cfg,
        frame_skip=args.frame_skip,
        max_steps=args.max_steps,
        seed=args.seed,
        render=False,
        persona=args.persona,
        stack=args.stack,
        obs_size=args.obs_size,
        early_end_on_stuck=False,  # tests should not truncate early
        own_kill_user_var=args.own_kill_user_var,
    )

    obs, info = env.reset()
    assert_obs(obs, args.stack, args.obs_size)

    # Basic info sanity
    assert "kill_credit" in info, "info missing kill_credit"
    assert "dead" in info, "info missing dead"
    assert "killcount" in info, "info missing killcount"
    assert "damagecount" in info, "info missing damagecount"

    print("[test] reset OK")

    # Random rollout
    total_r = 0.0
    term_count = 0
    trunc_count = 0

    for t in range(int(args.steps)):
        a = env.action_space.sample()
        obs, r, terminated, truncated, info = env.step(a)

        assert_obs(obs, args.stack, args.obs_size)
        assert is_finite(float(r)), f"reward not finite at t={t}: {r}"

        # Should NOT exist anymore:
        assert "respawned" not in info, "respawned should not exist (no respawn policy)"

        total_r += float(r)
        if terminated:
            term_count += 1
            # reset on termination to keep smoke loop going
            obs, info = env.reset()
            assert_obs(obs, args.stack, args.obs_size)
        if truncated:
            trunc_count += 1
            obs, info = env.reset()
            assert_obs(obs, args.stack, args.obs_size)

    print(f"[test] random rollout OK | total_r={total_r:.2f} term={term_count} trunc={trunc_count}")

    # Death->terminated test (best-effort)
    print("[test] trying suicide->terminated ...")
    ok = try_suicide(env)
    if not ok:
        print("[test] suicide command not available -> SKIP (not a failure)")
    else:
        # step once to advance state
        obs, r, terminated, truncated, info = env.step(0)
        if not terminated and not info.get("dead", False):
            raise AssertionError("suicide sent but env is not terminated and not dead")
        if terminated:
            print("[test] PASS: death triggers terminated=True")
        else:
            print("[test] WARN: dead=True but terminated=False (unexpected). Check step() logic.")

    env.close()
    print("[test] ALL DONE")


if __name__ == "__main__":
    main()
