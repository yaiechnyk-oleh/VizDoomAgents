from __future__ import annotations

import argparse

from stable_baselines3.common.vec_env import DummyVecEnv

from env import make_env


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cfg", type=str, required=True)
    p.add_argument("--frame_skip", type=int, default=4)
    p.add_argument("--max_steps", type=int, default=4200)
    p.add_argument("--hud_overlay", action="store_true")
    p.add_argument("--seed", type=int, default=123)
    args = p.parse_args()

    venv = DummyVecEnv([
        make_env(
            args.cfg,
            args.frame_skip,
            args.max_steps,
            args.hud_overlay,
            seed=args.seed,
            render=False,
        )
    ])

    env0 = venv.envs[0]

    # очікуємо, що в твоєму Doom env є:
    # env0.actions: List[np.ndarray]
    # env0.available_buttons: List[Button]
    n_actions = len(env0.actions)
    print("n_actions:", n_actions)

    # виводимо всі екшени, які не NOOP
    for idx in range(n_actions):
        act = env0.actions[idx]
        pressed = []
        for i, val in enumerate(act):
            if abs(float(val)) > 1e-9:
                pressed.append((env0.available_buttons[i].name, float(val)))

        if not pressed:
            pressed = [("NOOP", 0.0)]

        print(f"{idx:02d}: {pressed}")


if __name__ == "__main__":
    main()
