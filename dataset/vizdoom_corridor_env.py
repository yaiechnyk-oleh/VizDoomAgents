# old unused config
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# gym / gymnasium compatibility
try:
    import gymnasium as gym
except Exception:
    import gym

import vizdoom as vzd


def _try_gv(game: vzd.DoomGame, gv: vzd.GameVariable) -> Optional[float]:
    """Safe get_game_variable; returns None if not available."""
    try:
        return float(game.get_game_variable(gv))
    except Exception:
        return None


def _angle_diff(a: float, b: float) -> float:
    """Shortest signed angular difference in degrees."""
    d = (a - b) % 360.0
    if d > 180.0:
        d -= 360.0
    return d


@dataclass
class RewardConfig:
    # Core events (main learning signal)
    kill_reward: float = 8.0              # + per frag
    damage_reward: float = 0.02           # + per damage dealt (if available)
    pickup_reward: float = 0.02           # + per resource delta (ammo/armor/health) (capped)
    survival_reward: float = 0.002        # + per step alive (tiny)

    # Penalties
    time_penalty: float = 0.001           # - per step (tiny)
    damage_taken_penalty: float = 0.02    # - per HP lost
    death_penalty_base: float = 4.0       # - on death
    death_penalty_early: float = 4.0      # extra - if died early in episode

    # Shaping (must be small + capped)
    move_bonus_cap: float = 0.01          # max + per step from movement shaping
    move_bonus_scale: float = 0.003       # scales with distance moved
    stuck_penalty: float = 0.01           # - when trying to move but not moving
    spin_penalty: float = 0.005           # - when spinning in place

    # Anti-loop / anti-hack
    repeat_grace: int = 64                # steps allowed to repeat action without events
    repeat_penalty: float = 0.002         # - grows with repeat length
    loop_kill_after: int = 256            # force done after this if still looping
    loop_terminal_penalty: float = 2.0    # extra - on forced termination


@dataclass
class PersonaConfig:
    name: str
    reward: RewardConfig


PERSONAS: Dict[str, PersonaConfig] = {
    # Rusher: care about frags; still has anti-hack protections
    "rusher": PersonaConfig(
        name="rusher",
        reward=RewardConfig(
            kill_reward=10.0,
            damage_reward=0.03,
            pickup_reward=0.01,
            survival_reward=0.0015,
            time_penalty=0.001,
            damage_taken_penalty=0.02,
            death_penalty_base=4.0,
            death_penalty_early=4.0,
        ),
    ),
    # Default if you want
    "default": PersonaConfig(name="default", reward=RewardConfig()),
}


class SafeActionSet:
    """
    Build a safe discrete action set for VizDoom:
    - Prefer non-DELTA buttons (MOVE_LEFT, TURN_LEFT, etc.)
    - If only *_DELTA exists, we set it to +/- 1.0 (never 3.0+)
    - Avoid huge combinatorial explosion: keep it small and learnable
    """

    def __init__(self, game: vzd.DoomGame, allow_speed: bool = True):
        self.game = game
        self.allow_speed = allow_speed
        self.buttons = list(game.get_available_buttons())
        self.button_names = [b.name for b in self.buttons]
        self.actions: List[np.ndarray] = []
        self.action_names: List[str] = []

        self._build()

    def _btn_idx(self, name: str) -> Optional[int]:
        try:
            return self.button_names.index(name)
        except ValueError:
            return None

    def _make_action(self, pressed: Dict[str, float], name: str) -> None:
        act = np.zeros((len(self.buttons),), dtype=np.float32)
        for btn_name, val in pressed.items():
            idx = self._btn_idx(btn_name)
            if idx is not None:
                act[idx] = float(val)
        self.actions.append(act)
        self.action_names.append(name)

    def _build(self) -> None:
        # Helper: prefer non-delta, fallback to delta
        def move_lr(val: float) -> Dict[str, float]:
            # Prefer MOVE_LEFT / MOVE_RIGHT if exist; else MOVE_LEFT_RIGHT_DELTA
            if val < 0 and "MOVE_LEFT" in self.button_names:
                return {"MOVE_LEFT": 1.0}
            if val > 0 and "MOVE_RIGHT" in self.button_names:
                return {"MOVE_RIGHT": 1.0}
            if "MOVE_LEFT_RIGHT_DELTA" in self.button_names:
                return {"MOVE_LEFT_RIGHT_DELTA": float(np.clip(val, -1.0, 1.0))}
            return {}

        def move_fb(val: float) -> Dict[str, float]:
            if val > 0 and "MOVE_FORWARD" in self.button_names:
                return {"MOVE_FORWARD": 1.0}
            if val < 0 and "MOVE_BACKWARD" in self.button_names:
                return {"MOVE_BACKWARD": 1.0}
            if "MOVE_FORWARD_BACKWARD_DELTA" in self.button_names:
                return {"MOVE_FORWARD_BACKWARD_DELTA": float(np.clip(val, -1.0, 1.0))}
            return {}

        def turn(val: float) -> Dict[str, float]:
            if val < 0 and "TURN_LEFT" in self.button_names:
                return {"TURN_LEFT": 1.0}
            if val > 0 and "TURN_RIGHT" in self.button_names:
                return {"TURN_RIGHT": 1.0}
            if "TURN_LEFT_RIGHT_DELTA" in self.button_names:
                return {"TURN_LEFT_RIGHT_DELTA": float(np.clip(val, -1.0, 1.0))}
            return {}

        def speed_if() -> Dict[str, float]:
            return {"SPEED": 1.0} if (self.allow_speed and "SPEED" in self.button_names) else {}

        attack_btn = "ATTACK" if "ATTACK" in self.button_names else None

        # Always include NOOP
        self._make_action({}, "NOOP")

        # Basic movement / turning (with optional SPEED)
        for base_name, pressed in [
            ("FWD", {**speed_if(), **move_fb(+1.0)}),
            ("BACK", {**speed_if(), **move_fb(-1.0)}),
            ("STRAFE_L", {**speed_if(), **move_lr(-1.0)}),
            ("STRAFE_R", {**speed_if(), **move_lr(+1.0)}),
            ("TURN_L", turn(-1.0)),
            ("TURN_R", turn(+1.0)),
        ]:
            if len(pressed) > 0:
                self._make_action(pressed, base_name)

        # A couple of combos that реально потрібні для бою
        combos: List[Tuple[str, Dict[str, float]]] = [
            ("FWD+TURN_L", {**speed_if(), **move_fb(+1.0), **turn(-1.0)}),
            ("FWD+TURN_R", {**speed_if(), **move_fb(+1.0), **turn(+1.0)}),
            ("FWD+STRAFE_L", {**speed_if(), **move_fb(+1.0), **move_lr(-1.0)}),
            ("FWD+STRAFE_R", {**speed_if(), **move_fb(+1.0), **move_lr(+1.0)}),
        ]
        for name, pressed in combos:
            if len(pressed) > 0:
                self._make_action(pressed, name)

        # Attack actions (very important)
        if attack_btn is not None:
            self._make_action({attack_btn: 1.0}, "ATTACK")
            for name, pressed in combos[:2] + [("FWD", {**speed_if(), **move_fb(+1.0)})]:
                if len(pressed) > 0:
                    self._make_action({**pressed, attack_btn: 1.0}, f"{name}+ATTACK")

        # Remove duplicates
        uniq = {}
        new_actions, new_names = [], []
        for act, name in zip(self.actions, self.action_names):
            key = tuple(np.round(act, 3).tolist())
            if key not in uniq:
                uniq[key] = True
                new_actions.append(act)
                new_names.append(name)
        self.actions, self.action_names = new_actions, new_names


class VizDoomSafeDeathmatchEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 35}

    def __init__(
        self,
        cfg_path: str,
        persona: str = "rusher",
        frame_skip: int = 4,
        max_steps: int = 4200,
        hud_overlay: bool = False,
        seed: int = 0,
        render: bool = False,
    ):
        super().__init__()
        self.cfg_path = cfg_path
        self.persona = PERSONAS.get(persona, PERSONAS["default"])
        self.frame_skip = int(frame_skip)
        self.max_steps = int(max_steps)
        self.hud_overlay = bool(hud_overlay)
        self.seed_val = int(seed)
        self.render_enabled = bool(render)

        self.game = vzd.DoomGame()
        self.game.load_config(cfg_path)
        self.game.set_seed(self.seed_val)

        # For watching
        self.game.set_window_visible(self.render_enabled)
        self.game.set_mode(vzd.Mode.PLAYER)

        self.game.init()

        # Build safe action set
        self.action_set = SafeActionSet(self.game, allow_speed=True)
        self.actions = self.action_set.actions
        self.action_names = self.action_set.action_names

        self.action_space = gym.spaces.Discrete(len(self.actions))

        # Observation: we use grayscale + frame-diff => 2x84x84 like your model
        self.obs_h = 84
        self.obs_w = 84
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(2, self.obs_h, self.obs_w), dtype=np.uint8
        )

        # Episode state
        self._step = 0
        self._prev_gray: Optional[np.ndarray] = None

        # For reward shaping
        self._prev_kills = 0.0
        self._prev_damage = 0.0
        self._prev_health = 0.0
        self._prev_armor = 0.0
        self._prev_ammo = 0.0
        self._prev_pos = None  # (x,y)
        self._prev_ang = None

        # Anti-loop
        self._last_action = None
        self._repeat_len = 0
        self._event_since_repeat = False

    def close(self):
        try:
            self.game.close()
        except Exception:
            pass

    def render(self):
        # VizDoom renders itself when window_visible=True in PLAYER mode
        return None

    def _get_game_vars(self) -> Dict[str, float]:
        # Prefer FRAGCOUNT in deathmatch; fallback to KILLCOUNT
        frag = _try_gv(self.game, vzd.GameVariable.FRAGCOUNT)
        kills = frag if frag is not None else (_try_gv(self.game, vzd.GameVariable.KILLCOUNT) or 0.0)

        dmg = _try_gv(self.game, vzd.GameVariable.DAMAGECOUNT)  # may be None in some configs
        health = _try_gv(self.game, vzd.GameVariable.HEALTH) or 0.0
        armor = _try_gv(self.game, vzd.GameVariable.ARMOR) or 0.0

        # Ammo is tricky (depends on weapon). Try a few common vars.
        ammo = (
            _try_gv(self.game, vzd.GameVariable.AMMO2)
            or _try_gv(self.game, vzd.GameVariable.AMMO)
            or _try_gv(self.game, vzd.GameVariable.SELECTED_WEAPON_AMMO)
            or 0.0
        )

        x = _try_gv(self.game, vzd.GameVariable.POSITION_X)
        y = _try_gv(self.game, vzd.GameVariable.POSITION_Y)
        ang = _try_gv(self.game, vzd.GameVariable.ANGLE)

        return {
            "kills": float(kills),
            "damage": float(dmg) if dmg is not None else 0.0,
            "health": float(health),
            "armor": float(armor),
            "ammo": float(ammo),
            "x": float(x) if x is not None else float("nan"),
            "y": float(y) if y is not None else float("nan"),
            "angle": float(ang) if ang is not None else float("nan"),
        }

    def _get_obs(self) -> np.ndarray:
        state = self.game.get_state()
        if state is None or state.screen_buffer is None:
            # fallback
            return np.zeros((2, self.obs_h, self.obs_w), dtype=np.uint8)

        # screen_buffer: HxWxC (RGB)
        img = state.screen_buffer
        # downscale to 84x84 and grayscale
        # (VizDoom gives uint8)
        # Use simple area resize via numpy slicing (fast enough); can replace with cv2 if you want
        # We'll do nearest-neighbor for simplicity
        h, w = img.shape[:2]
        ys = (np.linspace(0, h - 1, self.obs_h)).astype(np.int32)
        xs = (np.linspace(0, w - 1, self.obs_w)).astype(np.int32)
        small = img[ys][:, xs]  # 84x84xC
        gray = (0.299 * small[..., 0] + 0.587 * small[..., 1] + 0.114 * small[..., 2]).astype(np.uint8)

        if self._prev_gray is None:
            diff = np.zeros_like(gray, dtype=np.uint8)
        else:
            diff = np.clip(np.abs(gray.astype(np.int16) - self._prev_gray.astype(np.int16)), 0, 255).astype(np.uint8)

        self._prev_gray = gray
        obs = np.stack([gray, diff], axis=0)  # 2x84x84
        return obs

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self.seed_val = int(seed)
            self.game.set_seed(self.seed_val)

        self.game.new_episode()
        self._step = 0
        self._prev_gray = None

        v = self._get_game_vars()
        self._prev_kills = v["kills"]
        self._prev_damage = v["damage"]
        self._prev_health = v["health"]
        self._prev_armor = v["armor"]
        self._prev_ammo = v["ammo"]
        self._prev_pos = (v["x"], v["y"]) if np.isfinite(v["x"]) and np.isfinite(v["y"]) else None
        self._prev_ang = v["angle"] if np.isfinite(v["angle"]) else None

        self._last_action = None
        self._repeat_len = 0
        self._event_since_repeat = False

        obs = self._get_obs()
        info = {"kills": 0.0}
        return obs, info

    def step(self, action: int):
        action = int(action)
        self._step += 1

        # Anti-loop tracking (event flag resets when action changes)
        if self._last_action == action:
            self._repeat_len += 1
        else:
            self._last_action = action
            self._repeat_len = 1
            self._event_since_repeat = False

        # Do action
        act_vec = self.actions[action]
        reward_game = self.game.make_action(act_vec, self.frame_skip)

        done = self.game.is_episode_finished() or (self._step >= self.max_steps)
        v = self._get_game_vars()

        # ---------- Reward shaping (SAFE) ----------
        rc = self.persona.reward
        r = 0.0
        events_happened = False

        # Survival / time
        r += rc.survival_reward
        r -= rc.time_penalty

        # Kills
        dk = max(0.0, v["kills"] - self._prev_kills)
        if dk > 0:
            r += rc.kill_reward * dk
            events_happened = True

        # Damage dealt (if available)
        ddmg = max(0.0, v["damage"] - self._prev_damage)
        if ddmg > 0:
            r += rc.damage_reward * ddmg
            events_happened = True

        # Pickups (cap to avoid farming health “up/down” loops)
        dh = max(0.0, v["health"] - self._prev_health)
        da = max(0.0, v["armor"] - self._prev_armor)
        dammo = max(0.0, v["ammo"] - self._prev_ammo)
        pickup = min(50.0, dh + da + dammo)  # cap
        if pickup > 0:
            r += rc.pickup_reward * pickup
            events_happened = True

        # Damage taken penalty
        lost_hp = max(0.0, self._prev_health - v["health"])
        if lost_hp > 0:
            r -= rc.damage_taken_penalty * lost_hp

        # Movement shaping (capped; NEVER can dominate kills)
        if self._prev_pos is not None and np.isfinite(v["x"]) and np.isfinite(v["y"]):
            dx = v["x"] - self._prev_pos[0]
            dy = v["y"] - self._prev_pos[1]
            dist = math.sqrt(dx * dx + dy * dy)

            move_bonus = min(rc.move_bonus_cap, rc.move_bonus_scale * dist)
            r += move_bonus

            # Stuck penalty: if action likely tries to move (has MOVE_* or delta), but dist tiny
            act = act_vec
            tries_move = False
            for nm in ["MOVE_FORWARD", "MOVE_BACKWARD", "MOVE_LEFT", "MOVE_RIGHT",
                       "MOVE_LEFT_RIGHT_DELTA", "MOVE_FORWARD_BACKWARD_DELTA"]:
                if nm in self.action_set.button_names:
                    idx = self.action_set.button_names.index(nm)
                    if abs(float(act[idx])) > 1e-6:
                        tries_move = True
                        break

            if tries_move and dist < 1e-3:
                r -= rc.stuck_penalty

            # Spin penalty: turning without displacement for long time
            if self._prev_ang is not None and np.isfinite(v["angle"]):
                dang = abs(_angle_diff(v["angle"], self._prev_ang))
                if dang > 5.0 and dist < 1e-3:
                    r -= rc.spin_penalty
        else:
            dist = 0.0

        # Anti-loop penalty (only if no real events)
        if (not events_happened) and (not self._event_since_repeat):
            if self._repeat_len > rc.repeat_grace:
                over = self._repeat_len - rc.repeat_grace
                r -= rc.repeat_penalty * min(1.0, over / 64.0)

                # if it loops too long => force episode end (cuts reward-hacking)
                if self._repeat_len >= rc.loop_kill_after:
                    done = True
                    r -= rc.loop_terminal_penalty

        if events_happened:
            self._event_since_repeat = True

        # Death penalty (scaled, not -60!)
        if self.game.is_player_dead():
            early = 1.0 - (self._step / max(1, self.max_steps))
            r -= (rc.death_penalty_base + rc.death_penalty_early * early)

        # Optional: include raw game reward very lightly (usually 0 in deathmatch cfg)
        r += 0.0 * float(reward_game)

        # Update prev vars
        self._prev_kills = v["kills"]
        self._prev_damage = v["damage"]
        self._prev_health = v["health"]
        self._prev_armor = v["armor"]
        self._prev_ammo = v["ammo"]
        self._prev_pos = (v["x"], v["y"]) if np.isfinite(v["x"]) and np.isfinite(v["y"]) else self._prev_pos
        self._prev_ang = v["angle"] if np.isfinite(v["angle"]) else self._prev_ang

        obs = self._get_obs()

        info = {
            "kills": float(v["kills"]),
            "health": float(v["health"]),
            "armor": float(v["armor"]),
            "ammo": float(v["ammo"]),
            "dead": bool(self.game.is_player_dead()),
            "action_id": action,
            "action_name": self.action_names[action],
            "repeat_len": int(self._repeat_len),
        }

        # Gym/Gymnasium compatibility: we return 4-tuple for gym, 5-tuple for gymnasium
        if hasattr(gym, "Env") and "gymnasium" in str(gym.__name__):
            terminated = bool(self.game.is_episode_finished() or self.game.is_player_dead())
            truncated = bool(self._step >= self.max_steps)
            return obs, float(r), terminated, truncated, info
        else:
            return obs, float(r), bool(done), info
