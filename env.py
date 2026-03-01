from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Set

import numpy as np

# Gymnasium-first (SB3>=2), fallback to gym
try:
    import gymnasium as gym
    from gymnasium import spaces
    from gymnasium.utils import seeding

    _USING_GYMNASIUM = True
except Exception:  # pragma: no cover
    import gym  # type: ignore
    from gym import spaces  # type: ignore
    from gym.utils import seeding  # type: ignore

    _USING_GYMNASIUM = False

import cv2
from vizdoom import DoomGame, Mode, Button, GameVariable, ScreenFormat, ScreenResolution  # type: ignore


# ---------------- Reward config ----------------

@dataclass
class RewardWeights:
    # Combat shaping (primary)
    damage: float = 0.030              # per damage dealt
    hit: float = 0.12                  # per HITCOUNT increment

    damage_taken: float = 0.055        # per damage taken (penalty)
    hits_taken: float = 0.08           # per HITS_TAKEN increment (penalty)

    death: float = 12.0                # per DEATHCOUNT increment (penalty)

    # Kill reward (secondary, must be attributed) - used only if NOT using game reward
    monster_kill: float = 7.0          # reward per attributed monster kill
    frag_penalty: float = 3.0          # penalty per FRAGCOUNT increment (avoid killing bots/players)

    # Attribution via "combat credit"
    kill_credit_damage_scale: float = 0.020
    kill_credit_hit_scale: float = 0.35
    kill_credit_decay: float = 0.035
    kill_credit_cap: float = 2.5
    kill_credit_cost_per_kill: float = 1.0

    # Hard recent window gate
    kill_requires_recent_damage_steps: int = 12
    kill_requires_recent_hit_steps: int = 12
    kill_requires_recent_attack_steps: int = 18

    # Pickups / resources (reward only on positive delta)
    health_pickup: float = 0.040
    armor_pickup: float = 0.0015
    ammo_pickup: float = 0.0030

    # Caps to avoid reward spikes from huge pickups (blue armor/mega ammo etc.)
    health_pickup_cap: float = 35.0
    armor_pickup_cap: float = 50.0
    ammo_pickup_cap: float = 30.0

    # Extra scaling
    low_health_pickup_mult: float = 1.8   # applied ONLY when hp is critical
    low_ammo_pickup_mult: float = 1.6     # applied ONLY when ammo is critical
    armor_have_scale: float = 0.05        # if already have armor, reward is tiny

    # Resource-aware multipliers (kept for other shaping)
    low_health_thr: float = 35.0
    low_ammo_thr: float = 5.0
    low_health_damage_taken_mult: float = 1.5

    # Movement / anti-camp
    move: float = 0.022
    under_fire_move_bonus: float = 0.018
    idle: float = 0.015
    bump: float = 0.030
    stuck: float = 0.20

    inactive: float = 0.020

    # weapon switch hygiene (penalties)
    weapon_switch: float = 0.030
    weapon_switch_noop: float = 0.012
    weapon_switch_unproductive: float = 0.040
    weapon_switch_spam_window: int = 10
    weapon_switch_spam_penalty: float = 0.020

    # Shooting hygiene
    shoot_no_damage: float = 0.006
    shoot_no_hit: float = 0.003
    shoot_no_ammo: float = 0.10
    shoot_waste_ammo: float = 0.020

    # Built-in scenario reward (from DoomGame.make_action)
    game_reward_scale: float = 1.0
    game_reward_clip: float = 30.0

    stuck_end_penalty: float = 8.0

    spin_in_place: float = 0.0009

    step_penalty: float = 0.0

    # -------- Safe distance reward (enemy proximity shaping) --------
    enemy_dist: float = 0.020
    enemy_dist_norm: float = 64.0
    enemy_min_safe: float = 80.0
    enemy_too_close: float = 48.0
    enemy_close_penalty: float = 0.080
    enemy_retreat: float = 0.020
    enemy_panic_penalty: float = 0.150

    enemy_retreat_low_health: float = 25.0
    enemy_retreat_low_ammo: float = 3.0

    # -------- Goal distance shaping (toward current high-level goal) --------
    goal_dist_pickup: float = 0.018
    goal_dist_enemy: float = 0.012
    goal_dist_norm: float = 64.0
    goal_dist_step_cap: float = 60.0

    # -------- strict switching policy --------
    goal_hp_crit: float = 20.0
    goal_ammo_crit: float = 5.0

    # Exit hysteresis
    goal_hp_exit_margin: float = 15.0
    goal_ammo_exit_margin: float = 6.0

    # -------- search mode --------
    search_move_bonus: float = 0.010
    search_turn_bonus: float = 0.006
    search_turn_norm_deg: float = 30.0
    search_idle_penalty: float = 0.002

    # -------- aiming / engagement (NEW) --------
    aim: float = 0.015
    aim_err_norm_deg: float = 45.0
    aim_center_thr_deg: float = 8.0
    aim_center_bonus: float = 0.050
    attack_on_target_bonus: float = 0.100
    no_attack_on_target_penalty: float = 0.004
    engage_dist_max: float = 320.0

    # target lock to avoid "skipping past enemies"
    target_hold_steps: int = 12

    clip_min: float = -30.0
    clip_max: float = 30.0


def persona_weights(persona: str) -> RewardWeights:
    persona = (persona or "rusher").lower().strip()
    if persona == "rusher":
        return RewardWeights(
            damage=0.075,
            hit=0.30,
            damage_taken=0.035,
            hits_taken=0.06,
            death=14.0,

            monster_kill=10.0,
            frag_penalty=3.5,

            kill_credit_damage_scale=0.022,
            kill_credit_hit_scale=0.38,
            kill_credit_decay=0.038,
            kill_credit_cap=2.7,
            kill_credit_cost_per_kill=1.0,

            # pickups tuned to avoid farming
            health_pickup=0.040,
            armor_pickup=0.0015,
            ammo_pickup=0.0030,
            health_pickup_cap=35.0,
            armor_pickup_cap=50.0,
            ammo_pickup_cap=30.0,
            armor_have_scale=0.05,
            low_health_pickup_mult=1.8,
            low_ammo_pickup_mult=1.6,

            low_health_thr=40.0,
            low_ammo_thr=6.0,

            move=0.008,
            under_fire_move_bonus=0.020,
            idle=0.010,
            bump=0.032,
            stuck=0.18,

            inactive=0.024,

            weapon_switch=0.035,
            weapon_switch_noop=0.015,
            weapon_switch_unproductive=0.050,
            weapon_switch_spam_window=10,
            weapon_switch_spam_penalty=0.025,

            shoot_no_damage=0.007,
            shoot_no_hit=0.0035,
            shoot_no_ammo=0.12,
            shoot_waste_ammo=0.020,

            game_reward_scale=1.0,
            game_reward_clip=30.0,

            stuck_end_penalty=8.0,

            spin_in_place=0.0010,

            kill_requires_recent_damage_steps=12,
            kill_requires_recent_hit_steps=12,
            kill_requires_recent_attack_steps=18,

            enemy_dist=0.016,
            enemy_retreat=0.014,
            enemy_dist_norm=64.0,
            enemy_min_safe=80.0,
            enemy_too_close=48.0,
            enemy_close_penalty=0.080,
            enemy_panic_penalty=0.150,
            enemy_retreat_low_health=25.0,
            enemy_retreat_low_ammo=3.0,

            goal_dist_pickup=0.020,
            goal_dist_enemy=0.012,
            goal_dist_norm=64.0,
            goal_dist_step_cap=60.0,

            goal_hp_crit=20.0,
            goal_ammo_crit=5.0,
            goal_hp_exit_margin=15.0,
            goal_ammo_exit_margin=6.0,

            search_move_bonus=0.012,
            search_turn_bonus=0.007,
            search_turn_norm_deg=30.0,
            search_idle_penalty=0.002,

            # NEW: aiming / engagement
            aim=0.10,
            aim_err_norm_deg=45.0,
            aim_center_thr_deg=15.0,
            aim_center_bonus=0.080,
            attack_on_target_bonus=0.200,
            no_attack_on_target_penalty=0.025,
            engage_dist_max=320.0,
            target_hold_steps=14,

            step_penalty=0.0,
        )
    return RewardWeights()


# ---------------- Env ----------------

class DoomDeathmatchEnv(gym.Env):
    """
    Project rules:
      - Death = end of episode (terminated=True).
      - No respawn inside an episode.
      - If early_end_on_stuck triggers: we TERMINATE + apply one-shot penalty.

    Goal switching (IMPORTANT):
      - Enemy is ALWAYS default priority.
      - Switch away from enemy ONLY when critical:
          hp <= goal_hp_crit  OR  ammo <= goal_ammo_crit
      - If enemy not visible and not critical -> goal_mode="search"
      - Armor is NOT a goal. It can be collected incidentally with a very small reward.
    """

    metadata = {"render_modes": ["human"], "render_fps": 35}

    ENEMY_NAMES = {
        "doomimp",
        "zombieman",
        "shotgunguy",
        "chaingunguy",
        "demon",
        "spectre",
        "cacodemon",
        "hellknight",
        "baronofhell",
        "arachnotron",
        "revenant",
        "mancubus",
        "painelemental",
        "lostsoul",
    }

    NON_ENEMY_NAMES = {
        "doomplayer",
        "marinechainsawvzd",
    }

    HEALTH_NAMES = {"stimpack", "medikit", "healthbonus"}
    ARMOR_NAMES = {"greenarmor", "bluearmor", "armorbonus"}
    AMMO_NAMES = {
        "clip", "clipbox",
        "shell", "shellbox",
        "rocketammo", "rocketbox",
        "cell", "cellpack",
    }

    def __init__(
        self,
        cfg_path: str,
        frame_skip: int = 4,
        max_steps: int = 4200,
        seed: int = 0,
        render: bool = False,
        persona: str = "rusher",
        obs_size: int = 84,
        stack: int = 2,
        stuck_window: int = 24,
        early_end_on_stuck: bool = True,
        vis_stuck_window: int = 30,
        vis_diff_thresh: float = 1.2,
        own_kill_user_var: int = 0,
        enable_weapon_actions: bool = True,
        use_game_reward: bool = True,
    ):
        super().__init__()
        self.cfg_path = cfg_path
        self.frame_skip = int(frame_skip)
        self.max_steps = int(max_steps)
        self.render_enabled = bool(render)

        self.persona = persona
        self.w = persona_weights(persona)

        self.obs_size = int(obs_size)
        self.stack = int(stack)
        if self.stack < 1:
            raise ValueError("stack must be >= 1")

        self.stuck_window = int(stuck_window)
        self.early_end_on_stuck = bool(early_end_on_stuck)
        self.vis_stuck_window = int(vis_stuck_window)
        self.vis_diff_thresh = float(vis_diff_thresh)

        self.enable_weapon_actions = bool(enable_weapon_actions)
        self.use_game_reward = bool(use_game_reward)

        self._np_random, _ = seeding.np_random(int(seed))
        self._seed = int(seed)

        self.debug = os.environ.get("DOOM_ENV_DEBUG", "0") == "1"

        # Init game
        self.game = DoomGame()
        self.game.load_config(cfg_path)

        try:
            self.game.set_objects_info_enabled(True)  # type: ignore[attr-defined]
        except Exception:
            pass

        self.game.set_window_visible(self.render_enabled)
        self.game.set_mode(Mode.ASYNC_PLAYER if self.render_enabled else Mode.PLAYER)

        self.game.set_screen_resolution(ScreenResolution.RES_320X240)
        self.game.set_screen_format(ScreenFormat.RGB24)

        self.game.set_seed(self._seed)
        self.game.init()

        self.available_buttons: List[Button] = list(self.game.get_available_buttons())
        self._btn_idx: Dict[str, int] = {b.name: i for i, b in enumerate(self.available_buttons)}

        self._vars = set(self.game.get_available_game_variables())

        # Optional "own kill counter" via USERk
        env_user_k = int(os.environ.get("DOOM_OWN_KILL_USER_VAR", "0") or "0")
        k = int(own_kill_user_var) if int(own_kill_user_var) > 0 else env_user_k
        self._own_kill_var: Optional[GameVariable] = None
        if 1 <= k <= 60:
            gv = getattr(GameVariable, f"USER{k}", None)
            if gv is not None and gv in self._vars:
                self._own_kill_var = gv

        self._assert_required_vars()

        # Action set
        self.actions: List[np.ndarray] = self._build_discrete_actions()
        self.action_space = spaces.Discrete(len(self.actions))

        self.observation_space = spaces.Box(
            low=0, high=255, shape=(self.stack, self.obs_size, self.obs_size), dtype=np.uint8
        )

        # Episode / internal counters
        self._step_count = 0
        self._stuck_steps = 0

        self._last_gray_stack: Optional[np.ndarray] = None
        self._frame_fifo: Optional[List[np.ndarray]] = None

        self._prev_gray_for_vis: Optional[np.ndarray] = None
        self._vis_static_steps = 0

        # Attribution state
        self._kill_credit = 0.0
        self._steps_since_damage = 999999
        self._steps_since_hit = 999999
        self._under_fire_steps = 0
        self._steps_since_attack = 999999

        # Weapon switch spam state
        self._steps_since_weapon_select = 999999
        self._weapon_select_streak = 0

        # Goal mode
        self._goal_mode: str = "enemy"

        # Target lock (NEW)
        self._target_id: Optional[int] = None
        self._target_hold: int = 0

        # Previous info for deltas
        self._prev_info: Optional[Dict[str, Any]] = None

        # Per-episode accumulators
        self._ep_killcount = 0.0
        self._ep_fragcount = 0.0
        self._ep_monster_kills_raw = 0.0
        self._ep_monster_kills = 0.0
        self._ep_suspicious_monster_kills = 0.0
        self._ep_damage_dealt = 0.0
        self._ep_damage_taken = 0.0
        self._ep_hits = 0.0
        self._ep_hits_taken = 0.0
        self._ep_deaths = 0.0

        if self.debug:
            print("[env] available_buttons:", [b.name for b in self.available_buttons])
            print("[env] available_vars:", [v.name for v in sorted(list(self._vars), key=lambda x: x.name)])
            print("[env] num_actions:", len(self.actions))
            if self._own_kill_var is not None:
                print("[env] own_kill_var enabled:", self._own_kill_var.name)
            else:
                print("[env] own_kill_var disabled (fallback attribution via combat credit)")
            print("[env] enable_weapon_actions:", self.enable_weapon_actions)
            print("[env] use_game_reward:", self.use_game_reward)

    # ---------- Helpers ----------

    @staticmethod
    def _norm_name(s: str) -> str:
        return str(s).strip().lower().replace(" ", "")

    def _has(self, var: GameVariable) -> bool:
        return var in self._vars

    def _get_var(self, var: GameVariable, default: float = 0.0) -> float:
        if not self._has(var):
            return default
        try:
            return float(self.game.get_game_variable(var))
        except Exception:
            return default

    def _assert_required_vars(self) -> None:
        required = [
            GameVariable.HEALTH,
            GameVariable.ARMOR,
            GameVariable.POSITION_X,
            GameVariable.POSITION_Y,
            GameVariable.ANGLE,
            GameVariable.KILLCOUNT,
            GameVariable.DAMAGECOUNT,
            GameVariable.DAMAGE_TAKEN,
            GameVariable.HITCOUNT,
            GameVariable.HITS_TAKEN,
            GameVariable.DEATHCOUNT,
            GameVariable.SELECTED_WEAPON,
            GameVariable.SELECTED_WEAPON_AMMO,
        ]
        missing = [v.name for v in required if v not in self._vars]
        if missing:
            raise RuntimeError(
                "Your cfg is missing required game variables: "
                + ", ".join(missing)
                + ".\nFix available_game_variables in cfg BEFORE training."
            )

        required_btns = ["ATTACK", "MOVE_FORWARD", "MOVE_BACKWARD"]
        missing_btns = [b for b in required_btns if b not in self._btn_idx]
        if missing_btns:
            raise RuntimeError(
                "Your cfg is missing required buttons: "
                + ", ".join(missing_btns)
                + ".\nFix available_buttons in cfg BEFORE training."
            )

    @staticmethod
    def _angle_delta_deg(a_now: float, a_prev: float) -> float:
        return ((a_now - a_prev + 180.0) % 360.0) - 180.0

    def _need_scale(self, val_prev: float, crit: float, exit_margin: float) -> float:
        """
        Piecewise-linear "need" in [0..1]:
          - if val <= crit -> 1
          - if val >= crit+margin -> 0
          - else linear decay in between
        """
        v = float(val_prev)
        c = float(crit)
        m = float(max(1e-6, exit_margin))
        if v <= c:
            return 1.0
        if v >= c + m:
            return 0.0
        return float(1.0 - (v - c) / m)

    def _get_rgb(self) -> np.ndarray:
        st = self.game.get_state()
        if st is None or st.screen_buffer is None:
            h, w = int(self.game.get_screen_height()), int(self.game.get_screen_width())
            return np.zeros((h, w, 3), dtype=np.uint8)

        buf = np.array(st.screen_buffer, copy=False)
        if buf.dtype != np.uint8:
            buf = buf.astype(np.uint8, copy=False)

        h, w = int(self.game.get_screen_height()), int(self.game.get_screen_width())

        if buf.ndim == 3 and buf.shape == (h, w, 3):
            rgb = buf
        elif buf.ndim == 3 and buf.shape == (3, h, w):
            rgb = np.transpose(buf, (1, 2, 0))
        elif buf.ndim == 1:
            rgb = buf.reshape(h, w, 3)
        else:
            if buf.ndim == 3 and buf.shape[-1] in (3, 4):
                rgb = buf[..., :3]
            elif buf.ndim == 3 and buf.shape[0] in (3, 4):
                rgb = np.transpose(buf[:3], (1, 2, 0))
            else:
                raise ValueError(f"Unexpected screen_buffer shape: {buf.shape}")

        return np.ascontiguousarray(rgb)

    def _preprocess_gray(self, rgb: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        gray = cv2.resize(gray, (self.obs_size, self.obs_size), interpolation=cv2.INTER_AREA)
        return gray

    def _get_obs_from_gray(self, gray: np.ndarray) -> np.ndarray:
        if self.stack == 1:
            self._last_gray_stack = gray
            return gray[None, :, :].astype(np.uint8)

        prev = self._last_gray_stack if self._last_gray_stack is not None else np.zeros_like(gray)
        self._last_gray_stack = gray

        if self.stack == 2:
            obs = np.stack([gray, prev], axis=0)
            return obs.astype(np.uint8)

        if self._frame_fifo is None:
            self._frame_fifo = [prev] * (self.stack - 1)
        self._frame_fifo = [gray] + self._frame_fifo[: self.stack - 1]
        obs = np.stack(self._frame_fifo, axis=0)
        return obs.astype(np.uint8)

    def _ammo_total(self, info: Dict[str, Any]) -> float:
        # strict: only selected weapon ammo
        return float(info.get("selected_weapon_ammo", 0.0))

    def _nearest_named_distance(self, info: Dict[str, Any], names: Set[str]) -> float:
        st = self.game.get_state()
        if st is None or not hasattr(st, "objects") or st.objects is None:
            return -1.0

        ax = float(info.get("pos_x", 0.0))
        ay = float(info.get("pos_y", 0.0))

        best = None
        for o in st.objects:
            nm = getattr(o, "name", None)
            if not nm:
                continue
            nl = self._norm_name(nm)
            if nl not in names:
                continue

            ox = float(getattr(o, "position_x", 0.0))
            oy = float(getattr(o, "position_y", 0.0))
            d = float(np.sqrt((ox - ax) ** 2 + (oy - ay) ** 2))
            best = d if best is None else min(best, d)

        return float(best) if best is not None else -1.0

    def _enemy_target_state(self, info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Returns enemy targeting info:
          enemy_dist              [-1 if none]
          enemy_visible           {0,1} (best-effort)
          enemy_angle_err         signed deg in [-180,180]
          enemy_angle_err_abs     abs deg in [0,180]
          enemy_target_id         int (or -1)
          enemy_on_target         {0,1} if abs_err <= aim_center_thr_deg
        Uses object id if available; otherwise falls back to stable index.
        Keeps last target for w.target_hold_steps while it remains present in objects.
        """
        st = self.game.get_state()
        if st is None or not hasattr(st, "objects") or st.objects is None:
            self._target_hold = max(0, self._target_hold - 1)
            if self._target_hold <= 0:
                self._target_id = None
            return {
                "enemy_dist": -1.0,
                "enemy_visible": 0.0,
                "enemy_angle_err": 0.0,
                "enemy_angle_err_abs": 180.0,
                "enemy_target_id": -1,
                "enemy_on_target": 0.0,
            }

        ax = float(info.get("pos_x", 0.0))
        ay = float(info.get("pos_y", 0.0))
        ang = float(info.get("angle", 0.0))

        enemies: List[Tuple[int, float, float]] = []  # (id, dist, err_signed)
        fallback_id = 1

        for o in st.objects:
            nm = getattr(o, "name", None)
            if not nm:
                continue
            nl = self._norm_name(nm)

            if nl in self.NON_ENEMY_NAMES:
                continue
            if "player" in nl or nl.startswith("marine"):
                continue
            if nl not in self.ENEMY_NAMES:
                continue

            vis_attr = getattr(o, "visible", None)
            if vis_attr is not None and (not bool(vis_attr)):
                continue

            ox = float(getattr(o, "position_x", 0.0))
            oy = float(getattr(o, "position_y", 0.0))
            dx = ox - ax
            dy = oy - ay
            dist = float(np.hypot(dx, dy))

            # Bearing from agent to object in degrees (0..360)
            bearing = (float(np.degrees(np.arctan2(dy, dx))) + 360.0) % 360.0
            # Signed error relative to agent angle
            err = float(self._angle_delta_deg(bearing, ang))

            oid = getattr(o, "id", None)
            if oid is None:
                oid = getattr(o, "object_id", None)
            if oid is None:
                oid = fallback_id
                fallback_id += 1

            enemies.append((int(oid), dist, err))

        if not enemies:
            self._target_hold = max(0, self._target_hold - 1)
            if self._target_hold <= 0:
                self._target_id = None
            return {
                "enemy_dist": -1.0,
                "enemy_visible": 0.0,
                "enemy_angle_err": 0.0,
                "enemy_angle_err_abs": 180.0,
                "enemy_target_id": -1,
                "enemy_on_target": 0.0,
            }

        chosen: Optional[Tuple[int, float, float]] = None
        if self._target_id is not None:
            for oid, dist, err in enemies:
                if oid == self._target_id:
                    chosen = (oid, dist, err)
                    break

        if chosen is None:
            chosen = min(enemies, key=lambda t: t[1])
            self._target_id = int(chosen[0])

        self._target_hold = int(self.w.target_hold_steps)

        oid, dist, err = chosen
        err_abs = float(abs(err))
        on_target = 1.0 if err_abs <= float(self.w.aim_center_thr_deg) else 0.0
        return {
            "enemy_dist": float(dist),
            "enemy_visible": 1.0,
            "enemy_angle_err": float(err),
            "enemy_angle_err_abs": float(err_abs),
            "enemy_target_id": int(oid),
            "enemy_on_target": float(on_target),
        }

    def _choose_goal_mode(self, info: Dict[str, Any]) -> str:
        """
        Strict priority policy:
          1) hp if hp <= goal_hp_crit
          2) ammo if ammo <= goal_ammo_crit
          3) enemy if enemy visible
          4) otherwise search
        With hysteresis for hp/ammo to avoid flicker.
        """
        w = self.w

        hp = float(info.get("health", 0.0))
        ammo = float(info.get("ammo_total", 0.0))
        enemy_visible = bool(float(info.get("enemy_visible", 0.0)) > 0.0)

        # Hysteresis keep mode until exit
        if self._goal_mode == "hp":
            if hp > (w.goal_hp_crit + w.goal_hp_exit_margin):
                self._goal_mode = "enemy"
            else:
                return "hp"

        if self._goal_mode == "ammo":
            if ammo > (w.goal_ammo_crit + w.goal_ammo_exit_margin):
                self._goal_mode = "enemy"
            else:
                return "ammo"

        if self._goal_mode == "search":
            if enemy_visible:
                self._goal_mode = "enemy"
            else:
                return "search"

        # Enter by strict priority
        if hp <= w.goal_hp_crit:
            self._goal_mode = "hp"
            return "hp"

        if ammo <= w.goal_ammo_crit:
            self._goal_mode = "ammo"
            return "ammo"

        if enemy_visible:
            self._goal_mode = "enemy"
            return "enemy"

        self._goal_mode = "search"
        return "search"

    def _get_info(self) -> Dict[str, Any]:
        health = self._get_var(GameVariable.HEALTH, 0.0)
        armor = self._get_var(GameVariable.ARMOR, 0.0)

        selected_weapon = self._get_var(GameVariable.SELECTED_WEAPON, 0.0)
        selected_weapon_ammo = self._get_var(GameVariable.SELECTED_WEAPON_AMMO, 0.0)

        killcount = self._get_var(GameVariable.KILLCOUNT, 0.0)
        fragcount = self._get_var(getattr(GameVariable, "FRAGCOUNT", GameVariable.KILLCOUNT), 0.0)
        deathcount = self._get_var(GameVariable.DEATHCOUNT, 0.0)

        damagecount = self._get_var(GameVariable.DAMAGECOUNT, 0.0)
        damage_taken = self._get_var(GameVariable.DAMAGE_TAKEN, 0.0)

        hitcount = self._get_var(GameVariable.HITCOUNT, 0.0)
        hits_taken = self._get_var(GameVariable.HITS_TAKEN, 0.0)

        px = self._get_var(GameVariable.POSITION_X, 0.0)
        py = self._get_var(GameVariable.POSITION_Y, 0.0)
        ang = self._get_var(GameVariable.ANGLE, 0.0)

        player_number = self._get_var(getattr(GameVariable, "PLAYER_NUMBER", GameVariable.HEALTH), 0.0)
        player_count = self._get_var(getattr(GameVariable, "PLAYER_COUNT", GameVariable.HEALTH), 0.0)

        dead = bool(self.game.is_player_dead())

        own_killcount = killcount
        if self._own_kill_var is not None:
            own_killcount = self._get_var(self._own_kill_var, killcount)

        info: Dict[str, Any] = {
            "health": float(health),
            "armor": float(armor),
            "selected_weapon": float(selected_weapon),
            "selected_weapon_ammo": float(selected_weapon_ammo),

            "killcount": float(killcount),
            "fragcount": float(fragcount),
            "deathcount": float(deathcount),
            "own_killcount": float(own_killcount),

            "damagecount": float(damagecount),
            "damage_taken": float(damage_taken),
            "hitcount": float(hitcount),
            "hits_taken": float(hits_taken),

            "pos_x": float(px),
            "pos_y": float(py),
            "angle": float(ang),
            "dead": dead,

            "player_number": float(player_number),
            "player_count": float(player_count),

            "step": int(self._step_count),
            "stuck_steps": int(self._stuck_steps),
            "vis_static_steps": int(self._vis_static_steps),

            "kill_credit": float(self._kill_credit),
            "steps_since_damage": int(self._steps_since_damage),
            "steps_since_hit": int(self._steps_since_hit),
            "steps_since_attack": int(self._steps_since_attack),

            "steps_since_weapon_select": int(self._steps_since_weapon_select),
            "weapon_select_streak": int(self._weapon_select_streak),

            "ep_killcount": float(self._ep_killcount),
            "ep_fragcount": float(self._ep_fragcount),
            "ep_monster_kills_raw": float(self._ep_monster_kills_raw),
            "ep_monster_kills": float(self._ep_monster_kills),
            "ep_suspicious_monster_kills": float(self._ep_suspicious_monster_kills),

            "ep_damage_dealt": float(self._ep_damage_dealt),
            "ep_damage_taken": float(self._ep_damage_taken),
            "ep_hits": float(self._ep_hits),
            "ep_hits_taken": float(self._ep_hits_taken),
            "ep_deaths": float(self._ep_deaths),

            "persona": self.persona,
        }
        info["ammo_total"] = float(self._ammo_total(info))
        return info

    # ---------- Actions ----------

    def _zero_action(self) -> np.ndarray:
        return np.zeros((len(self.available_buttons),), dtype=np.float32)

    def _set(self, a: np.ndarray, name: str, value: float) -> None:
        idx = self._btn_idx.get(name)
        if idx is None:
            return
        a[idx] = float(value)

    def _add_action(self, actions: List[np.ndarray], spec: List[Tuple[str, float]]) -> None:
        a = self._zero_action()
        ok = False
        for name, val in spec:
            if name in self._btn_idx:
                self._set(a, name, val)
                ok = True
        if ok:
            actions.append(a)

    def _build_discrete_actions(self) -> List[np.ndarray]:
        actions: List[np.ndarray] = []

        MF = "MOVE_FORWARD"
        MB = "MOVE_BACKWARD"
        LRD = "MOVE_LEFT_RIGHT_DELTA"
        TRD = "TURN_LEFT_RIGHT_DELTA"
        AT = "ATTACK"
        SP = "SPEED"

        STRAFE = 20.0
        TURN_AIM = 6.0
        TURN_RUN = 12.0
        TURN_FAST = 22.0

        actions.append(self._zero_action())                         # 00 NOOP

        self._add_action(actions, [(MF, 1.0)])                     # 01
        self._add_action(actions, [(MB, 1.0)])                     # 02

        self._add_action(actions, [(LRD, -STRAFE)])                # 03
        self._add_action(actions, [(LRD, +STRAFE)])                # 04

        self._add_action(actions, [(MF, 1.0), (LRD, -STRAFE)])     # 05
        self._add_action(actions, [(MF, 1.0), (LRD, +STRAFE)])     # 06

        self._add_action(actions, [(TRD, -TURN_AIM)])              # 07
        self._add_action(actions, [(TRD, +TURN_AIM)])              # 08

        self._add_action(actions, [(MF, 1.0), (TRD, -TURN_RUN)])   # 09
        self._add_action(actions, [(MF, 1.0), (TRD, +TURN_RUN)])   # 10

        self._add_action(actions, [(AT, 1.0)])                     # 11
        self._add_action(actions, [(AT, 1.0), (MF, 1.0)])          # 12
        self._add_action(actions, [(AT, 1.0), (TRD, -TURN_AIM)])   # 13
        self._add_action(actions, [(AT, 1.0), (TRD, +TURN_AIM)])   # 14
        self._add_action(actions, [(AT, 1.0), (LRD, -STRAFE)])     # 15
        self._add_action(actions, [(AT, 1.0), (LRD, +STRAFE)])     # 16

        self._add_action(actions, [(SP, 1.0), (MF, 1.0)])                          # 17
        self._add_action(actions, [(SP, 1.0), (MF, 1.0), (LRD, -STRAFE)])          # 18
        self._add_action(actions, [(SP, 1.0), (MF, 1.0), (LRD, +STRAFE)])          # 19
        self._add_action(actions, [(SP, 1.0), (MF, 1.0), (TRD, -TURN_FAST)])       # 20
        self._add_action(actions, [(SP, 1.0), (MF, 1.0), (TRD, +TURN_FAST)])       # 21

        if self.enable_weapon_actions:
            for i in range(1, 7):
                self._add_action(actions, [(f"SELECT_WEAPON{i}", 1.0)])
            self._add_action(actions, [("SELECT_NEXT_WEAPON", 1.0)])
            self._add_action(actions, [("SELECT_PREV_WEAPON", 1.0)])

        if len(actions) < 5:
            raise RuntimeError("Action set too small; check cfg buttons.")

        return actions

    def _action_flags(self, action_id: int) -> Dict[str, bool]:
        a_vec = self.actions[int(action_id)]

        def on(btn: str) -> bool:
            idx = self._btn_idx.get(btn)
            return bool(idx is not None and float(a_vec[idx]) != 0.0)

        is_attack = on("ATTACK")
        is_move = on("MOVE_FORWARD") or on("MOVE_BACKWARD") or on("MOVE_LEFT_RIGHT_DELTA")
        is_turn = on("TURN_LEFT_RIGHT_DELTA")

        is_weapon = False
        if self.enable_weapon_actions:
            if on("SELECT_NEXT_WEAPON") or on("SELECT_PREV_WEAPON"):
                is_weapon = True
            else:
                for i in range(1, 7):
                    if on(f"SELECT_WEAPON{i}"):
                        is_weapon = True
                        break

        return {"attack": is_attack, "move": is_move, "turn": is_turn, "weapon": is_weapon}

    # ---------- Reward ----------

    def _compute_reward(
        self,
        info: Dict[str, Any],
        prev: Optional[Dict[str, Any]],
        action_id: int,
        game_reward: float = 0.0,
    ) -> Tuple[float, Dict[str, float], float, float]:
        w = self.w
        comp: Dict[str, float] = {}

        def add(name: str, val: float) -> None:
            if val == 0.0:
                return
            comp[name] = comp.get(name, 0.0) + float(val)

        flags = self._action_flags(action_id)

        # Built-in scenario reward
        if self.use_game_reward and game_reward != 0.0:
            gr = float(np.clip(float(game_reward), -w.game_reward_clip, w.game_reward_clip))
            add("game", w.game_reward_scale * gr)

        if w.step_penalty != 0.0:
            add("step", -w.step_penalty)

        # movement delta
        dist = 0.0
        d_ang = 0.0
        if prev is not None:
            dx = float(info["pos_x"]) - float(prev.get("pos_x", info["pos_x"]))
            dy = float(info["pos_y"]) - float(prev.get("pos_y", info["pos_y"]))
            dist = float(np.sqrt(dx * dx + dy * dy))
            d_ang = float(self._angle_delta_deg(float(info["angle"]), float(prev.get("angle", info["angle"]))))

        move_term = min(dist / 16.0, 1.0)
        
        enemy_visible = bool(float(info.get("enemy_visible", 0.0)) > 0.0)
        if enemy_visible:
            enemy_dist_now = float(info.get("enemy_dist", -1.0))
            if enemy_dist_now >= 0 and enemy_dist_now <= w.engage_dist_max:
                move_term = min(move_term, 0.15)  # heavy cap during combat range
            else:
                move_term = min(move_term, 0.5)

        add("move", w.move * move_term)

        if self._under_fire_steps > 0:
            add("under_fire_move", w.under_fire_move_bonus * move_term)
            self._under_fire_steps -= 1

        if dist < 1.0:
            add("idle", -w.idle)

        if dist < 0.25 and abs(d_ang) > 1.0:
            add("spin", -w.spin_in_place * abs(d_ang))

        # bump / stuck logic
        if flags["move"] and dist < 0.25:
            add("bump", -w.bump)

        if dist < 0.25:
            self._stuck_steps += 1
        else:
            self._stuck_steps = 0

        if self._stuck_steps >= self.stuck_window:
            scale = min((self._stuck_steps - self.stuck_window + 1) / 20.0, 1.0)
            add("stuck", -w.stuck * scale)

        if prev is None:
            reward0 = float(np.clip(sum(comp.values()), w.clip_min, w.clip_max))
            return reward0, comp, 0.0, 0.0

        # ---- Deltas ----
        kill_d = max(0.0, float(info["killcount"]) - float(prev.get("killcount", 0.0)))
        frag_d = max(0.0, float(info.get("fragcount", 0.0)) - float(prev.get("fragcount", 0.0)))
        death_d = max(0.0, float(info.get("deathcount", 0.0)) - float(prev.get("deathcount", 0.0)))

        dmg_d = max(0.0, float(info["damagecount"]) - float(prev.get("damagecount", 0.0)))
        dmg_taken_d = max(0.0, float(info.get("damage_taken", 0.0)) - float(prev.get("damage_taken", 0.0)))

        hit_d = max(0.0, float(info.get("hitcount", 0.0)) - float(prev.get("hitcount", 0.0)))
        hits_taken_d = max(0.0, float(info.get("hits_taken", 0.0)) - float(prev.get("hits_taken", 0.0)))

        hp_now = float(info["health"])
        hp_prev = float(prev.get("health", hp_now))

        ar_now = float(info["armor"])
        ar_prev = float(prev.get("armor", ar_now))

        ammo_now = float(info.get("ammo_total", 0.0))
        ammo_prev = float(prev.get("ammo_total", ammo_now))

        hp_gain = max(0.0, hp_now - hp_prev)
        ar_gain = max(0.0, ar_now - ar_prev)
        ammo_gain = max(0.0, ammo_now - ammo_prev)

        ammo_sel_prev = float(prev.get("selected_weapon_ammo", float(info["selected_weapon_ammo"])))

        # ---- Enemy distance shaping ----
        enemy_now = float(info.get("enemy_dist", -1.0))
        enemy_prev = float(prev.get("enemy_dist", enemy_now))
        enemy_visible = bool(float(info.get("enemy_visible", 0.0)) > 0.0)

        err_abs_now = float(info.get("enemy_angle_err_abs", 180.0))
        err_abs_prev = float(prev.get("enemy_angle_err_abs", err_abs_now))

        if enemy_now >= 0.0 and enemy_prev >= 0.0:
            delta_close = float(enemy_prev - enemy_now)
            delta_close_norm = float(np.clip(delta_close / max(1e-6, w.enemy_dist_norm), -1.0, 1.0))

            retreat = (hp_prev <= w.enemy_retreat_low_health) or (ammo_prev <= w.enemy_retreat_low_ammo)

            if retreat:
                delta_far = float(enemy_now - enemy_prev)
                delta_far_norm = float(np.clip(delta_far / max(1e-6, w.enemy_dist_norm), -1.0, 1.0))
                add("enemy_dist_retreat", w.enemy_retreat * delta_far_norm)

                # Only panic if getting closer while trying to retreat, otherwise if escaping, don't penalize.
                if enemy_now < w.enemy_too_close and delta_far < 0.0:
                    panic_scale = float(
                        np.clip((w.enemy_too_close - enemy_now) / max(1e-6, w.enemy_too_close), 0.0, 1.0)
                    )
                    add("enemy_panic", -w.enemy_panic_penalty * panic_scale)
            else:
                add("enemy_dist", w.enemy_dist * delta_close_norm)

                if enemy_now < w.enemy_min_safe and delta_close_norm > 0.0:
                    denom = max(1e-6, (w.enemy_min_safe - w.enemy_too_close))
                    close_scale = float(np.clip((w.enemy_min_safe - enemy_now) / denom, 0.0, 1.0))
                    add("enemy_close", -w.enemy_close_penalty * close_scale * float(delta_close_norm))

        # ---- Aim shaping (NEW) ----
        if enemy_visible:
            d_err = float(err_abs_prev - err_abs_now)
            d_err_norm = float(np.clip(d_err / max(1e-6, w.aim_err_norm_deg), -1.0, 1.0))
            add("aim", w.aim * d_err_norm)

            if err_abs_now <= w.aim_center_thr_deg:
                add("aim_center", w.aim_center_bonus)
                if flags["attack"]:
                    add("attack_on_target", w.attack_on_target_bonus)
                else:
                    if enemy_now >= 0.0 and enemy_now <= w.engage_dist_max and ammo_prev > 0.0:
                        add("no_attack_on_target", -w.no_attack_on_target_penalty)

        # ---- Goal distance shaping ----
        goal_now = float(info.get("goal_dist", -1.0))
        goal_prev = float(prev.get("goal_dist", goal_now))
        goal_mode = str(info.get("goal_mode", prev.get("goal_mode", "enemy")))

        if goal_now >= 0.0 and goal_prev >= 0.0:
            d = float(np.clip(goal_prev - goal_now, -w.goal_dist_step_cap, w.goal_dist_step_cap))
            d_norm = float(np.clip(d / max(1e-6, w.goal_dist_norm), -1.0, 1.0))
            if goal_mode in ("hp", "ammo"):
                add("goal_dist_pickup", w.goal_dist_pickup * d_norm)
            elif goal_mode == "enemy":
                add("goal_dist_enemy_goal", w.goal_dist_enemy * d_norm)

        # ---- Search mode shaping ----
        if goal_mode == "search":
            add("search_move", w.search_move_bonus * move_term)
            turn_norm = min(abs(d_ang) / max(1e-6, w.search_turn_norm_deg), 1.0)
            add("search_turn", w.search_turn_bonus * turn_norm)
            if dist < 0.5:
                add("search_idle", -w.search_idle_penalty)

        # Update timers
        if dmg_d > 0:
            self._steps_since_damage = 0
        else:
            self._steps_since_damage += 1

        if hit_d > 0:
            self._steps_since_hit = 0
        else:
            self._steps_since_hit += 1

        if hits_taken_d > 0 or dmg_taken_d > 0:
            self._under_fire_steps = max(self._under_fire_steps, 12)

        # Damage/hit shaping
        add("damage", w.damage * dmg_d)
        add("hit", w.hit * hit_d)

        # Damage taken shaping (scale more when low HP)
        if dmg_taken_d > 0:
            mult = 1.0
            if hp_prev <= w.low_health_thr:
                mult *= w.low_health_damage_taken_mult
            add("damage_taken", -w.damage_taken * dmg_taken_d * mult)

        add("hits_taken", -w.hits_taken * hits_taken_d)

        # ---- Pickups shaping (NEED-BASED + CAPS) ----
        if hp_gain > 0:
            gain = float(min(hp_gain, w.health_pickup_cap))
            need = self._need_scale(hp_prev, w.goal_hp_crit, w.goal_hp_exit_margin)
            mult = w.low_health_pickup_mult if hp_prev <= w.goal_hp_crit else 1.0
            add("hp_pickup", w.health_pickup * gain * need * mult)

        if ammo_gain > 0:
            gain = float(min(ammo_gain, w.ammo_pickup_cap))
            need = self._need_scale(ammo_prev, w.goal_ammo_crit, w.goal_ammo_exit_margin)
            mult = w.low_ammo_pickup_mult if ammo_prev <= w.goal_ammo_crit else 1.0
            add("ammo_pickup", w.ammo_pickup * gain * need * mult)

        if ar_gain > 0:
            gain = float(min(ar_gain, w.armor_pickup_cap))
            need = 1.0 if ar_prev <= 0.0 else float(w.armor_have_scale)
            add("armor_pickup", w.armor_pickup * gain * need)

        # ---- Attribution credit update ----
        decay = w.kill_credit_decay * (float(self.frame_skip) / 4.0)
        self._kill_credit = max(0.0, float(self._kill_credit) - float(decay))

        if dmg_d > 0:
            self._kill_credit = min(w.kill_credit_cap, self._kill_credit + dmg_d * w.kill_credit_damage_scale)
        if hit_d > 0:
            self._kill_credit = min(w.kill_credit_cap, self._kill_credit + hit_d * w.kill_credit_hit_scale)

        # ---- Monster kill raw delta ----
        monster_kill_raw = 0.0
        if self._own_kill_var is not None:
            own_now = float(info.get("own_killcount", 0.0))
            own_prev = float(prev.get("own_killcount", own_now))
            monster_kill_raw = max(0.0, own_now - own_prev)
        else:
            monster_kill_raw = max(0.0, kill_d - frag_d)

        if frag_d > 0:
            add("frag_penalty", -w.frag_penalty * frag_d)

        # ---- Attribute monster kills ----
        monster_kill_attrib = 0.0
        if monster_kill_raw > 0:
            recent_hit_or_damage = (
                (self._steps_since_hit <= w.kill_requires_recent_hit_steps)
                or (self._steps_since_damage <= w.kill_requires_recent_damage_steps)
            )
            recent_attack = (self._steps_since_attack <= w.kill_requires_recent_attack_steps)
            hard_gate_ok = bool(recent_attack and recent_hit_or_damage)

            if self._own_kill_var is not None:
                monster_kill_attrib = monster_kill_raw
            else:
                if hard_gate_ok and self._kill_credit > 0.0:
                    cost = w.kill_credit_cost_per_kill * monster_kill_raw
                    ratio = min(1.0, self._kill_credit / max(1e-6, cost))
                    monster_kill_attrib = monster_kill_raw * ratio
                    self._kill_credit = max(
                        0.0,
                        self._kill_credit - (w.kill_credit_cost_per_kill * monster_kill_attrib),
                    )
                else:
                    monster_kill_attrib = 0.0

            if monster_kill_attrib > 0.0 and (not self.use_game_reward):
                add("monster_kill", w.monster_kill * monster_kill_attrib)

        # ---- inactive / no-effect penalty ----
        nothing_happened = (
            (dmg_d <= 0.0)
            and (hit_d <= 0.0)
            and (monster_kill_attrib <= 0.0)
            and (hp_gain <= 0.0)
            and (ar_gain <= 0.0)
            and (ammo_gain <= 0.0)
        )
        if (
            nothing_happened
            and (not flags["move"])
            and (not flags["turn"])
            and (not flags["attack"])
            and dist < 0.25
            and abs(d_ang) < 1.0
        ):
            add("inactive", -w.inactive)

        # ---- weapon switch hygiene ----
        prev_weapon = float(prev.get("selected_weapon", float(info["selected_weapon"])))
        weapon_now = float(info["selected_weapon"])
        weapon_changed = bool(weapon_now != prev_weapon)

        if flags["weapon"]:
            if weapon_changed:
                add("weapon_switch", -w.weapon_switch)
            else:
                add("weapon_switch_noop", -w.weapon_switch_noop)

            if nothing_happened:
                add("weapon_switch_unproductive", -w.weapon_switch_unproductive)

            if self._steps_since_weapon_select <= w.weapon_switch_spam_window:
                self._weapon_select_streak += 1
            else:
                self._weapon_select_streak = 1
            self._steps_since_weapon_select = 0

            if self._weapon_select_streak > 1:
                add("weapon_switch_spam", -w.weapon_switch_spam_penalty * float(self._weapon_select_streak - 1))
        else:
            self._steps_since_weapon_select += 1
            if self._steps_since_weapon_select > w.weapon_switch_spam_window:
                self._weapon_select_streak = 0

        # Shooting hygiene
        if flags["attack"]:
            shoot_scale = 1.0
            if enemy_visible:
                if err_abs_now <= w.aim_center_thr_deg * 2.0:
                    shoot_scale = 0.25
                elif err_abs_now <= w.aim_err_norm_deg:
                    shoot_scale = 0.50
                elif err_abs_now <= w.aim_err_norm_deg * 2.0:
                    shoot_scale = 0.80

            if ammo_now <= 0.0:
                add("shoot_no_ammo", -w.shoot_no_ammo)

            if (dmg_d <= 0.0) and (hit_d <= 0.0) and (monster_kill_attrib <= 0.0):
                add("shoot_no_damage", -w.shoot_no_damage * shoot_scale)

            if (hit_d <= 0.0) and (monster_kill_attrib <= 0.0):
                add("shoot_no_hit", -w.shoot_no_hit * shoot_scale)

            ammo_sel_now = float(info.get("selected_weapon_ammo", 0.0))
            ammo_spent = max(0.0, ammo_sel_prev - ammo_sel_now)
            if (
                ammo_spent > 0.0
                and (dmg_d <= 0.0)
                and (hit_d <= 0.0)
                and (monster_kill_attrib <= 0.0)
            ):
                add("shoot_waste_ammo", -w.shoot_waste_ammo * ammo_spent * shoot_scale)

        if death_d > 0:
            add("death", -w.death * death_d)

        raw_reward = float(sum(comp.values()))
        clipped = float(np.clip(raw_reward, w.clip_min, w.clip_max))
        if clipped != raw_reward:
            comp["clip_corr"] = comp.get("clip_corr", 0.0) + (clipped - raw_reward)

        return clipped, comp, float(monster_kill_attrib), float(monster_kill_raw)

    # ---------- Gym API ----------

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        if seed is not None:
            self._seed = int(seed)
            self._np_random, _ = seeding.np_random(self._seed)
            try:
                self.game.set_seed(self._seed)
            except Exception:
                pass

        self.game.new_episode()

        self._step_count = 0
        self._stuck_steps = 0
        self._vis_static_steps = 0

        self._kill_credit = 0.0
        self._steps_since_damage = 999999
        self._steps_since_hit = 999999
        self._under_fire_steps = 0
        self._steps_since_attack = 999999

        self._steps_since_weapon_select = 999999
        self._weapon_select_streak = 0

        self._goal_mode = "enemy"

        self._target_id = None
        self._target_hold = 0

        self._ep_killcount = 0.0
        self._ep_fragcount = 0.0
        self._ep_monster_kills_raw = 0.0
        self._ep_monster_kills = 0.0
        self._ep_suspicious_monster_kills = 0.0
        self._ep_damage_dealt = 0.0
        self._ep_damage_taken = 0.0
        self._ep_hits = 0.0
        self._ep_hits_taken = 0.0
        self._ep_deaths = 0.0

        self._last_gray_stack = None
        self._frame_fifo = None
        self._prev_gray_for_vis = None

        rgb = self._get_rgb()
        gray = self._preprocess_gray(rgb)
        obs = self._get_obs_from_gray(gray)

        info = self._get_info()
        info.update(self._enemy_target_state(info))

        info["goal_mode"] = self._choose_goal_mode(info)

        if info["goal_mode"] == "hp":
            info["goal_dist"] = float(self._nearest_named_distance(info, self.HEALTH_NAMES))
        elif info["goal_mode"] == "ammo":
            info["goal_dist"] = float(self._nearest_named_distance(info, self.AMMO_NAMES))
        elif info["goal_mode"] == "enemy":
            info["goal_dist"] = float(info.get("enemy_dist", -1.0))
        else:
            info["goal_dist"] = -1.0

        info["enemy_dist_delta"] = 0.0
        info["goal_dist_delta"] = 0.0

        self._prev_info = {
            "health": float(info["health"]),
            "armor": float(info["armor"]),
            "selected_weapon": float(info["selected_weapon"]),
            "selected_weapon_ammo": float(info["selected_weapon_ammo"]),
            "ammo_total": float(info.get("ammo_total", 0.0)),

            "killcount": float(info["killcount"]),
            "fragcount": float(info.get("fragcount", 0.0)),
            "deathcount": float(info.get("deathcount", 0.0)),
            "own_killcount": float(info.get("own_killcount", info["killcount"])),

            "damagecount": float(info["damagecount"]),
            "damage_taken": float(info.get("damage_taken", 0.0)),
            "hitcount": float(info.get("hitcount", 0.0)),
            "hits_taken": float(info.get("hits_taken", 0.0)),

            "pos_x": float(info["pos_x"]),
            "pos_y": float(info["pos_y"]),
            "angle": float(info["angle"]),

            "enemy_dist": float(info.get("enemy_dist", -1.0)),
            "enemy_visible": float(info.get("enemy_visible", 0.0)),
            "enemy_angle_err_abs": float(info.get("enemy_angle_err_abs", 180.0)),

            "goal_mode": str(info.get("goal_mode", "enemy")),
            "goal_dist": float(info.get("goal_dist", -1.0)),
        }

        if _USING_GYMNASIUM:
            return obs, info
        return obs

    def step(self, action: int):
        self._step_count += 1
        action_id = int(action)
        a_vec = self.actions[action_id]
        flags = self._action_flags(action_id)

        if flags["attack"]:
            self._steps_since_attack = 0
        else:
            self._steps_since_attack += 1

        game_reward = float(self.game.make_action(a_vec.tolist(), self.frame_skip))

        dead_now = bool(self.game.is_player_dead())
        episode_finished = bool(self.game.is_episode_finished())

        terminated = bool(episode_finished or dead_now)
        truncated = bool(self._step_count >= self.max_steps)

        rgb = self._get_rgb()
        gray = self._preprocess_gray(rgb)

        if self._prev_gray_for_vis is not None:
            diff = float(np.mean(np.abs(gray.astype(np.float32) - self._prev_gray_for_vis.astype(np.float32))))
            if diff < self.vis_diff_thresh:
                self._vis_static_steps += 1
            else:
                self._vis_static_steps = 0
        self._prev_gray_for_vis = gray

        obs = self._get_obs_from_gray(gray)
        info = self._get_info()

        info.update(self._enemy_target_state(info))
        if self._prev_info is not None:
            prev_enemy = float(self._prev_info.get("enemy_dist", -1.0))
            cur_enemy = float(info["enemy_dist"])
            info["enemy_dist_delta"] = float(cur_enemy - prev_enemy) if (prev_enemy >= 0.0 and cur_enemy >= 0.0) else 0.0
        else:
            info["enemy_dist_delta"] = 0.0

        info["goal_mode"] = self._choose_goal_mode(info)

        if info["goal_mode"] == "hp":
            info["goal_dist"] = float(self._nearest_named_distance(info, self.HEALTH_NAMES))
        elif info["goal_mode"] == "ammo":
            info["goal_dist"] = float(self._nearest_named_distance(info, self.AMMO_NAMES))
        elif info["goal_mode"] == "enemy":
            info["goal_dist"] = float(info.get("enemy_dist", -1.0))
        else:
            info["goal_dist"] = -1.0

        if self._prev_info is not None:
            prev_goal = float(self._prev_info.get("goal_dist", -1.0))
            cur_goal = float(info["goal_dist"])
            info["goal_dist_delta"] = float(cur_goal - prev_goal) if (prev_goal >= 0.0 and cur_goal >= 0.0) else 0.0
        else:
            info["goal_dist_delta"] = 0.0

        info["terminated_by_death"] = bool(dead_now and not episode_finished)
        info["terminated_by_game"] = bool(episode_finished)
        info["terminated_by_stuck"] = False

        info["weapon_select_pressed"] = 1.0 if flags["weapon"] else 0.0
        prev_weapon = float(self._prev_info.get("selected_weapon", info["selected_weapon"])) if self._prev_info else float(info["selected_weapon"])
        info["weapon_changed"] = 1.0 if float(info["selected_weapon"]) != prev_weapon else 0.0

        reward, comp, mk_attrib_step, mk_raw_step = self._compute_reward(
            info, self._prev_info, action_id, game_reward=game_reward
        )

        info["r_total"] = float(reward)
        info["game_reward"] = float(game_reward)
        for k, v in comp.items():
            info[f"r_{k}"] = float(v)
        info["monster_kill_raw_step"] = float(mk_raw_step)
        info["monster_kill_attrib_step"] = float(mk_attrib_step)

        if self._prev_info is not None:
            kill_d = max(0.0, float(info["killcount"]) - float(self._prev_info.get("killcount", 0.0)))
            frag_d = max(0.0, float(info.get("fragcount", 0.0)) - float(self._prev_info.get("fragcount", 0.0)))

            self._ep_killcount += kill_d
            self._ep_fragcount += frag_d
            self._ep_monster_kills_raw += mk_raw_step
            self._ep_monster_kills += mk_attrib_step
            self._ep_suspicious_monster_kills += max(0.0, mk_raw_step - mk_attrib_step)

            dmg_d = max(0.0, float(info["damagecount"]) - float(self._prev_info.get("damagecount", 0.0)))
            dmg_taken_d = max(0.0, float(info.get("damage_taken", 0.0)) - float(self._prev_info.get("damage_taken", 0.0)))
            hit_d = max(0.0, float(info.get("hitcount", 0.0)) - float(self._prev_info.get("hitcount", 0.0)))
            hits_taken_d = max(0.0, float(info.get("hits_taken", 0.0)) - float(self._prev_info.get("hits_taken", 0.0)))
            death_d = max(0.0, float(info.get("deathcount", 0.0)) - float(self._prev_info.get("deathcount", 0.0)))

            self._ep_damage_dealt += dmg_d
            self._ep_damage_taken += dmg_taken_d
            self._ep_hits += hit_d
            self._ep_hits_taken += hits_taken_d
            self._ep_deaths += death_d

            info["ep_killcount"] = float(self._ep_killcount)
            info["ep_fragcount"] = float(self._ep_fragcount)
            info["ep_monster_kills_raw"] = float(self._ep_monster_kills_raw)
            info["ep_monster_kills"] = float(self._ep_monster_kills)
            info["ep_suspicious_monster_kills"] = float(self._ep_suspicious_monster_kills)

            info["ep_damage_dealt"] = float(self._ep_damage_dealt)
            info["ep_damage_taken"] = float(self._ep_damage_taken)
            info["ep_hits"] = float(self._ep_hits)
            info["ep_hits_taken"] = float(self._ep_hits_taken)
            info["ep_deaths"] = float(self._ep_deaths)

        self._prev_info = {
            "health": float(info["health"]),
            "armor": float(info["armor"]),
            "selected_weapon": float(info["selected_weapon"]),
            "selected_weapon_ammo": float(info["selected_weapon_ammo"]),
            "ammo_total": float(info.get("ammo_total", 0.0)),

            "killcount": float(info["killcount"]),
            "fragcount": float(info.get("fragcount", 0.0)),
            "deathcount": float(info.get("deathcount", 0.0)),
            "own_killcount": float(info.get("own_killcount", info["killcount"])),

            "damagecount": float(info["damagecount"]),
            "damage_taken": float(info.get("damage_taken", 0.0)),
            "hitcount": float(info.get("hitcount", 0.0)),
            "hits_taken": float(info.get("hits_taken", 0.0)),

            "pos_x": float(info["pos_x"]),
            "pos_y": float(info["pos_y"]),
            "angle": float(info["angle"]),

            "enemy_dist": float(info.get("enemy_dist", -1.0)),
            "enemy_visible": float(info.get("enemy_visible", 0.0)),
            "enemy_angle_err_abs": float(info.get("enemy_angle_err_abs", 180.0)),

            "goal_mode": str(info.get("goal_mode", "enemy")),
            "goal_dist": float(info.get("goal_dist", -1.0)),
        }

        if self.early_end_on_stuck and (not terminated):
            stuck_trigger = (self._stuck_steps >= self.stuck_window * 3) or (self._vis_static_steps >= self.vis_stuck_window)
            if stuck_trigger:
                terminated = True
                truncated = False
                info["terminated_by_stuck"] = True

                penalty = float(self.w.stuck_end_penalty)
                comp["stuck_end"] = comp.get("stuck_end", 0.0) - penalty
                reward = float(np.clip(float(reward) - penalty, self.w.clip_min, self.w.clip_max))

                info["r_total"] = float(reward)
                info["r_stuck_end"] = float(comp["stuck_end"])
                for k, v in comp.items():
                    info[f"r_{k}"] = float(v)

        if _USING_GYMNASIUM:
            return obs, float(reward), terminated, truncated, info
        done = bool(terminated or truncated)
        return obs, float(reward), done, info

    def close(self):
        try:
            self.game.close()
        except Exception:
            pass


def make_env(
    cfg: str,
    frame_skip: int,
    max_steps: int,
    seed: int,
    render: bool = False,
    persona: str = "rusher",
    own_kill_user_var: int = 0,
    enable_weapon_actions: bool = True,
    use_game_reward: bool = True,
):
    def _thunk():
        return DoomDeathmatchEnv(
            cfg_path=cfg,
            frame_skip=frame_skip,
            max_steps=max_steps,
            seed=seed,
            render=render,
            persona=persona,
            early_end_on_stuck=True,
            own_kill_user_var=own_kill_user_var,
            enable_weapon_actions=enable_weapon_actions,
            use_game_reward=use_game_reward,
        )
    return _thunk
