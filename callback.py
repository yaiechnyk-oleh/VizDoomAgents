from __future__ import annotations

import os
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class ActionDiagnosticsCallback(BaseCallback):
    """
    Logs action-collapse diagnostics:
      - top action fraction
      - unique actions in rollout window
    """

    def __init__(self, verbose: int = 0, print_every_rollout: bool = True):
        super().__init__(verbose)
        self.print_every_rollout = print_every_rollout
        self.counts = Counter()

    def _on_step(self) -> bool:
        actions = self.locals.get("actions", None)
        if actions is not None:
            a = np.array(actions).reshape(-1)
            for x in a.tolist():
                self.counts[int(x)] += 1
        return True

    def _on_rollout_end(self) -> None:
        total = sum(self.counts.values())
        if total <= 0:
            return
        top_id, top_cnt = self.counts.most_common(1)[0]
        top_frac = float(top_cnt) / float(total)
        unique = len(self.counts)

        top3 = self.counts.most_common(3)
        top3_frac = float(sum(c for _, c in top3)) / float(total) if total > 0 else 0.0

        self.logger.record("diagnostics/action_top_frac", top_frac)
        self.logger.record("diagnostics/action_top_id", float(top_id))
        self.logger.record("diagnostics/action_unique", float(unique))
        self.logger.record("diagnostics/action_top3_frac", float(top3_frac))

        if self.print_every_rollout:
            print(f"[diag] top_frac={top_frac:.4f} top3_frac={top3_frac:.4f} top_id={top_id} unique={unique}")

        self.counts.clear()


class EntropyAnnealCallback(BaseCallback):
    """
    Linearly anneal ent_coef from start -> end over total_timesteps.
    """

    def __init__(self, ent_start: float, ent_end: float, total_timesteps: int, verbose: int = 0):
        super().__init__(verbose)
        self.ent_start = float(ent_start)
        self.ent_end = float(ent_end)
        self.total_timesteps = int(total_timesteps)

    def _on_step(self) -> bool:
        t = min(float(self.num_timesteps) / max(1.0, float(self.total_timesteps)), 1.0)
        ent = (1.0 - t) * self.ent_start + t * self.ent_end
        try:
            self.model.ent_coef = float(ent)
        except Exception:
            pass
        self.logger.record("diagnostics/ent_coef", float(ent))
        return True


class InfoStatsCallback(BaseCallback):
    """
    Collect mean of selected info keys over rollout and log them.
    Useful to ensure reward components behave as expected and to spot reward hacking.
    """

    def __init__(self, keys: Iterable[str], prefix: str = "info", verbose: int = 0, print_every_rollout: bool = False):
        super().__init__(verbose)
        self.keys = list(keys)
        self.prefix = str(prefix).strip()
        self.print_every_rollout = bool(print_every_rollout)
        self._acc = defaultdict(list)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", None)
        if not infos:
            return True
        for inf in infos:
            if not isinstance(inf, dict):
                continue
            for k in self.keys:
                if k in inf:
                    try:
                        self._acc[k].append(float(inf[k]))
                    except Exception:
                        pass
        return True

    def _on_rollout_end(self) -> None:
        if not self._acc:
            return
        for k, vals in self._acc.items():
            if not vals:
                continue
            m = float(np.mean(vals))
            self.logger.record(f"{self.prefix}/{k}", m)
        if self.print_every_rollout:
            msg = " ".join([f"{k}={float(np.mean(v)):.3f}" for k, v in self._acc.items() if v])
            print(f"[info-stats] {msg}")
        self._acc.clear()


class GoalModeStatsCallback(BaseCallback):
    """
    Tracks distribution of goal_mode over rollout:
      enemy / hp / ammo / armor / search
    Logs fractions so you can see if agent is stuck in "search" or "pickup" too often.
    """

    MODES = ("enemy", "hp", "ammo", "armor", "search")

    def __init__(self, verbose: int = 0, print_every_rollout: bool = False):
        super().__init__(verbose)
        self.print_every_rollout = bool(print_every_rollout)
        self._counts = Counter()
        self._total = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", None)
        if not infos:
            return True
        for inf in infos:
            if not isinstance(inf, dict):
                continue
            gm = inf.get("goal_mode", None)
            if isinstance(gm, str) and gm:
                self._counts[gm] += 1
                self._total += 1
        return True

    def _on_rollout_end(self) -> None:
        if self._total <= 0:
            self._counts.clear()
            self._total = 0
            return

        for m in self.MODES:
            frac = float(self._counts.get(m, 0)) / float(self._total)
            self.logger.record(f"goal_mode/{m}_frac", float(frac))

        other = self._total - sum(self._counts.get(m, 0) for m in self.MODES)
        if other > 0:
            self.logger.record("goal_mode/other_frac", float(other) / float(self._total))

        if self.print_every_rollout:
            msg = " ".join([f"{m}={self._counts.get(m,0)/self._total:.2f}" for m in self.MODES])
            print(f"[goal-mode] {msg}")

        self._counts.clear()
        self._total = 0


class EpisodeEndDiagnosticsCallback(BaseCallback):
    """
    Tracks why episodes end and logs fractions over rollout window:
      - terminated_by_death
      - terminated_by_game
      - terminated_by_stuck
      - TimeLimit.truncated (or truncated)
    Also logs mean episode length from VecMonitor if available.
    """

    def __init__(self, verbose: int = 1):
        super().__init__(verbose)
        self._n_episodes = 0
        self._by_death = 0
        self._by_game = 0
        self._by_stuck = 0
        self._by_trunc = 0
        self._ep_lens: List[float] = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", None)
        dones = self.locals.get("dones", None)
        if infos is None or dones is None:
            return True

        try:
            dones = np.array(dones).reshape(-1).tolist()
        except Exception:
            return True

        for i, d in enumerate(dones):
            if not d:
                continue
            inf = infos[i] if i < len(infos) else None
            if not isinstance(inf, dict):
                continue

            self._n_episodes += 1

            if bool(inf.get("terminated_by_stuck", False)):
                self._by_stuck += 1
            elif bool(inf.get("terminated_by_death", False)):
                self._by_death += 1
            elif bool(inf.get("terminated_by_game", False)):
                self._by_game += 1
            else:
                tl = inf.get("TimeLimit.truncated", None)
                if tl is True or bool(inf.get("truncated", False)):
                    self._by_trunc += 1

            ep = inf.get("episode", None)
            if isinstance(ep, dict) and "l" in ep:
                try:
                    self._ep_lens.append(float(ep["l"]))
                except Exception:
                    pass

        return True

    def _on_rollout_end(self) -> None:
        n = float(self._n_episodes)
        if n <= 0:
            return

        self.logger.record("episode_end/n_episodes_rollout", float(self._n_episodes))
        self.logger.record("episode_end/death_frac", float(self._by_death) / n)
        self.logger.record("episode_end/game_frac", float(self._by_game) / n)
        self.logger.record("episode_end/stuck_frac", float(self._by_stuck) / n)
        self.logger.record("episode_end/trunc_frac", float(self._by_trunc) / n)

        if self._ep_lens:
            self.logger.record("episode_end/mean_ep_len_rollout", float(np.mean(self._ep_lens)))

        if self.verbose:
            print(
                f"[end] eps={self._n_episodes} "
                f"death={self._by_death/n:.2f} game={self._by_game/n:.2f} "
                f"stuck={self._by_stuck/n:.2f} trunc={self._by_trunc/n:.2f} "
                f"mean_len={float(np.mean(self._ep_lens)) if self._ep_lens else 0.0:.1f}"
            )

        self._n_episodes = 0
        self._by_death = 0
        self._by_game = 0
        self._by_stuck = 0
        self._by_trunc = 0
        self._ep_lens.clear()


class RewardHackAlertCallback(BaseCallback):
    """
    Alerts if something suspicious happens:
      - raw monster-kill signals but no attribution (possible killcount contamination / hacks)
      - excessive time stuck/static
      - weirdly high weapon-select behavior (degenerate policies)
      - low combat signal (damage/hit/game) while moving a lot (pure running)
      - extremely high search fraction (policy not engaging combat)
    """

    def __init__(
        self,
        warn_suspicious_raw_per_rollout: float = 0.5,
        warn_mean_vis_static_steps: float = 25.0,
        warn_weapon_select_frac: float = 0.35,
        warn_stuck_term_frac: float = 0.25,
        warn_low_combat_signal: float = 0.05,
        warn_search_frac: float = 0.70,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.warn_suspicious_raw_per_rollout = float(warn_suspicious_raw_per_rollout)
        self.warn_mean_vis_static_steps = float(warn_mean_vis_static_steps)
        self.warn_weapon_select_frac = float(warn_weapon_select_frac)
        self.warn_stuck_term_frac = float(warn_stuck_term_frac)
        self.warn_low_combat_signal = float(warn_low_combat_signal)
        self.warn_search_frac = float(warn_search_frac)

        self._susp_raw = 0.0
        self._vis_static: List[float] = []
        self._stuck_steps: List[float] = []

        self._weapon_sel = 0.0
        self._total = 0.0

        # reward components (means)
        self._r_damage: List[float] = []
        self._r_hit: List[float] = []
        self._r_game: List[float] = []
        self._r_move: List[float] = []
        self._r_shoot_waste: List[float] = []
        self._r_shoot_no_dmg: List[float] = []
        self._r_goal_pickup: List[float] = []
        self._r_goal_enemy: List[float] = []

        # search shaping
        self._r_search_move: List[float] = []
        self._r_search_turn: List[float] = []
        self._r_search_idle: List[float] = []

        # goal_mode distribution
        self._gm_counts = Counter()
        self._gm_total = 0

        # end reasons inside rollout
        self._n_episodes = 0
        self._stuck_terms = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", None)
        actions = self.locals.get("actions", None)
        dones = self.locals.get("dones", None)

        if actions is not None:
            a = np.array(actions).reshape(-1).tolist()
            self._total += float(len(a))

        if infos:
            for inf in infos:
                if not isinstance(inf, dict):
                    continue

                # goal_mode counts
                gm = inf.get("goal_mode", None)
                if isinstance(gm, str) and gm:
                    self._gm_counts[gm] += 1
                    self._gm_total += 1

                # suspicious raw kills
                try:
                    raw = float(inf.get("monster_kill_raw_step", 0.0))
                    att = float(inf.get("monster_kill_attrib_step", 0.0))
                    self._susp_raw += max(0.0, raw - att)
                except Exception:
                    pass

                # static/stuck
                try:
                    self._vis_static.append(float(inf.get("vis_static_steps", 0.0)))
                except Exception:
                    pass
                try:
                    self._stuck_steps.append(float(inf.get("stuck_steps", 0.0)))
                except Exception:
                    pass

                # weapon select fraction
                try:
                    self._weapon_sel += float(inf.get("weapon_select_pressed", 0.0))
                except Exception:
                    pass

                # reward components (if present)
                if "r_damage" in inf:
                    try: self._r_damage.append(float(inf["r_damage"]))
                    except Exception: pass
                if "r_hit" in inf:
                    try: self._r_hit.append(float(inf["r_hit"]))
                    except Exception: pass
                if "r_game" in inf:
                    try: self._r_game.append(float(inf["r_game"]))
                    except Exception: pass
                if "r_move" in inf:
                    try: self._r_move.append(float(inf["r_move"]))
                    except Exception: pass
                if "r_shoot_waste_ammo" in inf:
                    try: self._r_shoot_waste.append(float(inf["r_shoot_waste_ammo"]))
                    except Exception: pass
                if "r_shoot_no_damage" in inf:
                    try: self._r_shoot_no_dmg.append(float(inf["r_shoot_no_damage"]))
                    except Exception: pass
                if "r_goal_dist_pickup" in inf:
                    try: self._r_goal_pickup.append(float(inf["r_goal_dist_pickup"]))
                    except Exception: pass
                if "r_goal_dist_enemy_goal" in inf:
                    try: self._r_goal_enemy.append(float(inf["r_goal_dist_enemy_goal"]))
                    except Exception: pass

                # search reward components
                if "r_search_move" in inf:
                    try: self._r_search_move.append(float(inf["r_search_move"]))
                    except Exception: pass
                if "r_search_turn" in inf:
                    try: self._r_search_turn.append(float(inf["r_search_turn"]))
                    except Exception: pass
                if "r_search_idle" in inf:
                    try: self._r_search_idle.append(float(inf["r_search_idle"]))
                    except Exception: pass

        # episode ends
        if dones is not None and infos is not None:
            try:
                dones_list = np.array(dones).reshape(-1).tolist()
            except Exception:
                dones_list = []
            for i, d in enumerate(dones_list):
                if not d:
                    continue
                if i >= len(infos) or not isinstance(infos[i], dict):
                    continue
                self._n_episodes += 1
                if bool(infos[i].get("terminated_by_stuck", False)):
                    self._stuck_terms += 1

        return True

    def _on_rollout_end(self) -> None:
        mean = lambda xs: float(np.mean(xs)) if xs else 0.0

        mean_vis = mean(self._vis_static)
        mean_stuck = mean(self._stuck_steps)
        weapon_frac = (float(self._weapon_sel) / float(self._total)) if self._total > 0 else 0.0

        m_r_damage = mean(self._r_damage)
        m_r_hit = mean(self._r_hit)
        m_r_game = mean(self._r_game)
        m_r_move = mean(self._r_move)

        m_r_shoot_waste = mean(self._r_shoot_waste)
        m_r_shoot_no_dmg = mean(self._r_shoot_no_dmg)

        stuck_term_frac = (float(self._stuck_terms) / float(self._n_episodes)) if self._n_episodes > 0 else 0.0
        combat_signal = float(m_r_damage + m_r_hit + m_r_game)

        m_r_goal_pickup = mean(self._r_goal_pickup)
        m_r_goal_enemy = mean(self._r_goal_enemy)

        m_r_search_move = mean(self._r_search_move)
        m_r_search_turn = mean(self._r_search_turn)
        m_r_search_idle = mean(self._r_search_idle)

        search_frac = 0.0
        if self._gm_total > 0:
            search_frac = float(self._gm_counts.get("search", 0)) / float(self._gm_total)

        # records
        self.logger.record("alerts/suspicious_raw_kills_rollout", float(self._susp_raw))
        self.logger.record("alerts/mean_vis_static_steps", float(mean_vis))
        self.logger.record("alerts/mean_stuck_steps", float(mean_stuck))
        self.logger.record("alerts/weapon_select_frac", float(weapon_frac))

        self.logger.record("alerts/mean_r_damage", float(m_r_damage))
        self.logger.record("alerts/mean_r_hit", float(m_r_hit))
        self.logger.record("alerts/mean_r_game", float(m_r_game))
        self.logger.record("alerts/mean_r_move", float(m_r_move))
        self.logger.record("alerts/mean_r_shoot_waste_ammo", float(m_r_shoot_waste))
        self.logger.record("alerts/mean_r_shoot_no_damage", float(m_r_shoot_no_dmg))

        self.logger.record("alerts/mean_r_goal_dist_pickup", float(m_r_goal_pickup))
        self.logger.record("alerts/mean_r_goal_dist_enemy_goal", float(m_r_goal_enemy))

        self.logger.record("alerts/mean_r_search_move", float(m_r_search_move))
        self.logger.record("alerts/mean_r_search_turn", float(m_r_search_turn))
        self.logger.record("alerts/mean_r_search_idle", float(m_r_search_idle))
        self.logger.record("alerts/search_frac_rollout", float(search_frac))

        self.logger.record("alerts/stuck_term_frac_rollout", float(stuck_term_frac))
        self.logger.record("alerts/combat_signal_rollout", float(combat_signal))

        if self.verbose:
            if self._susp_raw >= self.warn_suspicious_raw_per_rollout:
                print(f"[ALERT] suspicious_raw_kills in rollout: {self._susp_raw:.2f} (raw>attrib)")

            if mean_vis >= self.warn_mean_vis_static_steps:
                print(f"[ALERT] mean vis_static_steps high: {mean_vis:.1f} (possible camping/static)")

            if weapon_frac >= self.warn_weapon_select_frac:
                print(f"[ALERT] weapon_select_frac high: {weapon_frac:.2f} (policy collapse risk)")

            if stuck_term_frac >= self.warn_stuck_term_frac and self._n_episodes > 0:
                print(f"[ALERT] stuck_term_frac high: {stuck_term_frac:.2f} ({self._stuck_terms}/{self._n_episodes} episodes)")

            if combat_signal <= self.warn_low_combat_signal and m_r_move > 0.002:
                print(
                    f"[ALERT] low combat signal: (r_damage+r_hit+r_game)={combat_signal:.4f} while move={m_r_move:.4f}. "
                    f"Possible 'just running / not fighting'."
                )

            if search_frac >= self.warn_search_frac:
                print(f"[ALERT] search_frac high: {search_frac:.2f} (agent spending too much time searching / not engaging)")

        # reset
        self._susp_raw = 0.0
        self._vis_static.clear()
        self._stuck_steps.clear()
        self._weapon_sel = 0.0
        self._total = 0.0

        self._r_damage.clear()
        self._r_hit.clear()
        self._r_game.clear()
        self._r_move.clear()
        self._r_shoot_waste.clear()
        self._r_shoot_no_dmg.clear()

        self._r_goal_pickup.clear()
        self._r_goal_enemy.clear()

        self._r_search_move.clear()
        self._r_search_turn.clear()
        self._r_search_idle.clear()

        self._gm_counts.clear()
        self._gm_total = 0

        self._n_episodes = 0
        self._stuck_terms = 0


def evaluate_recurrent_policy(model, env, n_episodes: int, deterministic: bool = True) -> Dict[str, float]:
    """
    Recurrent evaluation for VecEnv with num_envs==1.
    Returns dict of means: reward, monster_kills(attrib), monster_kills_raw, suspicious, damage, hits, deaths.
    """
    rewards: List[float] = []
    mk_attrib: List[float] = []
    mk_raw: List[float] = []
    susp: List[float] = []
    dmg_dealt: List[float] = []
    dmg_taken: List[float] = []
    hits: List[float] = []
    hits_taken: List[float] = []
    deaths: List[float] = []
    frags: List[float] = []
    killcount: List[float] = []

    for _ in range(int(n_episodes)):
        obs = env.reset()
        if isinstance(obs, tuple) and len(obs) == 2:
            obs = obs[0]

        state = None
        episode_start = np.ones((env.num_envs,), dtype=bool)
        done = [False]
        ep_rew = 0.0
        last_info: Dict[str, Any] | None = None

        while not done[0]:
            action, state = model.predict(
                obs, state=state, episode_start=episode_start, deterministic=deterministic
            )
            obs, reward, done, infos = env.step(action)
            episode_start = np.array(done, dtype=bool)
            ep_rew += float(reward[0])
            if infos and len(infos) > 0:
                last_info = infos[0]

        rewards.append(ep_rew)
        if last_info is None:
            continue

        mk_attrib.append(float(last_info.get("ep_monster_kills", 0.0)))
        mk_raw.append(float(last_info.get("ep_monster_kills_raw", 0.0)))
        susp.append(float(last_info.get("ep_suspicious_monster_kills", 0.0)))

        killcount.append(float(last_info.get("ep_killcount", 0.0)))
        frags.append(float(last_info.get("ep_fragcount", 0.0)))

        dmg_dealt.append(float(last_info.get("ep_damage_dealt", 0.0)))
        dmg_taken.append(float(last_info.get("ep_damage_taken", 0.0)))
        hits.append(float(last_info.get("ep_hits", 0.0)))
        hits_taken.append(float(last_info.get("ep_hits_taken", 0.0)))
        deaths.append(float(last_info.get("ep_deaths", 0.0)))

    def mean(x):
        return float(np.mean(x)) if len(x) else 0.0

    return {
        "mean_reward": mean(rewards),

        "mean_monster_kills_attrib": mean(mk_attrib),
        "mean_monster_kills_raw": mean(mk_raw),
        "mean_suspicious_monster_kills": mean(susp),

        "mean_killcount": mean(killcount),
        "mean_fragcount": mean(frags),

        "mean_damage_dealt": mean(dmg_dealt),
        "mean_damage_taken": mean(dmg_taken),
        "mean_hits": mean(hits),
        "mean_hits_taken": mean(hits_taken),
        "mean_deaths": mean(deaths),
    }


class PeriodicEvalSaveCallback(BaseCallback):
    """
    Periodically evaluate and save best model (by mean reward).
    Also logs extra metrics to track "real gameplay" AND reward-hacking risk.
    """

    def __init__(
        self,
        eval_env,
        eval_freq: int,
        n_eval_episodes: int,
        best_path: str,
        deterministic: bool = False,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = int(eval_freq)
        self.n_eval_episodes = int(n_eval_episodes)
        self.best_path = best_path
        self.deterministic = bool(deterministic)
        self.best_mean_reward = -1e9

        os.makedirs(os.path.dirname(best_path), exist_ok=True)

    def _on_step(self) -> bool:
        if self.eval_freq <= 0:
            return True
        if self.num_timesteps % self.eval_freq != 0:
            return True

        m = evaluate_recurrent_policy(
            self.model, self.eval_env, self.n_eval_episodes, deterministic=self.deterministic
        )

        for k, v in m.items():
            self.logger.record(f"eval/{k}", float(v))

        if self.verbose:
            print(
                f"[eval] steps={self.num_timesteps} "
                f"R={m['mean_reward']:.2f} "
                f"mk_attrib={m['mean_monster_kills_attrib']:.2f} mk_raw={m['mean_monster_kills_raw']:.2f} "
                f"susp={m['mean_suspicious_monster_kills']:.2f} "
                f"dmg+={m['mean_damage_dealt']:.1f} dmg-={m['mean_damage_taken']:.1f} "
                f"hit={m['mean_hits']:.1f} hit-={m['mean_hits_taken']:.1f} "
                f"death={m['mean_deaths']:.2f} frag={m['mean_fragcount']:.2f}"
            )

        if m["mean_reward"] > self.best_mean_reward:
            self.best_mean_reward = m["mean_reward"]
            self.model.save(self.best_path)
            if self.verbose:
                print(f"[eval] new best -> saved {self.best_path}")

        return True
