"""
JADC2 Evaluation — Phase 5
============================
Compares two agent policies over N episodes:

  1. ScriptedBaseline — deterministic fire-at-nearest heuristic
  2. TrainedAgent     — loads MAPPO weights from a checkpoint (optional)

Metrics reported per policy:
  - Mean episode reward
  - Mean kills
  - Radar survival rate (fraction of episodes radar survives)
  - Mean episode length
  - Cost efficiency (kills per unit ammo cost)

Usage:
    python evaluate.py                            # scripted baseline only
    python evaluate.py --checkpoint checkpoints  # compare vs trained model
    python evaluate.py --episodes 50 --render    # render episodes visually
"""

import argparse
import time
from collections import defaultdict
from typing import Dict, Optional, List

import numpy as np

from jadc2.env import JADC2_Env
from jadc2.config import (
    THAAD_NUM_ACTIONS, AEGIS_MOVE_ACTIONS, AEGIS_COMBAT_ACTIONS,
    ARMOR_MOVE_ACTIONS, ARMOR_COMBAT_ACTIONS,
    BOMBER_MOVE_ACTIONS, BOMBER_COMBAT_ACTIONS,
    COST_THAAD_EXPENSIVE, COST_THAAD_CHEAP, COST_AEGIS_SM3,
    COST_ARMOR_AIRBURST, COST_BOMBER_BOMB,
)


# ── Scripted Baseline ─────────────────────────────────────────────────

class ScriptedBaseline:
    """
    Deterministic fire-at-nearest heuristic.

    Rules:
      - THAAD: always fires cheap interceptor at nearest drone; expensive at nearest missile
      - Aegis: moves toward nearest threat, fires SM-3 at target index 0
      - Armor: moves toward nearest drone cluster, fires airburst when threats are close
      - Bomber: flies north toward threats, drops bomb when near a cluster
    """

    def compute_actions(self, env: JADC2_Env) -> Dict[str, any]:
        actions = {}
        for agent_id in env.agents:
            agent_type = env._agent_types[agent_id]
            entity = env._blue_entities.get(agent_id)
            if entity is None or not entity.active:
                actions[agent_id] = env.action_space(agent_id).sample()
                continue

            active_red = [r for r in env._red_entities if r.active]
            drones   = [r for r in active_red if r.entity_type == "drone"]
            missiles = [r for r in active_red if r.entity_type == "missile"]

            if agent_type == "thaad":
                if missiles and entity.can_fire() and entity.ammo_expensive > 0:
                    actions[agent_id] = 1  # fire expensive at missile
                elif drones and entity.can_fire() and entity.ammo_cheap > 0:
                    actions[agent_id] = 2  # fire cheap at drone
                else:
                    actions[agent_id] = 0  # noop

            elif agent_type == "aegis":
                nearest = self._nearest(entity, active_red)
                move = self._move_toward(entity, nearest) if nearest else 0
                combat = 1 if (nearest and entity.ammo_sm3 > 0 and entity.can_fire()) else 0
                actions[agent_id] = np.array([move, combat])

            elif agent_type == "armor":
                nearest_drone = self._nearest(entity, drones)
                move = self._move_toward(entity, nearest_drone) if nearest_drone else 0
                in_range = nearest_drone and entity.distance_to(nearest_drone) <= entity.engagement_range
                combat = 1 if (in_range and entity.ammo_airburst > 0 and entity.can_fire()) else 0
                actions[agent_id] = np.array([move, combat])

            elif agent_type == "bomber":
                all_threats = active_red
                nearest = self._nearest(entity, all_threats)
                move = self._move_toward_8dir(entity, nearest) if nearest else 1  # default: fly north
                in_range = nearest and entity.distance_to(nearest) <= entity.engagement_range
                combat = 1 if (in_range and entity.ammo_bombs > 0 and entity.can_fire()) else 0
                actions[agent_id] = np.array([move, combat])

        return actions

    @staticmethod
    def _nearest(entity, candidates):
        if not candidates:
            return None
        return min(candidates, key=lambda r: entity.distance_to(r))

    @staticmethod
    def _move_toward(entity, target) -> int:
        if target is None:
            return 0
        dx = target.x - entity.x
        dy = target.y - entity.y
        if abs(dx) >= abs(dy):
            return 3 if dx > 0 else 4  # E or W
        else:
            return 1 if dy < 0 else 2  # N or S

    @staticmethod
    def _move_toward_8dir(entity, target) -> int:
        if target is None:
            return 1  # north
        dx = target.x - entity.x
        dy = target.y - entity.y
        if dx > 0 and dy < 0:  return 5  # NE
        if dx < 0 and dy < 0:  return 6  # NW
        if dx > 0 and dy > 0:  return 7  # SE
        if dx < 0 and dy > 0:  return 8  # SW
        if abs(dx) > abs(dy):
            return 3 if dx > 0 else 4
        return 1 if dy < 0 else 2


# ── Trained Agent Wrapper ─────────────────────────────────────────────

class TrainedAgent:
    """Loads a trained RLlib checkpoint and uses it for inference."""

    def __init__(self, checkpoint_path: str):
        try:
            import ray
            from ray.rllib.algorithms.ppo import PPO
            ray.init(ignore_reinit_error=True)
            self.algo = PPO.from_checkpoint(checkpoint_path)
            self.available = True
            print(f"  Loaded checkpoint: {checkpoint_path}")
        except Exception as e:
            print(f"  Could not load checkpoint: {e}")
            print("  Falling back to scripted baseline.")
            self.available = False
            self._fallback = ScriptedBaseline()

    def compute_actions(self, env: JADC2_Env) -> Dict[str, any]:
        if not self.available:
            return self._fallback.compute_actions(env)

        actions = {}
        for agent_id in env.agents:
            obs = env._build_observation(agent_id)
            policy_id = self._agent_to_policy(agent_id)
            action = self.algo.compute_single_action(obs, policy_id=policy_id)
            actions[agent_id] = action
        return actions

    @staticmethod
    def _agent_to_policy(agent_id: str) -> str:
        if agent_id.startswith("thaad"):  return "thaad_policy"
        if agent_id.startswith("aegis"):  return "aegis_policy"
        if agent_id.startswith("armor"):  return "armor_policy"
        if agent_id.startswith("bomber"): return "bomber_policy"
        return "default_policy"


# ── Evaluation Runner ─────────────────────────────────────────────────

def run_episodes(
    policy,
    num_episodes: int,
    render: bool = False,
    label: str = "Policy",
) -> Dict[str, float]:
    """
    Run N evaluation episodes and return aggregate statistics.
    """
    stats = defaultdict(list)

    for ep in range(num_episodes):
        env = JADC2_Env(render_mode="human" if render else None)
        obs, _ = env.reset(seed=ep * 13 + 7)

        episode_reward = 0.0
        done = False

        while not done and env.agents:
            actions = policy.compute_actions(env)
            obs, rewards, terminations, truncations, infos = env.step(actions)

            if render:
                result = env.render()
                if result is False:
                    break

            episode_reward += sum(rewards.values())
            done = all(terminations.get(a, False) or truncations.get(a, False)
                       for a in env.possible_agents)

        stats["reward"].append(episode_reward)
        stats["kills"].append(env._total_kills)
        stats["ep_len"].append(env.current_step)
        stats["radar_survived"].append(1 if env._radar and env._radar.operational else 0)

        # Rough cost efficiency: kills / total ammo spent (estimated from remaining ammo)
        ammo_remaining = 0
        ammo_max = 0
        for ent in env._blue_entities.values():
            if ent.entity_type == "thaad":
                ammo_remaining += ent.ammo_expensive + ent.ammo_cheap
                ammo_max       += 6 + 20
            elif ent.entity_type == "aegis":
                ammo_remaining += ent.ammo_sm3
                ammo_max       += 8
            elif ent.entity_type == "armor":
                ammo_remaining += ent.ammo_airburst
                ammo_max       += 30
            elif ent.entity_type == "bomber":
                ammo_remaining += ent.ammo_bombs
                ammo_max       += 4

        ammo_spent = ammo_max - ammo_remaining
        efficiency = env._total_kills / max(1, ammo_spent)
        stats["efficiency"].append(efficiency)

        env.close()

        print(
            f"  {label}  ep {ep + 1:3d}/{num_episodes} | "
            f"reward: {episode_reward:8.2f} | "
            f"kills: {env._total_kills:3d} | "
            f"radar: {'OK' if stats['radar_survived'][-1] else 'LOST'} | "
            f"steps: {env.current_step}"
        )

    return {k: float(np.mean(v)) for k, v in stats.items()}


def print_comparison(results: Dict[str, Dict[str, float]]):
    """Pretty-print a comparison table of all evaluated policies."""
    print()
    print("  Evaluation Results")
    print()

    headers = ["Policy", "Reward", "Kills", "Ep Len", "Radar%", "Efficiency"]
    row_fmt = "  {:<22} {:>10} {:>8} {:>8} {:>8} {:>12}"
    print(row_fmt.format(*headers))
    print("  " + "-" * 72)

    for policy_name, stats in results.items():
        print(row_fmt.format(
            policy_name,
            f"{stats['reward']:+.2f}",
            f"{stats['kills']:.1f}",
            f"{stats['ep_len']:.0f}",
            f"{stats['radar_survived'] * 100:.0f}%",
            f"{stats['efficiency']:.3f} k/rnd",
        ))

    print()


def main():
    parser = argparse.ArgumentParser(description="JADC2 Policy Evaluation")
    parser.add_argument("--episodes",   type=int,  default=10,   help="Episodes per policy")
    parser.add_argument("--checkpoint", type=str,  default=None, help="Path to trained checkpoint")
    parser.add_argument("--render",     action="store_true",      help="Render evaluation episodes")
    args = parser.parse_args()

    print()
    print("  JADC2 Phase 5 — Policy Evaluation")
    print(f"  Episodes : {args.episodes}")
    print(f"  Render   : {args.render}")
    print()

    results = {}

    # Always evaluate the scripted baseline
    print("  Running scripted baseline...")
    print()
    baseline = ScriptedBaseline()
    results["Scripted Baseline"] = run_episodes(
        baseline,
        args.episodes,
        render=args.render,
        label="Scripted",
    )

    # Evaluate trained agent if checkpoint provided
    if args.checkpoint:
        print()
        print(f"  Loading trained agent from {args.checkpoint}...")
        print()
        trained = TrainedAgent(args.checkpoint)
        results["Trained MAPPO"] = run_episodes(
            trained,
            args.episodes,
            render=args.render,
            label="Trained",
        )

    print_comparison(results)

    # Winner analysis
    if len(results) > 1:
        names  = list(results.keys())
        kills  = {n: results[n]["kills"]   for n in names}
        radars = {n: results[n]["radar_survived"] for n in names}
        best_kills = max(names, key=lambda n: kills[n])
        best_radar = max(names, key=lambda n: radars[n])
        print(f"  Highest kills      : {best_kills}  ({kills[best_kills]:.1f} avg)")
        print(f"  Best radar survival: {best_radar}  ({radars[best_radar] * 100:.0f}%)")
        print()


if __name__ == "__main__":
    main()
