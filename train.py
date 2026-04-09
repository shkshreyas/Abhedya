"""
Project Abhedya — MAPPO Training
==================================
Multi-Agent Proximal Policy Optimization with shared weights
and a centralized value function (MAPPO architecture).

Compatible with Ray RLlib 2.54 (Kaggle GPU environment).
Explicitly uses the LEGACY API stack for stability with PettingZoo
and custom CNN model configs.

Usage:
    python train.py [--iters N] [--checkpoint DIR] [--gpus N] [--workers N] [--time-limit MINS]

Kaggle 2x GPU:
    python train.py --iters 200 --gpus 2 --workers 4
"""

import argparse
import os
import sys
import time


def _check_dependencies():
    missing = []
    try:
        import ray
    except ImportError:
        missing.append("ray[rllib]")
    try:
        import torch
    except ImportError:
        missing.append("torch")
    if missing:
        print()
        print("  Missing dependencies for training:")
        for m in missing:
            print(f"    pip install {m}")
        print()
        sys.exit(1)


def build_mappo_config(num_workers: int = 2, num_gpus: float = 0.0):
    """
    Build a Ray RLlib PPO config for MAPPO-style multi-agent training.

    Policy groupings:
      - thaad_policy  : shared by thaad_0, thaad_1
      - aegis_policy  : shared by aegis_0, aegis_1
      - armor_policy  : shared by armor_0, armor_1
      - bomber_policy : shared by bomber_0

    Uses legacy API stack for compatibility with PettingZoo + custom CNN model.
    """
    from ray.rllib.algorithms.ppo import PPOConfig
    from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
    from ray.tune.registry import register_env
    from jadc2.env import JADC2_Env

    def env_creator(cfg):
        return ParallelPettingZooEnv(JADC2_Env())

    register_env("jadc2", env_creator)

    # Probe the environment for space definitions
    probe_env = ParallelPettingZooEnv(JADC2_Env())

    def policy_mapping(agent_id, *args, **kwargs):
        if agent_id.startswith("thaad"):
            return "thaad_policy"
        elif agent_id.startswith("aegis"):
            return "aegis_policy"
        elif agent_id.startswith("armor"):
            return "armor_policy"
        elif agent_id.startswith("bomber"):
            return "bomber_policy"
        return "default_policy"

    # Build policy specs using dict-style space access (RLlib 2.x)
    policies = {}
    for agent_id in probe_env.possible_agents:           # replaces get_agent_ids()
        policy_id = policy_mapping(agent_id)
        if policy_id not in policies:
            obs_space = probe_env.observation_space[agent_id]
            act_space = probe_env.action_space[agent_id]
            policies[policy_id] = (None, obs_space, act_space, {})

    probe_env.close()

    config = (
        PPOConfig()
        # ── CRITICAL: opt out of new API stack for PettingZoo + custom model ──
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        .environment("jadc2")
        .multi_agent(
            policies=policies,
            policy_mapping_fn=policy_mapping,
        )
        .training(
            train_batch_size=4000,
            minibatch_size=256,          # renamed from sgd_minibatch_size in RLlib 2.40+
            num_epochs=10,               # renamed from num_sgd_iter in RLlib 2.40+
            lr=3e-4,
            gamma=0.995,
            lambda_=0.97,
            clip_param=0.2,
            entropy_coeff=0.01,
            vf_loss_coeff=0.5,
            # CNN model for spatial 64×64×6 observations
            model={
                "conv_filters": [
                    [32, [5, 5], 2],
                    [64, [3, 3], 2],
                    [64, [3, 3], 1],
                ],
                "conv_activation": "relu",
                "fcnet_hiddens": [256, 128],
                "fcnet_activation": "relu",
                "use_lstm": False,
            },
        )
        .env_runners(
            num_env_runners=num_workers,
            rollout_fragment_length=200,
            num_envs_per_env_runner=1,
        )
        .resources(num_gpus=num_gpus)
        .framework("torch")
        .debugging(log_level="WARN")
    )

    return config


def _extract_metric(result, *keys, default=0.0):
    """Safely extract a metric across RLlib versions."""
    for key in keys:
        val = result.get(key)
        if val is not None:
            return val
    return default


def train(num_iters: int = 100, checkpoint_dir: str = "checkpoints",
          num_gpus: float = 0.0, num_workers: int = 2, time_limit_mins: float = 0.0):
    _check_dependencies()

    import ray

    # Tell Ray how many GPUs to reserve at the cluster level
    ray.init(ignore_reinit_error=True,
             num_gpus=int(num_gpus) if num_gpus > 0 else None)

    # RLlib 2.54 requires absolute path for algo.save()
    checkpoint_dir = os.path.abspath(checkpoint_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)

    config = build_mappo_config(num_workers=num_workers, num_gpus=num_gpus)
    algo   = config.build()                              # legacy API stack

    print()
    print("  Project Abhedya — MAPPO Training")
    print(f"  Iterations  : {num_iters}")
    print(f"  GPUs        : {num_gpus}")
    print(f"  Workers     : {num_workers}")
    if time_limit_mins > 0:
        print(f"  Time Limit  : {time_limit_mins} mins")
    print(f"  Checkpoint  : {checkpoint_dir}")
    print()

    best_reward = float("-inf")
    start_time  = time.time()

    for i in range(1, num_iters + 1):
        result = algo.train()

        mean_reward = _extract_metric(result, "episode_reward_mean")
        episode_len = _extract_metric(result, "episode_len_mean")
        timesteps   = _extract_metric(result, "timesteps_total")
        elapsed     = (time.time() - start_time) / 60.0

        print(
            f"  Iter {i:04d}/{num_iters:04d} | "
            f"reward: {mean_reward:8.3f} | "
            f"ep_len: {episode_len:6.1f} | "
            f"steps: {int(timesteps):,} | "
            f"elapsed: {elapsed:.1f}m"
        )

        if mean_reward > best_reward:
            best_reward = mean_reward
            path = algo.save(checkpoint_dir)
            print(f"            ✓ checkpoint saved → {path}")

        if time_limit_mins > 0 and elapsed >= time_limit_mins:
            print(f"\n  Time limit of {time_limit_mins} mins reached — stopping.")
            path = algo.save(checkpoint_dir)
            print(f"  Final checkpoint → {path}")
            break

    algo.stop()
    ray.shutdown()

    print()
    print(f"  Training complete. Best reward: {best_reward:.3f}")
    print(f"  Checkpoints saved to: {checkpoint_dir}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Project Abhedya — MAPPO Training")
    parser.add_argument("--iters",      type=int,   default=100,           help="Training iterations")
    parser.add_argument("--checkpoint", type=str,   default="checkpoints", help="Checkpoint directory")
    parser.add_argument("--gpus",       type=float, default=0.0,           help="Number of GPUs (e.g. 2.0 for Kaggle)")
    parser.add_argument("--workers",    type=int,   default=2,             help="Number of rollout workers")
    parser.add_argument("--time-limit", type=float, default=0.0,           help="Stop after N minutes")
    args = parser.parse_args()

    train(
        num_iters       = args.iters,
        checkpoint_dir  = args.checkpoint,
        num_gpus        = args.gpus,
        num_workers     = args.workers,
        time_limit_mins = args.time_limit,
    )
