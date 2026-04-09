"""
Project Abhedya — MAPPO Training
==================================
Multi-Agent Proximal Policy Optimization with shared weights
and a centralized value function (MAPPO architecture).

Compatible with Ray RLlib >= 2.40 (tested on Kaggle's Ray 2.54).
Agents of the same type share policy weights to accelerate learning.

Usage:
    python train.py [--iters N] [--checkpoint DIR] [--gpus N] [--workers N]

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

    Compatible with RLlib 2.54 (Kaggle default).
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

    # Build policy specs — use dict-style access (RLlib 2.x wraps spaces as dicts)
    policies = {}
    for agent_id in probe_env.get_agent_ids():
        policy_id = policy_mapping(agent_id)
        if policy_id not in policies:
            # observation_space and action_space are dicts in RLlib 2.54
            obs_space = probe_env.observation_space[agent_id]
            act_space = probe_env.action_space[agent_id]
            policies[policy_id] = (None, obs_space, act_space, {})

    probe_env.close()

    config = (
        PPOConfig()
        .environment("jadc2")
        .multi_agent(
            policies=policies,
            policy_mapping_fn=policy_mapping,
        )
        .training(
            # RLlib 2.40+ renamed parameters:
            minibatch_size=256,      # was: sgd_minibatch_size
            num_epochs=10,           # was: num_sgd_iter
            lr=3e-4,
            gamma=0.995,
            lambda_=0.97,
            clip_param=0.2,
            entropy_coeff=0.01,
            vf_loss_coeff=0.5,
            train_batch_size_per_learner=2000,   # per-GPU batch size
            # CNN model for spatial 64x64x6 observations
            model={
                "conv_filters": [
                    [32, [5, 5], 2],
                    [64, [3, 3], 2],
                    [64, [3, 3], 1],
                ],
                "conv_activation": "relu",
                "post_fcnet_hiddens": [256, 128],
                "post_fcnet_activation": "relu",
                "use_lstm": False,
            },
        )
        # RLlib 2.40+ renamed .rollouts() → .env_runners()
        .env_runners(
            num_env_runners=num_workers,
            rollout_fragment_length=200,
            num_envs_per_env_runner=1,
        )
        .learners(
            num_learners=max(1, int(num_gpus)),   # 1 learner per GPU
            num_gpus_per_learner=1 if num_gpus > 0 else 0,
        )
        .framework("torch")
        .debugging(log_level="WARN")
    )

    return config


def _extract_metric(result, *keys, default=0.0):
    """Safely extract a metric that may be nested differently across RLlib versions."""
    for key in keys:
        if key in result:
            return result[key]
    # RLlib 2.x nests metrics inside 'env_runners'
    env_runners = result.get("env_runners", {})
    for key in keys:
        if key in env_runners:
            return env_runners[key]
    return default


def train(num_iters: int = 100, checkpoint_dir: str = "checkpoints",
          num_gpus: float = 0.0, num_workers: int = 2, time_limit_mins: float = 0.0):
    _check_dependencies()

    import ray

    # Tell Ray how many GPUs to reserve — critical for multi-GPU training!
    ray.init(ignore_reinit_error=True,
             num_gpus=int(num_gpus) if num_gpus > 0 else None)
    os.makedirs(checkpoint_dir, exist_ok=True)

    config = build_mappo_config(num_workers=num_workers, num_gpus=num_gpus)
    algo  = config.build()

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

        # Compatible metric extraction for RLlib 2.x
        mean_reward = _extract_metric(
            result,
            "episode_reward_mean",
            "env_runner_results/episode_return_mean",
        )
        episode_len = _extract_metric(
            result,
            "episode_len_mean",
            "env_runner_results/episode_len_mean",
        )
        timesteps = _extract_metric(
            result,
            "timesteps_total",
            "num_env_steps_sampled_lifetime",
        )

        elapsed = (time.time() - start_time) / 60.0
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
    parser.add_argument("--iters",      type=int,   default=100,          help="Training iterations")
    parser.add_argument("--checkpoint", type=str,   default="checkpoints", help="Checkpoint directory")
    parser.add_argument("--gpus",       type=float, default=0.0,          help="Number of GPUs (e.g. 2.0 for Kaggle)")
    parser.add_argument("--workers",    type=int,   default=2,            help="Number of rollout workers")
    parser.add_argument("--time-limit", type=float, default=0.0,          help="Stop training after N minutes")
    args = parser.parse_args()

    train(
        num_iters       = args.iters,
        checkpoint_dir  = args.checkpoint,
        num_gpus        = args.gpus,
        num_workers     = args.workers,
        time_limit_mins = args.time_limit,
    )
