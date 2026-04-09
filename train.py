"""
JADC2 MAPPO Training — Phase 4
================================
Multi-Agent Proximal Policy Optimization with shared weights
and a centralized value function (MAPPO architecture).

Uses Ray RLlib with PPO configured for multi-agent execution.
Agents of the same type share policy weights to accelerate learning.

Usage:
    python train.py [--iters N] [--checkpoint DIR]

Requirements:
    pip install ray[rllib] torch
"""

import argparse
import os
import sys


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

    A centralized global-state observation is provided to the value
    function during training, while actors only see their local view.
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
        # Map each agent to its type-specific shared policy
        if agent_id.startswith("thaad"):
            return "thaad_policy"
        elif agent_id.startswith("aegis"):
            return "aegis_policy"
        elif agent_id.startswith("armor"):
            return "armor_policy"
        elif agent_id.startswith("bomber"):
            return "bomber_policy"
        return "default_policy"

    policies = {}
    for agent_id in probe_env.get_agent_ids():
        policy_id = policy_mapping(agent_id)
        if policy_id not in policies:
            obs_space    = probe_env.observation_space[agent_id]
            act_space    = probe_env.action_space[agent_id]
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
            minibatch_size=256,          # was: sgd_minibatch_size
            num_epochs=10,               # was: num_sgd_iter
            lr=3e-4,
            gamma=0.995,
            lambda_=0.97,
            clip_param=0.2,
            entropy_coeff=0.01,
            vf_loss_coeff=0.5,
            # CNN model for spatial observations
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
        .rollouts(
            num_rollout_workers=num_workers,
            rollout_fragment_length=200,
            num_envs_per_worker=1,
        )
        .resources(num_gpus=num_gpus)
        .framework("torch")
        .debugging(log_level="WARN")
    )

    return config


def train(num_iters: int = 100, checkpoint_dir: str = "checkpoints"):
    _check_dependencies()

    import ray
    from ray import tune

    ray.init(ignore_reinit_error=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    config = build_mappo_config()
    algo  = config.build()

    print()
    print("  JADC2 MAPPO Training")
    print(f"  Iterations   : {num_iters}")
    print(f"  Checkpoint   : {checkpoint_dir}")
    print()

    best_reward = float("-inf")

    for i in range(1, num_iters + 1):
        result = algo.train()

        mean_reward = result.get("episode_reward_mean", 0.0)
        episode_len = result.get("episode_len_mean",    0.0)
        timesteps   = result.get("timesteps_total",     0)

        print(
            f"  Iter {i:04d}/{num_iters:04d} | "
            f"reward: {mean_reward:8.3f} | "
            f"ep_len: {episode_len:6.1f} | "
            f"steps: {timesteps:,}"
        )

        if mean_reward > best_reward:
            best_reward = mean_reward
            path = algo.save(checkpoint_dir)
            print(f"            checkpoint saved: {path}")

    algo.stop()
    ray.shutdown()

    print()
    print(f"  Training complete. Best reward: {best_reward:.3f}")
    print(f"  Checkpoints saved to: {checkpoint_dir}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="JADC2 MAPPO Training")
    parser.add_argument("--iters",      type=int, default=100,          help="Number of training iterations")
    parser.add_argument("--checkpoint", type=str, default="checkpoints", help="Checkpoint output directory")
    args = parser.parse_args()

    train(num_iters=args.iters, checkpoint_dir=args.checkpoint)
