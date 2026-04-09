"""
Project Abhedya — MAPPO Training
==================================
Multi-Agent Proximal Policy Optimization with shared weights
and a centralized value function (MAPPO architecture).

Tested against Ray RLlib 2.54 on Kaggle 2x T4 GPU.

CNN math for 64x64 obs (no padding, output = floor((in-k)/s)+1):
  [32, [8,8], 4]  → 15x15
  [64, [4,4], 2]  → 6x6
  [128, [6,6], 1] → 1x1  ✓ (required by RLlib VisionNet)

Usage:
    python train.py --iters 200 --gpus 2 --workers 4
"""

import argparse
import os
import sys
import time
import warnings

# Suppress deprecation noise from RLlib/Ray
warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ.setdefault("PYTHONWARNINGS", "ignore::DeprecationWarning")


def _check_dependencies():
    missing = []
    try:
        import ray  # noqa
    except ImportError:
        missing.append("ray[rllib]")
    try:
        import torch  # noqa
    except ImportError:
        missing.append("torch")
    if missing:
        print("\n  Missing dependencies:")
        for m in missing:
            print(f"    pip install {m}")
        print()
        sys.exit(1)


def build_mappo_config(num_workers: int = 2, num_gpus: float = 0.0):
    """
    Build RLlib PPO config for MAPPO multi-agent training.

    - Legacy API stack (enable_rl_module_and_learner=False)
      required for PettingZoo wrapper + custom CNN model dict.
    - CNN filters computed to reduce 64x64 → 1x1 as VisionNet requires.
    - observation_space/action_space accessed as dicts (RLlib 2.x wrapper).
    - possible_agents used instead of deprecated get_agent_ids().
    - Absolute checkpoint path required by RLlib 2.54 pyarrow backend.
    """
    from ray.rllib.algorithms.ppo import PPOConfig
    from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
    from ray.tune.registry import register_env
    from jadc2.env import JADC2_Env

    def env_creator(cfg):
        return ParallelPettingZooEnv(JADC2_Env())

    register_env("jadc2", env_creator)

    probe_env = ParallelPettingZooEnv(JADC2_Env())

    def policy_mapping(agent_id, *args, **kwargs):
        if agent_id.startswith("thaad"):   return "thaad_policy"
        if agent_id.startswith("aegis"):   return "aegis_policy"
        if agent_id.startswith("armor"):   return "armor_policy"
        if agent_id.startswith("bomber"):  return "bomber_policy"
        return "default_policy"

    policies = {}
    for agent_id in probe_env.possible_agents:
        pid = policy_mapping(agent_id)
        if pid not in policies:
            # RLlib 2.x: observation_space and action_space are dicts
            obs_space = probe_env.observation_space[agent_id]
            act_space = probe_env.action_space[agent_id]
            policies[pid] = (None, obs_space, act_space, {})

    probe_env.close()

    config = (
        PPOConfig()
        # ── Legacy API stack: required for PettingZoo + model dict ──
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
            minibatch_size=256,      # RLlib 2.40+: was sgd_minibatch_size
            num_epochs=10,           # RLlib 2.40+: was num_sgd_iter
            lr=3e-4,
            gamma=0.995,
            lambda_=0.97,
            clip_param=0.2,
            entropy_coeff=0.01,
            vf_loss_coeff=0.5,
            model={
                # CNN filters MUST reduce spatial dims to 1x1 for VisionNet.
                # 64x64 → 15x15 → 6x6 → 1x1  (verified mathematically)
                # 64x64 → 16x16 → 8x8 → 1x1
                "conv_filters": [
                    [32,  [8, 8], 4],
                    [64,  [4, 4], 2],
                    [128, [8, 8], 1],   # 8x8 kernel natively reduces 8x8 map to 1x1
                ],
                "conv_activation": "relu",
                "post_fcnet_hiddens": [256, 128],
                "post_fcnet_activation": "relu",
                "use_lstm": False,
            },
        )
        # RLlib 2.54: .rollouts() is hard-deprecated, must use .env_runners()
        .env_runners(
            num_env_runners=num_workers,
            rollout_fragment_length=200,
            num_envs_per_env_runner=1,
        )
        .resources(num_gpus=num_gpus)
        .framework("torch")
        .debugging(log_level="ERROR")   # suppress all but real errors
    )

    return config


def _get(result, *keys, default=0.0):
    """Safely read a metric from RLlib result dict."""
    for k in keys:
        v = result.get(k)
        if v is not None:
            return float(v)
    return default


def train(
    num_iters: int = 100,
    checkpoint_dir: str = "checkpoints",
    num_gpus: float = 0.0,
    num_workers: int = 2,
    time_limit_mins: float = 0.0,
):
    _check_dependencies()
    import ray

    ray.init(
        ignore_reinit_error=True,
        num_gpus=int(num_gpus) if num_gpus > 0 else None,
        logging_level="ERROR",       # suppress Ray INFO spam
    )

    # RLlib 2.54 pyarrow backend requires absolute path for algo.save()
    checkpoint_dir = os.path.abspath(checkpoint_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)

    config = build_mappo_config(num_workers=num_workers, num_gpus=num_gpus)
    algo   = config.build()

    print()
    print("  +--------------------------------------+")
    print("  |  Project Abhedya — MAPPO Training    |")
    print("  +--------------------------------------+")
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

        mean_reward = _get(result, "episode_reward_mean")
        episode_len = _get(result, "episode_len_mean")
        timesteps   = _get(result, "timesteps_total")
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
            print(f"            ✓ checkpoint → {path}")

        if time_limit_mins > 0 and elapsed >= time_limit_mins:
            print(f"\n  Time limit reached — saving final checkpoint.")
            algo.save(checkpoint_dir)
            break

    algo.stop()
    ray.shutdown()

    print()
    print(f"  ✓ Training complete. Best reward: {best_reward:.3f}")
    print(f"  ✓ Checkpoints at: {checkpoint_dir}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Project Abhedya — MAPPO Training")
    parser.add_argument("--iters",      type=int,   default=100,           help="Training iterations")
    parser.add_argument("--checkpoint", type=str,   default="checkpoints", help="Checkpoint directory")
    parser.add_argument("--gpus",       type=float, default=0.0,           help="GPUs to use (e.g. 2.0)")
    parser.add_argument("--workers",    type=int,   default=2,             help="Rollout workers")
    parser.add_argument("--time-limit", type=float, default=0.0,           help="Stop after N minutes")
    args = parser.parse_args()

    train(
        num_iters       = args.iters,
        checkpoint_dir  = args.checkpoint,
        num_gpus        = args.gpus,
        num_workers     = args.workers,
        time_limit_mins = args.time_limit,
    )
