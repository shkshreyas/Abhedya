
import sys
import numpy as np

from jadc2.env import JADC2_Env


def main():
    print()
    print("  JADC2  Tactical Defense Command Simulation")
    print("  Phase 1-3 Demo: Environment + Combat + Red AI Test")
    print()

    env = JADC2_Env(render_mode="human")

    print("  Environment initialized.")
    print(f"  Agents   : {env.possible_agents}")
    print(f"  Obs shape: ({64}, {64}, {6})")
    for agent_id in env.possible_agents:
        print(f"  {agent_id:<12} action space: {env.action_space(agent_id)}")
    print()

    observations, infos = env.reset(seed=42)
    print("  Environment reset. Initial observations received.")
    first = observations[env.possible_agents[0]]
    print(f"  Observation range: [{first.min():.4f}, {first.max():.4f}]")
    print()

    num_steps = 500
    print(f"  Running {num_steps} steps with random actions.")
    print("  Close the window or press ESC to exit early.")
    print()

    running = True
    step = 0

    while running and step < num_steps and env.agents:
        actions = {agent_id: env.action_space(agent_id).sample() for agent_id in env.agents}
        observations, rewards, terminations, truncations, infos = env.step(actions)

        result = env.render()
        if result is False:
            running = False
            break

        step += 1
        if step % 50 == 0:
            total_reward = sum(rewards.get(a, 0) for a in rewards)
            active_agents = len(env.agents)
            red_active = sum(1 for r in env._red_entities if r.active)
            kills = env._total_kills
            print(
                f"  T+{step:04d} | agents: {active_agents} | "
                f"threats: {red_active} | kills: {kills} | "
                f"team reward: {total_reward:.3f}"
            )

        if all(terminations.get(a, False) for a in list(terminations)):
            print()
            print("  Episode ended early (termination condition).")
            break

    print()
    print(f"  Simulation ended after {step} steps.")
    print(f"  Total kills: {env._total_kills}")
    env.close()
    print("  Environment closed.")
    print()


if __name__ == "__main__":
    main()
