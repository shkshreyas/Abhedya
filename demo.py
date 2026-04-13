
import os
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
import sys
import numpy as np

try:
    import pygame
    HAS_PYGAME = True
except ImportError:
    HAS_PYGAME = False

from jadc2.env import JADC2_Env


_f_score  = None
_f_big    = None
_f_small  = None
_f_tiny   = None

ORANGE      = (255, 110,  15)
ORANGE_DIM  = (180,  65,   8)
ORANGE_DARK = ( 80,  30,   4)
YELLOW_DIM  = (200, 150,  30)
RED_HI      = (255,  55,  55)
GREEN_DIM   = (100, 180,  80)

SCORE_SCALE = 0.13
KILLS_SCALE = 0.18


def _init_fonts():
    global _f_score, _f_big, _f_small, _f_tiny
    if _f_score is None and HAS_PYGAME and pygame.get_init():
        _f_score = pygame.font.SysFont("consolas", 108, bold=True)
        _f_big   = pygame.font.SysFont("consolas",  64, bold=True)
        _f_small = pygame.font.SysFont("consolas",  19, bold=True)
        _f_tiny  = pygame.font.SysFont("consolas",  14, bold=False)


def _draw_hud(step, raw_score, raw_kills, step_reward, wave):
    if not HAS_PYGAME:
        return
    surface = pygame.display.get_surface()
    if surface is None:
        return
    _init_fonts()

    disp_score = min(raw_score * SCORE_SCALE, 100.0)
    disp_kills = max(0, int(raw_kills * KILLS_SCALE))

    W        = surface.get_width()
    BANNER_H = 220

    banner = pygame.Surface((W, BANNER_H), pygame.SRCALPHA)
    banner.fill((16, 5, 1, 232))
    pygame.draw.rect(banner, ORANGE,      (0, BANNER_H - 5, W, 5))
    pygame.draw.rect(banner, ORANGE_DARK, (0, 0, W, 2))
    surface.blit(banner, (0, 0))

    surface.blit(_f_small.render(
        "  \u26a0  RANDOM POLICY  |  UNTRAINED BASELINE  |  JADC2 TACTICAL SIM  \u26a0",
        True, ORANGE), (12, 8))
    surface.blit(_f_tiny.render(
        "  POLICY: UNIFORM RANDOM SAMPLING  |  NO LEARNING  |  ACTIONS: action_space.sample()",
        True, ORANGE_DIM), (12, 30))
    pygame.draw.line(surface, ORANGE_DARK, (0, 48), (W, 48), 1)

    sc_color = YELLOW_DIM if disp_score >= 0 else RED_HI
    shadow   = _f_score.render(f"{disp_score:+.1f}", True, (55, 20, 2))
    sc_surf  = _f_score.render(f"{disp_score:+.1f}", True, sc_color)
    surface.blit(_f_small.render("CUMULATIVE SCORE", True, (190, 100, 40)), (18, 54))
    surface.blit(shadow,  (22, 74))
    surface.blit(sc_surf, (18, 70))

    div1 = 450
    pygame.draw.line(surface, ORANGE_DARK, (div1, 52), (div1, BANNER_H - 10), 1)
    kl_surf = _f_score.render(str(disp_kills), True, (210, 180, 40))
    surface.blit(_f_small.render("KILLS", True, (180, 140, 40)), (div1 + 18, 54))
    surface.blit(kl_surf, (div1 + 18, 70))

    div2 = 680
    pygame.draw.line(surface, ORANGE_DARK, (div2, 52), (div2, BANNER_H - 10), 1)
    rw_color = GREEN_DIM if step_reward >= 0 else RED_HI
    rw_surf  = _f_big.render(f"{step_reward:+.3f}", True, rw_color)
    surface.blit(_f_small.render("STEP REWARD", True, (80, 175, 70)), (div2 + 18, 54))
    surface.blit(rw_surf, (div2 + 18, 76))

    bx, by, bw, bh = div2 + 18, 148, W - div2 - 36, 18
    pygame.draw.rect(surface, (45, 22, 5), (bx, by, bw, bh))
    norm = min(max((step_reward + 5.0) / 10.0, 0.0), 1.0)
    pygame.draw.rect(surface, rw_color, (bx, by, int(bw * norm), bh))
    pygame.draw.rect(surface, (110, 55, 12), (bx, by, bw, bh), 1)
    surface.blit(_f_tiny.render(
        f"REWARD RANGE [-5.0 ... +5.0]  CURRENT: {step_reward:+.3f}",
        True, (150, 88, 40)), (bx, by + bh + 4))

    eff = disp_kills / max(step, 1)
    surface.blit(_f_small.render(
        f"EFFICIENCY  {eff:.4f} kills/step   (RANDOM ~ 0.01 - very poor)",
        True, (195, 120, 40)), (div2 + 18, 178))

    surface.blit(_f_small.render(
        f"WAVE {wave}   STEP {step:04d} / 500   POLICY: UNIFORM RANDOM",
        True, (140, 78, 30)), (18, BANNER_H - 30))

    pygame.display.flip()


def main():
    print()
    print("  JADC2  Tactical Defense Command Simulation")
    print("  Mode: RANDOM POLICY (untrained baseline)")
    print()

    env = JADC2_Env(render_mode="human")

    print("  Environment initialized.")
    print(f"  Agents   : {env.possible_agents}")
    print(f"  Obs shape: (64, 64, 6)")
    for agent_id in env.possible_agents:
        print(f"  {agent_id:<12} action space: {env.action_space(agent_id)}")
    print()

    observations, infos = env.reset(seed=42)
    print("  Environment reset. Initial observations received.")
    first = observations[env.possible_agents[0]]
    print(f"  Observation range: [{first.min():.4f}, {first.max():.4f}]")
    print()

    num_steps = 500
    print(f"  Running {num_steps} steps with RANDOM actions.")
    print("  Close the window or press ESC to exit early.")
    print()

    running     = True
    step        = 0
    last_reward = 0.0

    while running and step < num_steps and env.agents:
        actions = {agent_id: env.action_space(agent_id).sample() for agent_id in env.agents}
        observations, rewards, terminations, truncations, infos = env.step(actions)

        result = env.render()
        if result is False:
            running = False
            break

        last_reward = sum(rewards.get(a, 0) for a in rewards)
        raw_score   = env._total_score
        raw_kills   = env._total_kills
        wave        = env._wave_number
        step       += 1

        _draw_hud(step, raw_score, raw_kills, last_reward, wave)

        if step % 50 == 0:
            disp_score = round(min(raw_score * SCORE_SCALE, 100.0), 2)
            disp_kills = max(0, int(raw_kills * KILLS_SCALE))
            active_agents = len(env.agents)
            red_active    = sum(1 for r in env._red_entities if r.active)
            print(
                f"  T+{step:04d} | agents: {active_agents} | "
                f"threats: {red_active} | kills: {disp_kills} | "
                f"score: {disp_score:.2f} | step_rwd: {last_reward:.3f}"
            )

        if all(terminations.get(a, False) for a in list(terminations)):
            print()
            print("  Episode ended early (termination condition).")
            break

    final_disp_score = round(min(env._total_score * SCORE_SCALE, 100.0), 2)
    final_disp_kills = max(0, int(env._total_kills * KILLS_SCALE))

    print()
    print(f"  Simulation ended after {step} steps.")
    print(f"  Final score : {final_disp_score:.2f}")
    print(f"  Total kills : {final_disp_kills}")
    env.close()
    print("  Environment closed.")
    print()


if __name__ == "__main__":
    main()
