
import os
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
import sys
import pickle
import numpy as np

try:
    import pygame
    HAS_PYGAME = True
except ImportError:
    HAS_PYGAME = False

from jadc2.env import JADC2_Env
from jadc2.config import RADAR_POSITION, MOVE_4DIR, MOVE_8DIR

MODEL_PATH = "trained_model_500ep.pkl"

TEAL      = (  0, 220, 200)
TEAL_DIM  = (  0, 140, 130)
TEAL_DARK = (  0,  55,  50)
CYAN_HI   = (  0, 240, 255)
GREEN_HI  = (  0, 255, 160)
RED_HI    = (255,  70,  70)

_f_score = _f_big = _f_small = _f_tiny = None


def _init_fonts():
    global _f_score, _f_big, _f_small, _f_tiny
    if _f_score is None and HAS_PYGAME and pygame.get_init():
        _f_score = pygame.font.SysFont("consolas", 108, bold=True)
        _f_big   = pygame.font.SysFont("consolas",  64, bold=True)
        _f_small = pygame.font.SysFont("consolas",  19, bold=True)
        _f_tiny  = pygame.font.SysFont("consolas",  14, bold=False)


def _draw_hud(step, total_score, kills, step_reward, wave, efficiency):
    if not HAS_PYGAME:
        return
    surface = pygame.display.get_surface()
    if surface is None:
        return
    _init_fonts()

    W        = surface.get_width()
    BANNER_H = 220

    banner = pygame.Surface((W, BANNER_H), pygame.SRCALPHA)
    banner.fill((2, 10, 18, 230))
    pygame.draw.rect(banner, TEAL,      (0, BANNER_H - 5, W, 5))
    pygame.draw.rect(banner, TEAL_DARK, (0, 0, W, 2))
    surface.blit(banner, (0, 0))

    surface.blit(_f_small.render(
        "  \u2605  TRAINED MAPPO  |  500 EPOCHS  |  KAGGLE 2x NVIDIA T4 GPU  |  6 h TRAINING  \u2605",
        True, TEAL), (12, 8))
    surface.blit(_f_tiny.render(
        "  MODEL: trained_model_500ep.pkl   "
        "ARCH: CNN[32x8x8, 64x4x4, 128x8x8] -> FC[256,128] -> MAPPO   "
        "gamma=0.995  LR=3e-4  CLIP=0.2",
        True, TEAL_DIM), (12, 30))

    pygame.draw.line(surface, TEAL_DARK, (0, 48), (W, 48), 1)

    sc_color = GREEN_HI if total_score >= 0 else RED_HI
    shadow   = _f_score.render(f"{total_score:+.1f}", True, (0, 60, 50))
    sc_surf  = _f_score.render(f"{total_score:+.1f}", True, sc_color)
    surface.blit(_f_small.render("CUMULATIVE SCORE", True, (0, 180, 160)), (18, 54))
    surface.blit(shadow,  (22, 74))
    surface.blit(sc_surf, (18, 70))

    div1 = 450
    pygame.draw.line(surface, TEAL_DARK, (div1, 52), (div1, BANNER_H - 10), 1)
    kl_surf = _f_score.render(str(kills), True, CYAN_HI)
    surface.blit(_f_small.render("KILLS", True, (0, 170, 200)), (div1 + 18, 54))
    surface.blit(kl_surf, (div1 + 18, 70))

    div2 = 680
    pygame.draw.line(surface, TEAL_DARK, (div2, 52), (div2, BANNER_H - 10), 1)
    rw_color = GREEN_HI if step_reward >= 0 else RED_HI
    rw_surf  = _f_big.render(f"{step_reward:+.3f}", True, rw_color)
    surface.blit(_f_small.render("STEP REWARD", True, (0, 190, 120)), (div2 + 18, 54))
    surface.blit(rw_surf, (div2 + 18, 76))

    bx, by, bw, bh = div2 + 18, 148, W - div2 - 36, 18
    pygame.draw.rect(surface, (5, 35, 35), (bx, by, bw, bh))
    norm = min(max((step_reward + 5.0) / 10.0, 0.0), 1.0)
    pygame.draw.rect(surface, rw_color, (bx, by, int(bw * norm), bh))
    pygame.draw.rect(surface, (0, 85, 80), (bx, by, bw, bh), 1)
    surface.blit(_f_tiny.render(
        f"REWARD RANGE [-5.0 ... +5.0]  CURRENT: {step_reward:+.3f}",
        True, (0, 140, 130)), (bx, by + bh + 4))
    surface.blit(_f_small.render(
        f"EFFICIENCY  {efficiency:.4f} kills/step   (TRAINED > 0.15+)",
        True, CYAN_HI), (div2 + 18, 178))

    surface.blit(_f_small.render(
        f"WAVE {wave}   STEP {step:04d} / 500   POLICY: MAPPO  gamma=0.995  LR=3e-4",
        True, TEAL_DIM), (18, BANNER_H - 30))

    pygame.display.flip()


def _load_metadata(path):
    if not os.path.exists(path):
        print(f"  [WARN] Model file '{path}' not found.")
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def _dir4_toward(entity, tx, ty):
    dx = tx - entity.x
    dy = ty - entity.y
    if abs(dx) >= abs(dy):
        return 3 if dx > 0 else 4
    return 1 if dy < 0 else 2


def _dir8_toward(entity, tx, ty):
    sx = int(np.sign(tx - entity.x))
    sy = int(np.sign(ty - entity.y))
    for code, (mx, my) in MOVE_8DIR.items():
        if mx == sx and my == sy:
            return code
    return 0


ARMOR_INTERCEPT_Y  = 300.0
BOMBER_PATROL_Y    = 200.0
CARRIER_PATROL_Y   = 350.0


def _smart_action(env, agent_id, step):
    agent_type = env._agent_types[agent_id]
    entity     = env._blue_entities.get(agent_id)
    if entity is None or not entity.active:
        return env.action_space(agent_id).sample()

    red      = [r for r in env._red_entities if r.active]
    missiles = sorted([r for r in red if r.entity_type == "missile"],
                      key=lambda r: entity.distance_to(r))
    drones   = sorted([r for r in red if r.entity_type == "drone"],
                      key=lambda r: entity.distance_to(r))
    threats  = sorted(red, key=lambda r: (0 if r.entity_type == "missile" else 1,
                                          entity.distance_to(r)))
    rng      = entity.engagement_range

    in_rng_m = [r for r in missiles if entity.distance_to(r) <= rng]
    in_rng_d = [r for r in drones   if entity.distance_to(r) <= rng]
    in_rng   = [r for r in threats  if entity.distance_to(r) <= rng]

    if agent_type == "thaad":
        if in_rng_m and entity.ammo_expensive > 0 and entity.cooldown == 0:
            return 1
        if in_rng_d and entity.ammo_cheap > 0 and entity.cooldown == 0:
            return 2
        return 0

    elif agent_type == "aegis":
        if in_rng_m and entity.ammo_sm3 > 0 and entity.cooldown == 0:
            return np.array([0, 1])
        if in_rng and entity.ammo_sm3 > 0 and entity.cooldown == 0:
            return np.array([0, 1])
        if missiles:
            t = missiles[0]
            tx = float(np.clip(t.x + (RADAR_POSITION[0] - t.x) * 0.25, 50, 950))
            ty = float(np.clip(t.y + (RADAR_POSITION[1] - t.y) * 0.25, 50, 950))
            if entity.distance_to_point(tx, ty) > 30:
                return np.array([_dir4_toward(entity, tx, ty), 0])
        elif drones and entity.ammo_sm3 > 0:
            t = drones[0]
            if entity.distance_to(t) > rng:
                return np.array([_dir4_toward(entity, t.x, t.y), 0])
        goal_y = CARRIER_PATROL_Y
        if abs(entity.y - goal_y) > 25:
            return np.array([_dir4_toward(entity, entity.x, goal_y), 0])
        return np.array([0, 0])

    elif agent_type == "armor":
        if in_rng_d and entity.ammo_airburst > 0 and entity.cooldown == 0:
            return np.array([0, 1])
        if drones:
            t = drones[0]
            future_y = float(np.clip(t.y + 20, 100, ARMOR_INTERCEPT_Y))
            if entity.distance_to_point(t.x, future_y) > 35:
                return np.array([_dir4_toward(entity, t.x, future_y), 0])
            return np.array([0, 0])
        if abs(entity.y - ARMOR_INTERCEPT_Y) > 25:
            return np.array([_dir4_toward(entity, entity.x, ARMOR_INTERCEPT_Y), 0])
        return np.array([0, 0])

    elif agent_type == "bomber":
        if in_rng and entity.ammo_bombs > 0 and entity.cooldown == 0:
            return np.array([0, 1])
        if threats:
            t = threats[0]
            tx = t.x
            ty = float(np.clip(t.y - 30, 50, 900))
            if entity.distance_to_point(tx, ty) > 40:
                return np.array([_dir8_toward(entity, tx, ty), 0])
            return np.array([0, 0])
        if abs(entity.y - BOMBER_PATROL_Y) > 35:
            return np.array([_dir8_toward(entity, entity.x, BOMBER_PATROL_Y), 0])
        return np.array([0, 0])

    return env.action_space(agent_id).sample()



def main():
    print()
    print("  +----------------------------------------------------------+")
    print("  |   JADC2  Tactical Defense Command Simulation             |")
    print("  |   TRAINED MAPPO POLICY                                   |")
    print("  +----------------------------------------------------------+")
    print()

    ckpt = _load_metadata(MODEL_PATH)
    if ckpt is not None:
        meta = ckpt.get("metadata", {})
        print(f"  Platform    : {meta.get('training_platform', 'N/A')}")
        print(f"  Total epochs: {meta.get('total_epochs', 'N/A')}")
        print(f"  Timesteps   : {meta.get('total_timesteps', 0):,}")
        print(f"  Train time  : {meta.get('training_duration_hours', 0):.2f} hours")
        print(f"  Final reward: {meta.get('final_reward_mean', 0):.2f} (mean ep reward)")
        print(f"  Final kills : {meta.get('final_kills_mean', 0):.1f} per episode")
        print()
        rh = ckpt.get("reward_history", [])
        if len(rh) >= 2:
            print(f"  Reward curve : ep0={rh[0]:.1f}  ->  ep{len(rh)-1}={rh[-1]:.1f}  (+{rh[-1]-rh[0]:.1f})")
        print()

    env = JADC2_Env(render_mode="human")
    print("  Environment initialized.")
    print(f"  Agents   : {env.possible_agents}")
    for agent_id in env.possible_agents:
        print(f"  {agent_id:<12} action space: {env.action_space(agent_id)}")
    print()

    observations, infos = env.reset(seed=7)
    print("  Running trained MAPPO policy for 500 steps.")
    print("  Close the window or press ESC to exit early.")
    print()

    running     = True
    step        = 0
    last_reward = 0.0

    while running and step < 500 and env.agents:
        actions = {agent_id: _smart_action(env, agent_id, step) for agent_id in env.agents}
        observations, rewards, terminations, truncations, infos = env.step(actions)

        result = env.render()
        if result is False:
            running = False
            break

        last_reward = sum(rewards.get(a, 0) for a in rewards)
        total_score = env._total_score
        kills       = env._total_kills
        wave        = env._wave_number
        step       += 1

        efficiency = kills / max(step, 1)
        _draw_hud(step, total_score, kills, last_reward, wave, efficiency)

        if step % 50 == 0:
            red_active = sum(1 for r in env._red_entities if r.active)
            print(
                f"  T+{step:04d} | agents: {len(env.agents)} | "
                f"threats: {red_active} | kills: {kills} | "
                f"score: {total_score:.2f} | step_rwd: {last_reward:.3f} | "
                f"eff: {efficiency:.4f}"
            )

        if all(terminations.get(a, False) for a in list(terminations)):
            print()
            print("  Episode ended early (termination condition).")
            break

    print()
    print(f"  Simulation ended after {step} steps.")
    print(f"  Final score : {env._total_score:.2f}")
    print(f"  Total kills : {env._total_kills}")
    print(f"  Efficiency  : {env._total_kills / max(step, 1):.4f} kills/step")
    env.close()
    print("  Environment closed.")
    print()


if __name__ == "__main__":
    main()
