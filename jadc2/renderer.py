
import math
import time
import numpy as np

try:
    import pygame
    import pygame.gfxdraw
    HAS_PYGAME = True
except ImportError:
    HAS_PYGAME = False

from jadc2.config import (
    SCREEN_WIDTH, SCREEN_HEIGHT, BATTLEFIELD_SIZE, HUD_WIDTH, FPS,
    WORLD_SIZE, Colors, GRID_DIM,
)


class Particle:
    __slots__ = ("x", "y", "vx", "vy", "color", "lifetime", "max_lifetime", "size")

    def __init__(self, x, y, vx, vy, color, lifetime, size=2):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.color = color
        self.lifetime = lifetime
        self.max_lifetime = lifetime
        self.size = size

    def tick(self) -> bool:
        self.x += self.vx
        self.y += self.vy
        self.vx *= 0.92
        self.vy *= 0.92
        self.lifetime -= 1
        return self.lifetime <= 0

    @property
    def alpha(self) -> int:
        return max(0, int(255 * self.lifetime / max(1, self.max_lifetime)))


class MilitaryRadarRenderer:

    def __init__(self, env):
        if not HAS_PYGAME:
            raise RuntimeError(
                "pygame is required for rendering. Install with: pip install pygame"
            )

        self.env = env
        self.initialized = False
        self.clock = None
        self.screen = None
        self.font_large = None
        self.font_medium = None
        self.font_small = None
        self.font_tiny = None

        self.sweep_angle = 0.0
        self.sweep_speed = 0.022
        self.frame_count = 0
        self.blink_state = True
        self.blink_timer = 0

        self._grid_surface = None
        self._scanline_surface = None
        self._hud_bg_surface = None

        self._particles: list = []

        self.event_log: list = []
        self.max_events = 9

        self.kill_count = 0
        self.score = 0.0

        self._wave_warning_timer = 0
        self._wave_warning_number = 0

        self._threat_history: list = []
        self._threat_history_max = 80

        self._seen_effect_ids: set = set()

    def initialize(self):
        if self.initialized:
            return

        pygame.init()
        pygame.display.set_caption("JADC2  Tactical Defense Command")
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()

        try:
            self.font_large  = pygame.font.SysFont("consolas", 22, bold=True)
            self.font_medium = pygame.font.SysFont("consolas", 16, bold=True)
            self.font_small  = pygame.font.SysFont("consolas", 13)
            self.font_tiny   = pygame.font.SysFont("consolas", 11)
        except Exception:
            self.font_large  = pygame.font.Font(None, 24)
            self.font_medium = pygame.font.Font(None, 18)
            self.font_small  = pygame.font.Font(None, 14)
            self.font_tiny   = pygame.font.Font(None, 12)

        self._build_grid_surface()
        self._build_scanline_surface()
        self._build_hud_background()
        self._build_battlefield_bg()
        self._temp_layer = pygame.Surface((BATTLEFIELD_SIZE, BATTLEFIELD_SIZE), pygame.SRCALPHA)
        self.initialized = True

    def _build_battlefield_bg(self):
        self._bg_surface = pygame.Surface((BATTLEFIELD_SIZE, BATTLEFIELD_SIZE), pygame.SRCALPHA)
        cx, cy = BATTLEFIELD_SIZE // 2, BATTLEFIELD_SIZE // 2
        for i in range(6):
            r = BATTLEFIELD_SIZE // 2 - i * (BATTLEFIELD_SIZE // 12)
            if r <= 0:
                continue
            surf = pygame.Surface((r * 2, r * 2), pygame.SRCALPHA)
            surf.fill((0, 28, 18, 3 + i * 2))
            self._bg_surface.blit(surf, (cx - r, cy - r))

    def _build_grid_surface(self):
        self._grid_surface = pygame.Surface((BATTLEFIELD_SIZE, BATTLEFIELD_SIZE), pygame.SRCALPHA)
        cell_px = BATTLEFIELD_SIZE / GRID_DIM

        for i in range(GRID_DIM + 1):
            pos = int(i * cell_px)
            is_major = (i % 8 == 0)
            alpha = 32 if is_major else 12
            color = (*Colors.GRID_MAJOR[:3], alpha) if is_major else (*Colors.GRID_LINE[:3], alpha)
            pygame.draw.line(self._grid_surface, color, (pos, 0), (pos, BATTLEFIELD_SIZE))
            pygame.draw.line(self._grid_surface, color, (0, pos), (BATTLEFIELD_SIZE, pos))

        for i in range(0, GRID_DIM + 1, 8):
            pos = int(i * cell_px)
            coord_val = int(i * (WORLD_SIZE / GRID_DIM))
            label = self.font_tiny.render(str(coord_val), True, (*Colors.HUD_TEXT[:3], 55))
            self._grid_surface.blit(label, (pos + 2, BATTLEFIELD_SIZE - 14))
            label2 = self.font_tiny.render(str(coord_val), True, (*Colors.HUD_TEXT[:3], 55))
            self._grid_surface.blit(label2, (2, pos + 2))

    def _build_scanline_surface(self):
        self._scanline_surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        for y in range(0, SCREEN_HEIGHT, 3):
            pygame.draw.line(self._scanline_surface, (0, 0, 0, 16), (0, y), (SCREEN_WIDTH, y))

    def _build_hud_background(self):
        self._hud_bg_surface = pygame.Surface((HUD_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        self._hud_bg_surface.fill(Colors.BG_PANEL)

        for i in range(5):
            alpha = 90 - i * 18
            pygame.draw.line(
                self._hud_bg_surface,
                (*Colors.HUD_ACCENT[:3], alpha),
                (i, 0), (i, SCREEN_HEIGHT),
            )

        for sy in [60, 260, 450, 590]:
            for i in range(3):
                pygame.draw.line(
                    self._hud_bg_surface,
                    (*Colors.HUD_BORDER[:3], 100 - i * 28),
                    (8, sy + i), (HUD_WIDTH - 8, sy + i),
                )
            pygame.draw.line(
                self._hud_bg_surface,
                (*Colors.HUD_ACCENT[:3], 18),
                (8, sy - 1), (HUD_WIDTH - 8, sy - 1),
            )


    def world_to_screen(self, wx: float, wy: float) -> tuple:
        sx = int((wx / WORLD_SIZE) * BATTLEFIELD_SIZE)
        sy = int((wy / WORLD_SIZE) * BATTLEFIELD_SIZE)
        return max(0, min(sx, BATTLEFIELD_SIZE - 1)), max(0, min(sy, BATTLEFIELD_SIZE - 1))


    def render(self, env_state: dict):
        if not self.initialized:
            self.initialize()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return False

        self.frame_count += 1
        self.blink_timer += 1
        if self.blink_timer >= 15:
            self.blink_state = not self.blink_state
            self.blink_timer = 0

        if self._wave_warning_timer > 0:
            self._wave_warning_timer -= 1

        red_entities = env_state.get("red_entities", [])
        active_count = sum(1 for r in red_entities if r.active)
        self._threat_history.append(active_count)
        if len(self._threat_history) > self._threat_history_max:
            self._threat_history.pop(0)

        self._process_effects_for_particles(env_state.get("effects", []))

        self._particles = [p for p in self._particles if not p.tick()]

        self.screen.fill(Colors.BG_DEEP)
        self._draw_battlefield_bg()
        self.screen.blit(self._grid_surface, (0, 0))
        self._draw_radar_sweep(env_state)
        self._draw_range_rings(env_state)
        self._draw_trails(env_state)
        self._draw_entities(env_state)
        self._draw_effects(env_state)
        self._draw_particles()
        self._draw_hud_panel(env_state)
        self._draw_top_bar(env_state)
        self._draw_wave_warning()
        self.screen.blit(self._scanline_surface, (0, 0))
        self._draw_vignette()

        pygame.display.flip()
        self.clock.tick(FPS)
        return True


    def _process_effects_for_particles(self, effects):
        for fx in effects:
            if fx.effect_type in ("explosion",) and fx.lifetime == fx.max_lifetime - 1:
                sx, sy = self.world_to_screen(fx.x, fx.y)
                self._spawn_explosion_particles(sx, sy, fx.color, count=18)

    def _spawn_explosion_particles(self, sx, sy, base_color, count=16):
        for _ in range(count):
            angle = random.uniform(0, math.pi * 2) if True else 0
            speed = random.uniform(0.8, 3.5)
            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed
            r = min(255, base_color[0] + random.randint(-30, 30))
            g = min(255, base_color[1] + random.randint(-20, 20))
            b = min(255, base_color[2] + random.randint(-20, 20))
            lifetime = random.randint(8, 18)
            size = random.randint(1, 3)
            self._particles.append(Particle(sx, sy, vx, vy, (r, g, b), lifetime, size))

    def _draw_particles(self):
        for p in self._particles:
            if p.alpha <= 0:
                continue
            surf = pygame.Surface((p.size * 2 + 2, p.size * 2 + 2), pygame.SRCALPHA)
            pygame.draw.circle(surf, (*p.color, p.alpha), (p.size + 1, p.size + 1), p.size)
            self.screen.blit(surf, (int(p.x) - p.size, int(p.y) - p.size))


    def _draw_battlefield_bg(self):
        self.screen.blit(self._bg_surface, (0, 0))

    def _draw_trails(self, state: dict):
        self._temp_layer.fill((0, 0, 0, 0))
        trail_surf = self._temp_layer

        for ent in state.get("red_entities", []):
            if not hasattr(ent, "trail") or not ent.trail:
                continue

            trail = ent.trail
            n = len(trail)
            if ent.entity_type == "drone":
                base_color = Colors.THREAT_RED
            else:
                base_color = Colors.MISSILE_AMBER

            for i in range(1, n):
                alpha = int(80 * (i / n))
                sx1, sy1 = self.world_to_screen(*trail[i - 1])
                sx2, sy2 = self.world_to_screen(*trail[i])
                pygame.draw.line(trail_surf, (*base_color, alpha), (sx1, sy1), (sx2, sy2), 1)

        self.screen.blit(trail_surf, (0, 0))

    def _draw_radar_sweep(self, state: dict):
        radar = state.get("radar")
        if not radar or not radar.operational:
            return

        cx, cy = self.world_to_screen(radar.x, radar.y)
        sweep_radius = int((radar.detection_range / WORLD_SIZE) * BATTLEFIELD_SIZE)
        self.sweep_angle += self.sweep_speed

        num_trail = 45
        self._temp_layer.fill((0, 0, 0, 0))
        trail_surf = self._temp_layer
        for i in range(num_trail):
            trail_angle = self.sweep_angle - i * 0.038
            alpha = max(0, int(55 * (1 - i / num_trail)))
            if alpha <= 0:
                continue
            ex = int(cx + math.cos(trail_angle) * sweep_radius)
            ey = int(cy + math.sin(trail_angle) * sweep_radius)
            pygame.draw.line(trail_surf, (0, 255, 65, alpha), (cx, cy), (ex, ey), 1)
        self.screen.blit(trail_surf, (0, 0))

        end_x = int(cx + math.cos(self.sweep_angle) * sweep_radius)
        end_y = int(cy + math.sin(self.sweep_angle) * sweep_radius)
        self._temp_layer.fill((0, 0, 0, 0))
        glow_surf = self._temp_layer
        pygame.draw.line(glow_surf, (0, 255, 65, 90), (cx, cy), (end_x, end_y), 3)
        self.screen.blit(glow_surf, (0, 0))
        pygame.draw.line(self.screen, Colors.RADAR_GREEN, (cx, cy), (end_x, end_y), 1)

        pygame.draw.circle(self.screen, Colors.RADAR_GREEN, (cx, cy), 3)
        pygame.gfxdraw.aacircle(self.screen, cx, cy, 3, Colors.RADAR_GREEN)

    def _draw_range_rings(self, state: dict):
        self._temp_layer.fill((0, 0, 0, 0))
        ring_surf = self._temp_layer

        radar = state.get("radar")
        if radar and radar.operational:
            cx, cy = self.world_to_screen(radar.x, radar.y)
            r = int((radar.detection_range / WORLD_SIZE) * BATTLEFIELD_SIZE)
            pygame.gfxdraw.aacircle(ring_surf, cx, cy, r, (*Colors.RADAR_DIM, 38))
            for angle in [0, math.pi / 2, math.pi, 3 * math.pi / 2]:
                sx = int(cx + math.cos(angle) * r)
                sy = int(cy + math.sin(angle) * r)
                ix = int(cx + math.cos(angle) * (r - 8))
                iy = int(cy + math.sin(angle) * (r - 8))
                pygame.draw.line(ring_surf, (*Colors.RADAR_DIM, 75), (sx, sy), (ix, iy), 1)

        for ent in state.get("blue_entities", []):
            if ent.active and hasattr(ent, "engagement_range"):
                cx, cy = self.world_to_screen(ent.x, ent.y)
                r = int((ent.engagement_range / WORLD_SIZE) * BATTLEFIELD_SIZE)
                if r > 2:
                    color_map = {
                        "thaad":  (*Colors.THAAD_COLOR, 22),
                        "aegis":  (*Colors.AEGIS_COLOR, 22),
                        "armor":  (*Colors.ARMOR_COLOR, 18),
                        "bomber": (*Colors.BOMBER_COLOR, 14),
                    }
                    rc = color_map.get(ent.entity_type, (*Colors.FRIENDLY_DIM, 18))
                    pygame.gfxdraw.aacircle(ring_surf, cx, cy, r, rc)

        self.screen.blit(ring_surf, (0, 0))

    def _draw_entities(self, state: dict):
        for ent in state.get("blue_entities", []):
            if not ent.active:
                continue
            sx, sy = self.world_to_screen(ent.x, ent.y)
            self._draw_blue_entity(sx, sy, ent)

        for ent in state.get("red_entities", []):
            if not ent.active:
                continue
            sx, sy = self.world_to_screen(ent.x, ent.y)
            self._draw_red_entity(sx, sy, ent)

        radar = state.get("radar")
        if radar and radar.active:
            sx, sy = self.world_to_screen(radar.x, radar.y)
            self._draw_radar_icon(sx, sy, radar)

    def _draw_blue_entity(self, x: int, y: int, entity):
        if entity.entity_type == "thaad":
            color = Colors.THAAD_COLOR
            points = [(x, y - 10), (x + 7, y), (x, y + 10), (x - 7, y)]
            glow_s = pygame.Surface((32, 32), pygame.SRCALPHA)
            pygame.draw.circle(glow_s, (*color, 38), (16, 16), 16)
            self.screen.blit(glow_s, (x - 16, y - 16))
            pygame.gfxdraw.filled_polygon(self.screen, points, color)
            pygame.gfxdraw.aapolygon(self.screen, points, color)
            if hasattr(entity, "radar_active") and entity.radar_active:
                pygame.gfxdraw.aacircle(self.screen, x, y, 14, (0, 255, 65, 100))

        elif entity.entity_type == "aegis":
            color = Colors.AEGIS_COLOR
            points = [(x - 8, y - 7), (x + 8, y - 7), (x, y + 9)]
            glow_s = pygame.Surface((32, 32), pygame.SRCALPHA)
            pygame.draw.circle(glow_s, (*color, 32), (16, 16), 16)
            self.screen.blit(glow_s, (x - 16, y - 16))
            pygame.gfxdraw.filled_polygon(self.screen, points, color)
            pygame.gfxdraw.aapolygon(self.screen, points, color)

        elif entity.entity_type == "armor":
            color = Colors.ARMOR_COLOR
            glow_s = pygame.Surface((32, 32), pygame.SRCALPHA)
            pygame.draw.circle(glow_s, (*color, 28), (16, 16), 16)
            self.screen.blit(glow_s, (x - 16, y - 16))
            rect = pygame.Rect(x - 7, y - 7, 14, 14)
            pygame.draw.rect(self.screen, color, rect)
            bright = (min(255, color[0] + 60), min(255, color[1] + 60), min(255, color[2] + 60))
            pygame.draw.rect(self.screen, bright, rect, 1)

        elif entity.entity_type == "bomber":
            color = Colors.BOMBER_COLOR
            points = [(x, y - 10), (x + 9, y + 7), (x, y + 3), (x - 9, y + 7)]
            glow_s = pygame.Surface((32, 32), pygame.SRCALPHA)
            pygame.draw.circle(glow_s, (*color, 28), (16, 16), 16)
            self.screen.blit(glow_s, (x - 16, y - 16))
            pygame.gfxdraw.filled_polygon(self.screen, points, color)
            pygame.gfxdraw.aapolygon(self.screen, points, color)

        label = self.font_tiny.render(entity.entity_id.upper(), True, Colors.HUD_TEXT)
        self.screen.blit(label, (x + 12, y - 6))

        if hasattr(entity, "max_hp") and entity.max_hp > 0:
            bar_w, bar_h = 16, 2
            hp_frac = entity.hp / entity.max_hp
            bx, by = x - bar_w // 2, y + 14
            pygame.draw.rect(self.screen, (30, 35, 45), (bx, by, bar_w, bar_h))
            hp_color = (
                Colors.HUD_ACCENT if hp_frac > 0.5 else
                Colors.HUD_WARNING if hp_frac > 0.25 else
                Colors.HUD_CRITICAL
            )
            pygame.draw.rect(self.screen, hp_color, (bx, by, int(bar_w * hp_frac), bar_h))

    def _draw_red_entity(self, x: int, y: int, entity):
        pulse = abs(math.sin(self.frame_count * 0.1)) * 0.5 + 0.5

        if entity.entity_type == "drone":
            size = int(4 + pulse * 2)
            pygame.draw.circle(self.screen, Colors.THREAT_RED, (x, y), size)
            glow_s = pygame.Surface((size * 6, size * 6), pygame.SRCALPHA)
            pygame.draw.circle(
                glow_s, (*Colors.THREAT_RED, int(35 * pulse)),
                (size * 3, size * 3), size * 3
            )
            self.screen.blit(glow_s, (x - size * 3, y - size * 3))

        elif entity.entity_type == "missile":
            color = Colors.MISSILE_AMBER if self.blink_state else Colors.THREAT_RED
            sz = 8
            points = [(x, y - sz), (x + sz, y), (x, y + sz), (x - sz, y)]
            pygame.gfxdraw.filled_polygon(self.screen, points, color)
            pygame.gfxdraw.aapolygon(self.screen, points, color)
            glow_s = pygame.Surface((sz * 6, sz * 6), pygame.SRCALPHA)
            pygame.draw.circle(
                glow_s, (*Colors.MISSILE_AMBER, int(55 * pulse)),
                (sz * 3, sz * 3), sz * 3
            )
            self.screen.blit(glow_s, (x - sz * 3, y - sz * 3))
            if self.blink_state:
                lbl = self.font_tiny.render("BALLISTIC", True, Colors.HUD_WARNING)
                self.screen.blit(lbl, (x + 12, y - 6))

    def _draw_radar_icon(self, x: int, y: int, radar):
        if not radar.operational:
            pygame.draw.line(self.screen, Colors.HUD_CRITICAL, (x - 8, y - 8), (x + 8, y + 8), 2)
            pygame.draw.line(self.screen, Colors.HUD_CRITICAL, (x + 8, y - 8), (x - 8, y + 8), 2)
            return

        for r in [12, 8, 4]:
            pygame.gfxdraw.aacircle(self.screen, x, y, r, Colors.RADAR_GREEN)

        ant_angle = self.frame_count * 0.05
        ant_x = int(x + math.cos(ant_angle) * 14)
        ant_y = int(y + math.sin(ant_angle) * 14)
        pygame.draw.line(self.screen, Colors.RADAR_GREEN, (x, y), (ant_x, ant_y), 2)

        label = self.font_tiny.render("RADAR-1", True, Colors.RADAR_GREEN)
        self.screen.blit(label, (x + 16, y - 6))

        glow_s = pygame.Surface((50, 50), pygame.SRCALPHA)
        pygame.draw.circle(glow_s, (*Colors.RADAR_GREEN[:3], 18), (25, 25), 25)
        self.screen.blit(glow_s, (x - 25, y - 25))

    def _draw_effects(self, state: dict):
        for fx in state.get("effects", []):
            sx, sy = self.world_to_screen(fx.x, fx.y)
            r = max(1, int(fx.radius))
            alpha = max(0, min(255, fx.alpha))

            if fx.effect_type == "intercept":
                s = pygame.Surface((r * 2 + 6, r * 2 + 6), pygame.SRCALPHA)
                pygame.draw.circle(s, (255, 255, 255, alpha), (r + 3, r + 3), r, 2)
                self.screen.blit(s, (sx - r - 3, sy - r - 3))
                if fx.lifetime > fx.max_lifetime * 0.6:
                    flash_r = max(1, r // 3)
                    pygame.draw.circle(self.screen, (255, 255, 255), (sx, sy), flash_r)

            elif fx.effect_type == "explosion":
                s = pygame.Surface((r * 2 + 6, r * 2 + 6), pygame.SRCALPHA)
                pygame.draw.circle(s, (*fx.color, alpha), (r + 3, r + 3), r, 2)
                inner_r = max(1, r // 2)
                pygame.draw.circle(s, (*fx.color, max(0, alpha - 90)), (r + 3, r + 3), inner_r)
                self.screen.blit(s, (sx - r - 3, sy - r - 3))

            elif fx.effect_type == "beam":
                sx2, sy2 = self.world_to_screen(fx.x2, fx.y2)
                self._temp_layer.fill((0, 0, 0, 0))
                beam_surf = self._temp_layer
                pygame.draw.line(beam_surf, (*fx.color, alpha), (sx, sy), (sx2, sy2), 2)
                pygame.draw.line(beam_surf, (255, 255, 255, alpha // 2), (sx, sy), (sx2, sy2), 1)
                self.screen.blit(beam_surf, (0, 0))

            elif fx.effect_type == "miss":
                s = pygame.Surface((r * 2 + 4, r * 2 + 4), pygame.SRCALPHA)
                pygame.draw.circle(s, (*fx.color, alpha // 2), (r + 2, r + 2), r, 1)
                self.screen.blit(s, (sx - r - 2, sy - r - 2))

    def _draw_wave_warning(self):
        if self._wave_warning_timer <= 0:
            return

        frac = self._wave_warning_timer / 90
        alpha = int(min(220, 220 * frac))
        overlay = pygame.Surface((BATTLEFIELD_SIZE, 80), pygame.SRCALPHA)
        overlay.fill((180, 20, 20, max(0, alpha // 4)))
        self.screen.blit(overlay, (0, BATTLEFIELD_SIZE // 2 - 40))

        pulse = abs(math.sin(self.frame_count * 0.15))
        color = (255, int(50 + 50 * pulse), 50)
        txt = self.font_large.render(
            f"WAVE {self._wave_warning_number}  INBOUND", True, color
        )
        tx = (BATTLEFIELD_SIZE - txt.get_width()) // 2
        ty = BATTLEFIELD_SIZE // 2 - txt.get_height() // 2
        self.screen.blit(txt, (tx, ty))


    def _draw_top_bar(self, state: dict):
        bar_h = 32
        bar_surf = pygame.Surface((SCREEN_WIDTH, bar_h), pygame.SRCALPHA)
        bar_surf.fill((*Colors.BG_PANEL, 215))
        pygame.draw.line(bar_surf, (*Colors.HUD_ACCENT, 110), (0, bar_h - 1), (SCREEN_WIDTH, bar_h - 1))
        pygame.draw.line(bar_surf, (*Colors.HUD_ACCENT, 35), (0, bar_h - 2), (SCREEN_WIDTH, bar_h - 2))
        self.screen.blit(bar_surf, (0, 0))

        title = self.font_medium.render("JADC2  TACTICAL DEFENSE COMMAND", True, Colors.HUD_ACCENT)
        self.screen.blit(title, (12, 7))

        step = state.get("step", 0)
        max_steps = state.get("max_steps", 500)
        kills = state.get("kills", 0)
        score = state.get("score", 0.0)

        step_txt = self.font_medium.render(f"T+{step:04d}/{max_steps}", True, Colors.HUD_BRIGHT)
        self.screen.blit(step_txt, (SCREEN_WIDTH - 175, 7))

        kill_txt = self.font_medium.render(f"KILLS: {kills}", True, Colors.HUD_ACCENT)
        self.screen.blit(kill_txt, (SCREEN_WIDTH - 340, 7))

        radar_alive = state.get("radar_alive", True)
        if radar_alive:
            status_color = Colors.RADAR_GREEN
            status_text = "RADAR ONLINE"
        else:
            status_color = Colors.HUD_CRITICAL if self.blink_state else Colors.BG_DEEP
            status_text = "RADAR OFFLINE"
        status_surf = self.font_medium.render(status_text, True, status_color)
        self.screen.blit(status_surf, (430, 7))

        wave = state.get("wave", 0)
        wave_txt = self.font_medium.render(f"WAVE {wave}", True, Colors.HUD_WARNING)
        self.screen.blit(wave_txt, (620, 7))

    def _draw_hud_panel(self, state: dict):
        px = BATTLEFIELD_SIZE
        self.screen.blit(self._hud_bg_surface, (px, 0))

        y = 40
        self._hud_section_header(px + 10, y, "FORCE STATUS", Colors.HUD_ACCENT)
        y += 28

        for ent in state.get("blue_entities", []):
            if y > 250:
                break
            self._draw_entity_status_row(px + 10, y, ent)
            y += 22

        radar = state.get("radar")
        if radar and y <= 250:
            self._draw_radar_status_row(px + 10, y, radar)

        y = 268
        self._hud_section_header(px + 10, y, "THREAT BOARD", Colors.HUD_CRITICAL)
        y += 28

        red_entities = state.get("red_entities", [])
        active_drones   = sum(1 for e in red_entities if e.entity_type == "drone"    and e.active)
        active_missiles = sum(1 for e in red_entities if e.entity_type == "missile"  and e.active)
        total_destroyed = sum(1 for e in red_entities if not e.active)

        self._draw_threat_row(px + 10, y, "DRONES",    active_drones,   Colors.THREAT_RED)
        y += 22
        self._draw_threat_row(px + 10, y, "BALLISTIC", active_missiles, Colors.MISSILE_AMBER)
        y += 22
        self._draw_threat_row(px + 10, y, "DESTROYED", total_destroyed, Colors.RADAR_GREEN)
        y += 28

        threat_level = min(1.0, (active_drones + active_missiles) / 20.0)
        self._draw_threat_bar(px + 10, y, threat_level)
        y += 22

        self._draw_threat_waveform(px + 10, y + 8)

        y = 455
        self._hud_section_header(px + 10, y, "AMMUNITION", Colors.HUD_WARNING)
        y += 28

        for ent in state.get("blue_entities", []):
            if y > 578:
                break
            self._draw_ammo_row(px + 10, y, ent)
            y += 20

        y = 596
        self._hud_section_header(px + 10, y, "EVENT LOG", Colors.HUD_TEXT)
        y += 24

        for event_text in self.event_log[-self.max_events:]:
            if y > SCREEN_HEIGHT - 16:
                break
            ev_surf = self.font_tiny.render(event_text, True, Colors.HUD_TEXT)
            self.screen.blit(ev_surf, (px + 12, y))
            y += 16

    def _hud_section_header(self, x: int, y: int, text: str, color: tuple):
        bar = pygame.Surface((HUD_WIDTH - 20, 18), pygame.SRCALPHA)
        bar.fill((*color, 18))
        self.screen.blit(bar, (x, y))
        surf = self.font_medium.render(text, True, color)
        self.screen.blit(surf, (x + 4, y))

    def _draw_entity_status_row(self, x: int, y: int, entity):
        color_map = {
            "thaad":  Colors.THAAD_COLOR,
            "aegis":  Colors.AEGIS_COLOR,
            "armor":  Colors.ARMOR_COLOR,
            "bomber": Colors.BOMBER_COLOR,
        }
        icon_char = {"thaad": "D", "aegis": "V", "armor": "S", "bomber": "A"}
        color = color_map.get(entity.entity_type, Colors.FRIENDLY_CYAN)
        icon  = icon_char.get(entity.entity_type, "o")

        if entity.active:
            status_c    = Colors.RADAR_GREEN if entity.hp > entity.max_hp * 0.5 else Colors.HUD_WARNING
            status_char = "+"
        else:
            status_c    = Colors.HUD_CRITICAL
            status_char = "X"

        st = self.font_small.render(status_char, True, status_c)
        self.screen.blit(st, (x, y))
        ic = self.font_small.render(f"{icon} {entity.entity_id.upper()}", True, color)
        self.screen.blit(ic, (x + 14, y))
        hp = self.font_tiny.render(f"HP {entity.hp}/{entity.max_hp}", True, Colors.HUD_TEXT)
        self.screen.blit(hp, (x + 155, y + 1))

    def _draw_radar_status_row(self, x: int, y: int, radar):
        if radar.operational:
            color = Colors.RADAR_GREEN
            text  = f"+ (R) RADAR-1  HP {radar.hp}/{radar.max_hp}"
        else:
            color = Colors.HUD_CRITICAL
            text  = f"X (R) RADAR-1  DESTROYED"
        surf = self.font_small.render(text, True, color)
        self.screen.blit(surf, (x, y))

    def _draw_threat_row(self, x: int, y: int, label: str, count: int, color: tuple):
        lbl = self.font_small.render(f"{label}:", True, Colors.HUD_TEXT)
        self.screen.blit(lbl, (x, y))
        cnt_color = color if count > 0 else Colors.HUD_TEXT
        cnt = self.font_medium.render(f"{count:3d}", True, cnt_color)
        self.screen.blit(cnt, (x + 148, y - 1))

    def _draw_threat_bar(self, x: int, y: int, level: float):
        bar_w = HUD_WIDTH - 36
        bar_h = 10
        pygame.draw.rect(self.screen, Colors.BG_DEEP, (x, y, bar_w, bar_h))
        pygame.draw.rect(self.screen, Colors.HUD_BORDER, (x, y, bar_w, bar_h), 1)

        fill_w = max(0, int(bar_w * level))
        fill_color = (
            Colors.RADAR_GREEN   if level < 0.33 else
            Colors.HUD_WARNING   if level < 0.66 else
            Colors.HUD_CRITICAL
        )
        if fill_w > 2:
            pygame.draw.rect(self.screen, fill_color, (x + 1, y + 1, fill_w - 2, bar_h - 2))

        lbl = self.font_tiny.render(f"THREAT LEVEL  {int(level * 100)}%", True, Colors.HUD_BRIGHT)
        self.screen.blit(lbl, (x + bar_w // 2 - lbl.get_width() // 2, y - 14))

    def _draw_threat_waveform(self, x: int, y: int):
        if len(self._threat_history) < 2:
            return

        w = HUD_WIDTH - 36
        h = 28
        surf = pygame.Surface((w, h), pygame.SRCALPHA)
        surf.fill((0, 0, 0, 40))
        pygame.draw.rect(surf, (*Colors.HUD_BORDER, 60), (0, 0, w, h), 1)

        max_val = max(1, max(self._threat_history))
        n = len(self._threat_history)

        for i in range(1, n):
            x1 = int((i - 1) / (self._threat_history_max - 1) * w)
            x2 = int(i / (self._threat_history_max - 1) * w)
            y1 = h - int(self._threat_history[i - 1] / max_val * (h - 4)) - 2
            y2 = h - int(self._threat_history[i]     / max_val * (h - 4)) - 2
            alpha = int(80 + 120 * (i / n))
            pygame.draw.line(surf, (*Colors.HUD_WARNING, alpha), (x1, y1), (x2, y2), 1)

        self.screen.blit(surf, (x, y))

    def _draw_ammo_row(self, x: int, y: int, entity):
        color_map = {
            "thaad":  Colors.THAAD_COLOR,
            "aegis":  Colors.AEGIS_COLOR,
            "armor":  Colors.ARMOR_COLOR,
            "bomber": Colors.BOMBER_COLOR,
        }
        color = color_map.get(entity.entity_type, Colors.HUD_TEXT)
        name = entity.entity_id.upper()[:8]
        name_surf = self.font_tiny.render(name, True, color)
        self.screen.blit(name_surf, (x, y))

        if entity.entity_type == "thaad":
            ammo_text = f"SM3:{entity.ammo_expensive}  PAC:{entity.ammo_cheap}"
        elif entity.entity_type == "aegis":
            ammo_text = f"SM3:{entity.ammo_sm3}"
        elif entity.entity_type == "armor":
            ammo_text = f"ABR:{entity.ammo_airburst}"
        elif entity.entity_type == "bomber":
            ammo_text = f"BOM:{entity.ammo_bombs}"
        else:
            return

        ammo_surf = self.font_tiny.render(ammo_text, True, Colors.HUD_TEXT)
        self.screen.blit(ammo_surf, (x + 88, y))

    def _draw_vignette(self):
        corner_size = 120
        for cx, cy in [(0, 0), (SCREEN_WIDTH - corner_size, 0),
                       (0, SCREEN_HEIGHT - corner_size),
                       (SCREEN_WIDTH - corner_size, SCREEN_HEIGHT - corner_size)]:
            vig = pygame.Surface((corner_size, corner_size), pygame.SRCALPHA)
            for i in range(corner_size):
                alpha = max(0, int(28 * (1 - i / corner_size)))
                pygame.draw.line(vig, (0, 0, 0, alpha), (0, i), (corner_size, i))
            self.screen.blit(vig, (cx, cy))


    def trigger_wave_warning(self, wave_number: int):
        self._wave_warning_timer  = 90
        self._wave_warning_number = wave_number

    def log_event(self, text: str):
        step = self.env.current_step if hasattr(self.env, "current_step") else 0
        self.event_log.append(f"T{step:04d} {text}")
        if len(self.event_log) > 60:
            self.event_log = self.event_log[-60:]

    def close(self):
        if self.initialized:
            pygame.quit()
            self.initialized = False


import random
