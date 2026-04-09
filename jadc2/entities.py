"""
JADC2 Entities — Dataclass definitions for all battlefield objects.
====================================================================
Blue Team (learning agents) and Red Team (scripted adversaries).
Each entity tracks position, health, ammo, cooldowns, and status.
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from jadc2.config import (
    THAAD_HP, THAAD_AMMO_EXPENSIVE, THAAD_AMMO_CHEAP, THAAD_RANGE, THAAD_COOLDOWN,
    AEGIS_HP, AEGIS_AMMO_SM3, AEGIS_SPEED, AEGIS_RANGE, AEGIS_RADAR_RANGE, AEGIS_COOLDOWN,
    ARMOR_HP, ARMOR_AMMO_AIRBURST, ARMOR_SPEED, ARMOR_RANGE, ARMOR_COOLDOWN,
    BOMBER_HP, BOMBER_AMMO_BOMBS, BOMBER_SPEED, BOMBER_RANGE, BOMBER_STEALTH, BOMBER_COOLDOWN,
    RADAR_HP, RADAR_DETECTION_RANGE,
    DRONE_HP, DRONE_SPEED, DRONE_DAMAGE,
    MISSILE_HP, MISSILE_SPEED, MISSILE_DAMAGE,
    WORLD_SIZE,
)


# ══════════════════════════════════════════════════════════════════════
#  BASE ENTITY
# ══════════════════════════════════════════════════════════════════════

@dataclass
class Entity:
    """Base class for all battlefield entities."""
    x: float = 0.0
    y: float = 0.0
    hp: int = 1
    max_hp: int = 1
    active: bool = True
    entity_type: str = "generic"
    team: str = "neutral"       # "blue" or "red"
    entity_id: str = ""

    def take_damage(self, amount: int) -> bool:
        """Apply damage. Returns True if entity is destroyed."""
        self.hp = max(0, self.hp - amount)
        if self.hp <= 0:
            self.active = False
            return True
        return False

    def distance_to(self, other: "Entity") -> float:
        """Euclidean distance to another entity."""
        return np.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

    def distance_to_point(self, px: float, py: float) -> float:
        """Euclidean distance to an (x, y) coordinate."""
        return np.sqrt((self.x - px) ** 2 + (self.y - py) ** 2)

    def clamp_position(self):
        """Keep entity within world boundaries."""
        self.x = np.clip(self.x, 0, WORLD_SIZE)
        self.y = np.clip(self.y, 0, WORLD_SIZE)


# ══════════════════════════════════════════════════════════════════════
#  BLUE TEAM — LEARNING AGENTS
# ══════════════════════════════════════════════════════════════════════

@dataclass
class AirDefenseBattery(Entity):
    """
    THAAD / Patriot Battery — Stationary air defense system.
    Can fire expensive (SM-3) or cheap (PAC-3) interceptors.
    Can toggle radar between active/passive to avoid anti-radiation missiles.
    """
    hp: int = THAAD_HP
    max_hp: int = THAAD_HP
    entity_type: str = "thaad"
    team: str = "blue"

    # Ammunition
    ammo_expensive: int = THAAD_AMMO_EXPENSIVE
    ammo_cheap: int = THAAD_AMMO_CHEAP

    # Systems
    radar_active: bool = True
    engagement_range: float = THAAD_RANGE
    cooldown: int = 0
    cooldown_max: int = THAAD_COOLDOWN

    def can_fire(self) -> bool:
        return self.active and self.cooldown == 0

    def fire_expensive(self) -> bool:
        """Fire SM-3 class interceptor. Returns True if successful."""
        if self.can_fire() and self.ammo_expensive > 0:
            self.ammo_expensive -= 1
            self.cooldown = self.cooldown_max
            return True
        return False

    def fire_cheap(self) -> bool:
        """Fire PAC-3 class interceptor. Returns True if successful."""
        if self.can_fire() and self.ammo_cheap > 0:
            self.ammo_cheap -= 1
            self.cooldown = self.cooldown_max
            return True
        return False

    def toggle_radar(self):
        """Switch radar active/passive."""
        self.radar_active = not self.radar_active

    def tick_cooldown(self):
        if self.cooldown > 0:
            self.cooldown -= 1


@dataclass
class AegisDestroyer(Entity):
    """
    Aegis Destroyer — Mobile naval platform with SM-3 interceptors.
    Can share radar telemetry with land-based systems.
    """
    hp: int = AEGIS_HP
    max_hp: int = AEGIS_HP
    entity_type: str = "aegis"
    team: str = "blue"

    ammo_sm3: int = AEGIS_AMMO_SM3
    speed: float = AEGIS_SPEED
    engagement_range: float = AEGIS_RANGE
    radar_range: float = AEGIS_RADAR_RANGE
    radar_active: bool = True
    cooldown: int = 0
    cooldown_max: int = AEGIS_COOLDOWN
    sharing_telemetry: bool = False

    def can_fire(self) -> bool:
        return self.active and self.cooldown == 0

    def fire_sm3(self) -> bool:
        if self.can_fire() and self.ammo_sm3 > 0:
            self.ammo_sm3 -= 1
            self.cooldown = self.cooldown_max
            return True
        return False

    def move(self, dx: float, dy: float):
        if self.active:
            self.x += dx * self.speed
            self.y += dy * self.speed
            self.clamp_position()

    def share_telemetry(self):
        self.sharing_telemetry = True

    def tick_cooldown(self):
        if self.cooldown > 0:
            self.cooldown -= 1
        self.sharing_telemetry = False


@dataclass
class ArmoredColumn(Entity):
    """
    Armored Column / Tank — Mobile ground unit for close-range drone defense.
    Uses proximity airburst rounds against low-altitude drones.
    Can secure ground launch zones.
    """
    hp: int = ARMOR_HP
    max_hp: int = ARMOR_HP
    entity_type: str = "armor"
    team: str = "blue"

    ammo_airburst: int = ARMOR_AMMO_AIRBURST
    speed: float = ARMOR_SPEED
    engagement_range: float = ARMOR_RANGE
    cooldown: int = 0
    cooldown_max: int = ARMOR_COOLDOWN
    is_secured: bool = False   # Holding a defensive position

    def can_fire(self) -> bool:
        return self.active and self.cooldown == 0

    def fire_airburst(self) -> bool:
        if self.can_fire() and self.ammo_airburst > 0:
            self.ammo_airburst -= 1
            self.cooldown = self.cooldown_max
            return True
        return False

    def move(self, dx: float, dy: float):
        if self.active:
            self.x += dx * self.speed
            self.y += dy * self.speed
            self.clamp_position()
            self.is_secured = False

    def secure_position(self):
        self.is_secured = True

    def tick_cooldown(self):
        if self.cooldown > 0:
            self.cooldown -= 1


@dataclass
class StealthBomber(Entity):
    """
    Stealth Bomber — Fast, fragile air unit for striking Red Team launch sites.
    """
    hp: int = BOMBER_HP
    max_hp: int = BOMBER_HP
    entity_type: str = "bomber"
    team: str = "blue"

    ammo_bombs: int = BOMBER_AMMO_BOMBS
    speed: float = BOMBER_SPEED
    engagement_range: float = BOMBER_RANGE
    stealth_factor: float = BOMBER_STEALTH
    cooldown: int = 0
    cooldown_max: int = BOMBER_COOLDOWN

    def can_fire(self) -> bool:
        return self.active and self.cooldown == 0

    def drop_bomb(self) -> bool:
        if self.can_fire() and self.ammo_bombs > 0:
            self.ammo_bombs -= 1
            self.cooldown = self.cooldown_max
            return True
        return False

    def move(self, dx: float, dy: float):
        if self.active:
            norm = np.sqrt(dx**2 + dy**2) if (dx != 0 or dy != 0) else 1.0
            self.x += (dx / norm) * self.speed if norm > 0 else 0
            self.y += (dy / norm) * self.speed if norm > 0 else 0
            self.clamp_position()

    def tick_cooldown(self):
        if self.cooldown > 0:
            self.cooldown -= 1


@dataclass
class RadarStation(Entity):
    """
    Central Early-Warning Radar — Static high-value asset.
    Provides long-range detection. Primary Red Team target.
    """
    hp: int = RADAR_HP
    max_hp: int = RADAR_HP
    entity_type: str = "radar"
    team: str = "blue"

    detection_range: float = RADAR_DETECTION_RANGE
    operational: bool = True

    def take_damage(self, amount: int) -> bool:
        destroyed = super().take_damage(amount)
        if destroyed:
            self.operational = False
        return destroyed


# ══════════════════════════════════════════════════════════════════════
#  RED TEAM — SCRIPTED ADVERSARIES (Spawning logic in Phase 3)
# ══════════════════════════════════════════════════════════════════════

@dataclass
class Drone(Entity):
    """Low-cost decoy drone — cheap, expendable, fast."""
    hp: int = DRONE_HP
    max_hp: int = DRONE_HP
    entity_type: str = "drone"
    team: str = "red"

    speed: float = DRONE_SPEED
    damage: int = DRONE_DAMAGE
    is_decoy: bool = True
    trail: list = field(default_factory=list)
    trail_max: int = 12

    def move_toward(self, tx: float, ty: float):
        """Move directly toward a target position."""
        if not self.active:
            return
        self.trail.append((self.x, self.y))
        if len(self.trail) > self.trail_max:
            self.trail.pop(0)
        dx = tx - self.x
        dy = ty - self.y
        dist = np.sqrt(dx**2 + dy**2)
        if dist > 0:
            self.x += (dx / dist) * self.speed
            self.y += (dy / dist) * self.speed
            self.clamp_position()


@dataclass
class BallisticMissile(Entity):
    """High-value ballistic missile — targets Blue Team radar systems."""
    hp: int = MISSILE_HP
    max_hp: int = MISSILE_HP
    entity_type: str = "missile"
    team: str = "red"

    speed: float = MISSILE_SPEED
    damage: int = MISSILE_DAMAGE
    target_x: float = 500.0
    target_y: float = 700.0   # Default: radar position
    trail: list = field(default_factory=list)
    trail_max: int = 22

    def move_toward_target(self):
        """Fly toward assigned target."""
        if not self.active:
            return
        self.trail.append((self.x, self.y))
        if len(self.trail) > self.trail_max:
            self.trail.pop(0)
        dx = self.target_x - self.x
        dy = self.target_y - self.y
        dist = np.sqrt(dx**2 + dy**2)
        if dist > 0:
            self.x += (dx / dist) * self.speed
            self.y += (dy / dist) * self.speed
            self.clamp_position()


# ══════════════════════════════════════════════════════════════════════
#  VISUAL EFFECTS (for renderer)
# ══════════════════════════════════════════════════════════════════════

@dataclass
class VisualEffect:
    """Transient visual effect for the renderer (explosions, flashes, beams)."""
    x: float = 0.0
    y: float = 0.0
    x2: float = 0.0            # Endpoint for beam effects
    y2: float = 0.0
    effect_type: str = "explosion"  # "explosion", "intercept", "flash", "beam", "miss"
    radius: float = 10.0
    max_radius: float = 40.0
    alpha: int = 255
    lifetime: int = 15         # Frames remaining
    max_lifetime: int = 15
    color: tuple = (255, 180, 50)

    def tick(self) -> bool:
        """Advance effect by one frame. Returns True if expired."""
        self.lifetime -= 1
        self.radius = min(self.radius + 2.5, self.max_radius)
        self.alpha = max(0, int(255 * (self.lifetime / max(1, self.max_lifetime))))
        return self.lifetime <= 0
