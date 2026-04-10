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


@dataclass
class Entity:
    x: float = 0.0
    y: float = 0.0
    hp: int = 1
    max_hp: int = 1
    active: bool = True
    entity_type: str = "generic"
    team: str = "neutral"
    entity_id: str = ""

    def take_damage(self, amount: int) -> bool:
        self.hp = max(0, self.hp - amount)
        if self.hp <= 0:
            self.active = False
            return True
        return False

    def distance_to(self, other: "Entity") -> float:
        return np.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

    def distance_to_point(self, px: float, py: float) -> float:
        return np.sqrt((self.x - px) ** 2 + (self.y - py) ** 2)

    def clamp_position(self):
        self.x = np.clip(self.x, 0, WORLD_SIZE)
        self.y = np.clip(self.y, 0, WORLD_SIZE)


@dataclass
class AirDefenseBattery(Entity):
    hp: int = THAAD_HP
    max_hp: int = THAAD_HP
    entity_type: str = "thaad"
    team: str = "blue"

    ammo_expensive: int = THAAD_AMMO_EXPENSIVE
    ammo_cheap: int = THAAD_AMMO_CHEAP

    radar_active: bool = True
    engagement_range: float = THAAD_RANGE
    cooldown: int = 0
    cooldown_max: int = THAAD_COOLDOWN

    def can_fire(self) -> bool:
        return self.active and self.cooldown == 0

    def fire_expensive(self) -> bool:
        if self.can_fire() and self.ammo_expensive > 0:
            self.ammo_expensive -= 1
            self.cooldown = self.cooldown_max
            return True
        return False

    def fire_cheap(self) -> bool:
        if self.can_fire() and self.ammo_cheap > 0:
            self.ammo_cheap -= 1
            self.cooldown = self.cooldown_max
            return True
        return False

    def toggle_radar(self):
        self.radar_active = not self.radar_active

    def tick_cooldown(self):
        if self.cooldown > 0:
            self.cooldown -= 1


@dataclass
class AegisDestroyer(Entity):
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
    hp: int = ARMOR_HP
    max_hp: int = ARMOR_HP
    entity_type: str = "armor"
    team: str = "blue"

    ammo_airburst: int = ARMOR_AMMO_AIRBURST
    speed: float = ARMOR_SPEED
    engagement_range: float = ARMOR_RANGE
    cooldown: int = 0
    cooldown_max: int = ARMOR_COOLDOWN
    is_secured: bool = False

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


@dataclass
class Drone(Entity):
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
    hp: int = MISSILE_HP
    max_hp: int = MISSILE_HP
    entity_type: str = "missile"
    team: str = "red"

    speed: float = MISSILE_SPEED
    damage: int = MISSILE_DAMAGE
    target_x: float = 500.0
    target_y: float = 700.0
    trail: list = field(default_factory=list)
    trail_max: int = 22

    def move_toward_target(self):
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


@dataclass
class VisualEffect:
    x: float = 0.0
    y: float = 0.0
    x2: float = 0.0
    y2: float = 0.0
    effect_type: str = "explosion"
    radius: float = 10.0
    max_radius: float = 40.0
    alpha: int = 255
    lifetime: int = 15
    max_lifetime: int = 15
    color: tuple = (255, 180, 50)

    def tick(self) -> bool:
        self.lifetime -= 1
        self.radius = min(self.radius + 2.5, self.max_radius)
        self.alpha = max(0, int(255 * (self.lifetime / max(1, self.max_lifetime))))
        return self.lifetime <= 0
