
from __future__ import annotations

import functools
import random
from typing import Dict, List, Optional, Any

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

from jadc2.config import (
    WORLD_SIZE, GRID_DIM, CELL_SIZE, MAX_STEPS, NUM_OBS_CHANNELS,
    NUM_THAAD, NUM_AEGIS, NUM_ARMOR, NUM_BOMBER, NUM_RADAR,
    THAAD_NUM_ACTIONS, AEGIS_MOVE_ACTIONS, AEGIS_COMBAT_ACTIONS,
    ARMOR_MOVE_ACTIONS, ARMOR_COMBAT_ACTIONS,
    BOMBER_MOVE_ACTIONS, BOMBER_COMBAT_ACTIONS,
    MOVE_4DIR, MOVE_8DIR,
    BLUE_SPAWN_REGION, RED_SPAWN_REGION, RADAR_POSITION,
    Colors, SCREEN_WIDTH, SCREEN_HEIGHT,
    W_HIT, W_COST, W_RADAR, W_DMG,
    COST_THAAD_EXPENSIVE, COST_THAAD_CHEAP, COST_AEGIS_SM3,
    COST_ARMOR_AIRBURST, COST_BOMBER_BOMB,
    DRONE_COST, MISSILE_COST,
    HIT_PROB_SM3_MISSILE, HIT_PROB_SM3_DRONE,
    HIT_PROB_PAC3_DRONE, HIT_PROB_PAC3_MISSILE,
    HIT_PROB_AEGIS_SM3, HIT_PROB_AIRBURST, HIT_PROB_BOMB,
    COLLISION_RADIUS_DRONE, COLLISION_RADIUS_MISSILE,
    DRONE_RADAR_PRIORITY, MISSILE_RETARGET_PROB,
    WAVE_SCHEDULE,
)
from jadc2.entities import (
    Entity, AirDefenseBattery, AegisDestroyer, ArmoredColumn,
    StealthBomber, RadarStation, Drone, BallisticMissile, VisualEffect,
)
from jadc2.renderer import MilitaryRadarRenderer

import os as _os
import platform
if platform.system() != "Windows":
    if _os.environ.get("SDL_VIDEODRIVER") is None and _os.environ.get("DISPLAY") is None:
        _os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        _os.environ.setdefault("SDL_AUDIODRIVER", "dummy")


def env(render_mode: Optional[str] = None) -> JADC2_Env:
    return JADC2_Env(render_mode=render_mode)


class JADC2_Env(ParallelEnv):

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "jadc2_v0",
        "is_parallelizable": True,
        "render_fps": 30,
    }

    def __init__(self, render_mode: Optional[str] = None):
        super().__init__()

        self.render_mode = render_mode

        self.possible_agents = []
        self._agent_types = {}

        for i in range(NUM_THAAD):
            aid = f"thaad_{i}"
            self.possible_agents.append(aid)
            self._agent_types[aid] = "thaad"

        for i in range(NUM_AEGIS):
            aid = f"aegis_{i}"
            self.possible_agents.append(aid)
            self._agent_types[aid] = "aegis"

        for i in range(NUM_ARMOR):
            aid = f"armor_{i}"
            self.possible_agents.append(aid)
            self._agent_types[aid] = "armor"

        for i in range(NUM_BOMBER):
            aid = f"bomber_{i}"
            self.possible_agents.append(aid)
            self._agent_types[aid] = "bomber"

        self.agents = list(self.possible_agents)

        self._agent_ids = set(self.possible_agents)

        self._observation_spaces = {}
        self._action_spaces = {}

        for agent_id in self.possible_agents:
            self._observation_spaces[agent_id] = spaces.Box(
                low=0.0, high=1.0,
                shape=(GRID_DIM, GRID_DIM, NUM_OBS_CHANNELS),
                dtype=np.float32,
            )

            agent_type = self._agent_types[agent_id]
            if agent_type == "thaad":
                self._action_spaces[agent_id] = spaces.Discrete(THAAD_NUM_ACTIONS)
            elif agent_type == "aegis":
                self._action_spaces[agent_id] = spaces.MultiDiscrete(
                    [AEGIS_MOVE_ACTIONS, AEGIS_COMBAT_ACTIONS]
                )
            elif agent_type == "armor":
                self._action_spaces[agent_id] = spaces.MultiDiscrete(
                    [ARMOR_MOVE_ACTIONS, ARMOR_COMBAT_ACTIONS]
                )
            elif agent_type == "bomber":
                self._action_spaces[agent_id] = spaces.MultiDiscrete(
                    [BOMBER_MOVE_ACTIONS, BOMBER_COMBAT_ACTIONS]
                )

        self._blue_entities: Dict[str, Entity] = {}
        self._red_entities: List[Entity] = []
        self._radar: Optional[RadarStation] = None
        self._effects: List[VisualEffect] = []
        self.current_step = 0
        self._cumulative_rewards = {a: 0.0 for a in self.possible_agents}
        self._total_kills = 0
        self._total_score = 0.0
        self._wave_number = 0
        self._red_entity_counter = 0

        self._renderer: Optional[MilitaryRadarRenderer] = None
        if self.render_mode == "human":
            self._renderer = MilitaryRadarRenderer(self)


    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: str) -> spaces.Space:
        return self._observation_spaces[agent]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: str) -> spaces.Space:
        return self._action_spaces[agent]


    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[Dict[str, np.ndarray], Dict[str, dict]]:
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        self.agents = list(self.possible_agents)
        self.current_step = 0
        self._effects = []
        self._red_entities = []
        self._cumulative_rewards = {a: 0.0 for a in self.possible_agents}
        self._total_kills = 0
        self._total_score = 0.0
        self._wave_number = 0
        self._red_entity_counter = 0

        self._blue_entities = {}
        spawn = BLUE_SPAWN_REGION

        for i in range(NUM_THAAD):
            aid = f"thaad_{i}"
            x = spawn["x_min"] + (i + 1) * (spawn["x_max"] - spawn["x_min"]) / (NUM_THAAD + 1)
            y = spawn["y_min"] + np.random.uniform(0, 100)
            self._blue_entities[aid] = AirDefenseBattery(x=x, y=y, entity_id=aid)

        for i in range(NUM_AEGIS):
            aid = f"aegis_{i}"
            x = spawn["x_min"] + (i + 1) * (spawn["x_max"] - spawn["x_min"]) / (NUM_AEGIS + 1)
            y = spawn["y_min"] - 100 + np.random.uniform(-50, 50)
            self._blue_entities[aid] = AegisDestroyer(x=x, y=y, entity_id=aid)

        for i in range(NUM_ARMOR):
            aid = f"armor_{i}"
            x = RADAR_POSITION[0] + (i - NUM_ARMOR / 2) * 100 + np.random.uniform(-30, 30)
            y = RADAR_POSITION[1] + np.random.uniform(-50, 50)
            self._blue_entities[aid] = ArmoredColumn(x=x, y=y, entity_id=aid)

        for i in range(NUM_BOMBER):
            aid = f"bomber_{i}"
            x = spawn["x_min"] + np.random.uniform(200, 600)
            y = spawn["y_max"] - np.random.uniform(0, 50)
            self._blue_entities[aid] = StealthBomber(x=x, y=y, entity_id=aid)

        self._radar = RadarStation(
            x=RADAR_POSITION[0],
            y=RADAR_POSITION[1],
            entity_id="radar_0",
        )

        self._spawn_wave(5, 1, initial=True)

        observations = {agent: self._build_observation(agent) for agent in self.agents}
        infos = {agent: {} for agent in self.agents}

        if self._renderer:
            self._renderer.log_event("SIMULATION INITIALIZED")
            self._renderer.log_event(f"BLUE FORCE: {len(self._blue_entities)} assets deployed")
            self._renderer.kill_count = 0
            self._renderer.score = 0.0

        return observations, infos


    def step(
        self, actions: Dict[str, Any]
    ) -> tuple[
        Dict[str, np.ndarray],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, bool],
        Dict[str, dict],
    ]:
        self.current_step += 1
        step_rewards = {a: 0.0 for a in self.possible_agents}

        for agent_id, action in actions.items():
            if agent_id not in self._blue_entities:
                continue
            entity = self._blue_entities[agent_id]
            if not entity.active:
                continue
            agent_type = self._agent_types[agent_id]
            reward = self._process_action(agent_id, agent_type, entity, action)
            step_rewards[agent_id] += reward

        for ent in self._blue_entities.values():
            if hasattr(ent, "tick_cooldown"):
                ent.tick_cooldown()

        self._tick_red_ai()

        damage_events = self._check_collisions()

        self._check_wave_spawn()

        if self._radar and self._radar.operational:
            for a in self.possible_agents:
                step_rewards[a] += W_RADAR * 0.01

        for agent_id, damage in damage_events.items():
            step_rewards[agent_id] -= W_DMG * damage

        for a in self.possible_agents:
            self._cumulative_rewards[a] += step_rewards.get(a, 0.0)

        self._total_score = sum(self._cumulative_rewards.values())

        self._effects = [fx for fx in self._effects if not fx.tick()]

        observations = {agent: self._build_observation(agent) for agent in self.agents}
        rewards = {a: step_rewards.get(a, 0.0) for a in self.agents}

        truncations = {a: self.current_step >= MAX_STEPS for a in self.agents}
        terminations = {a: False for a in self.agents}

        if self._radar and not self._radar.operational:
            for a in self.agents:
                terminations[a] = True
            if self._renderer:
                self._renderer.log_event("MISSION FAILED — RADAR DESTROYED")

        all_blue_dead = all(not e.active for e in self._blue_entities.values())
        if all_blue_dead:
            for a in self.agents:
                terminations[a] = True
            if self._renderer:
                self._renderer.log_event("MISSION FAILED — ALL ASSETS LOST")

        infos = {a: {
            "step": self.current_step,
            "kills": self._total_kills,
            "score": self._total_score,
            "wave": self._wave_number,
        } for a in self.agents}

        self.agents = [
            a for a in self.agents
            if not terminations.get(a, False) and not truncations.get(a, False)
        ]

        return observations, rewards, terminations, truncations, infos


    def _process_action(self, agent_id: str, agent_type: str, entity, action) -> float:
        reward = 0.0

        if agent_type == "thaad":
            act = int(action)
            if act == 1:
                reward += self._thaad_fire_expensive(entity)
            elif act == 2:
                reward += self._thaad_fire_cheap(entity)
            elif act == 3:
                entity.toggle_radar()

        elif agent_type == "aegis":
            move_act = int(action[0])
            combat_act = int(action[1])
            if move_act in MOVE_4DIR:
                dx, dy = MOVE_4DIR[move_act]
                entity.move(dx, dy)
            if 1 <= combat_act <= 4:
                reward += self._aegis_fire_sm3(entity, combat_act - 1)
            elif 5 <= combat_act <= 8:
                entity.share_telemetry()

        elif agent_type == "armor":
            move_act = int(action[0])
            combat_act = int(action[1])
            if move_act in MOVE_4DIR:
                dx, dy = MOVE_4DIR[move_act]
                entity.move(dx, dy)
            if combat_act == 1:
                reward += self._armor_fire_airburst(entity)
            elif combat_act == 2:
                entity.secure_position()

        elif agent_type == "bomber":
            move_act = int(action[0])
            combat_act = int(action[1])
            if move_act in MOVE_8DIR:
                dx, dy = MOVE_8DIR[move_act]
                entity.move(dx, dy)
            if combat_act == 1:
                reward += self._bomber_drop_bomb(entity)

        return reward


    def _thaad_fire_expensive(self, entity: AirDefenseBattery) -> float:
        target = self._find_nearest_red_in_range(entity, "missile", entity.engagement_range)
        if target is None:
            target = self._find_nearest_red_in_range(entity, "drone", entity.engagement_range)
        if target is None:
            return 0.0
        if entity.fire_expensive():
            hit_prob = HIT_PROB_SM3_MISSILE if target.entity_type == "missile" else HIT_PROB_SM3_DRONE
            return self._resolve_intercept(entity, target, COST_THAAD_EXPENSIVE, hit_prob)
        return 0.0

    def _thaad_fire_cheap(self, entity: AirDefenseBattery) -> float:
        target = self._find_nearest_red_in_range(entity, "drone", entity.engagement_range)
        if target is None:
            target = self._find_nearest_red_in_range(entity, "missile", entity.engagement_range)
        if target is None:
            return 0.0
        if entity.fire_cheap():
            hit_prob = HIT_PROB_PAC3_DRONE if target.entity_type == "drone" else HIT_PROB_PAC3_MISSILE
            return self._resolve_intercept(entity, target, COST_THAAD_CHEAP, hit_prob)
        return 0.0

    def _aegis_fire_sm3(self, entity: AegisDestroyer, target_index: int) -> float:
        targets_in_range = [
            r for r in self._red_entities
            if r.active and entity.distance_to(r) <= entity.engagement_range
        ]
        targets_in_range.sort(key=lambda r: (0 if r.entity_type == "missile" else 1,
                                              entity.distance_to(r)))
        if target_index >= len(targets_in_range):
            return 0.0
        target = targets_in_range[target_index]
        if entity.fire_sm3():
            return self._resolve_intercept(entity, target, COST_AEGIS_SM3, HIT_PROB_AEGIS_SM3)
        return 0.0

    def _armor_fire_airburst(self, entity: ArmoredColumn) -> float:
        if not entity.fire_airburst():
            return 0.0
        drones_in_range = [
            r for r in self._red_entities
            if r.active and r.entity_type == "drone"
            and entity.distance_to(r) <= entity.engagement_range
        ]
        if not drones_in_range:
            return -W_COST * COST_ARMOR_AIRBURST * 0.3

        total_reward = -W_COST * COST_ARMOR_AIRBURST
        n = len(drones_in_range)

        for drone in drones_in_range:
            if random.random() < HIT_PROB_AIRBURST:
                drone.take_damage(drone.hp)
                self._effects.append(VisualEffect(
                    x=drone.x, y=drone.y,
                    effect_type="explosion",
                    color=(255, 140, 30),
                    max_radius=18, lifetime=10, max_lifetime=10,
                ))
                total_reward += W_HIT * DRONE_COST
                self._total_kills += 1
                if self._renderer:
                    self._renderer.kill_count += 1
                    self._renderer.log_event(f"AIRBURST: DRONE ELIMINATED ({n} in area)")

        return total_reward

    def _bomber_drop_bomb(self, entity: StealthBomber) -> float:
        if not entity.drop_bomb():
            return 0.0
        targets_in_range = [
            r for r in self._red_entities
            if r.active and entity.distance_to(r) <= entity.engagement_range
        ]
        if not targets_in_range:
            return -W_COST * COST_BOMBER_BOMB * 0.5

        total_reward = -W_COST * COST_BOMBER_BOMB
        self._effects.append(VisualEffect(
            x=entity.x, y=entity.y,
            effect_type="explosion",
            color=(255, 200, 60),
            radius=5, max_radius=55, lifetime=20, max_lifetime=20,
        ))

        for target in targets_in_range:
            if random.random() < HIT_PROB_BOMB:
                destroyed = target.take_damage(3)
                if destroyed:
                    val = MISSILE_COST if target.entity_type == "missile" else DRONE_COST
                    total_reward += W_HIT * val
                    self._total_kills += 1
                    if self._renderer:
                        self._renderer.kill_count += 1
                        self._renderer.log_event(
                            f"BOMB: {target.entity_type.upper()} {target.entity_id} DESTROYED"
                        )

        return total_reward

    def _resolve_intercept(
        self,
        shooter: Entity,
        target: Entity,
        weapon_cost: float,
        hit_prob: float,
    ) -> float:
        if random.random() < hit_prob:
            target.take_damage(target.hp)
            target_value = MISSILE_COST if target.entity_type == "missile" else DRONE_COST

            self._effects.append(VisualEffect(
                x=shooter.x, y=shooter.y,
                x2=target.x, y2=target.y,
                effect_type="beam",
                color=(200, 240, 255),
                lifetime=8, max_lifetime=8,
            ))
            self._effects.append(VisualEffect(
                x=target.x, y=target.y,
                effect_type="intercept",
                color=(255, 255, 255),
                max_radius=30, lifetime=12, max_lifetime=12,
            ))

            self._total_kills += 1
            if self._renderer:
                self._renderer.kill_count += 1
                self._renderer.log_event(
                    f"INTERCEPT: {target.entity_type.upper()} {target.entity_id} DESTROYED"
                )

            net_value = W_HIT * target_value - W_COST * weapon_cost
            return net_value
        else:
            self._effects.append(VisualEffect(
                x=target.x, y=target.y,
                effect_type="miss",
                color=(255, 100, 30),
                max_radius=12, lifetime=6, max_lifetime=6,
            ))
            return -W_COST * weapon_cost * 0.4


    def _check_collisions(self) -> Dict[str, float]:
        damage_events: Dict[str, float] = {}

        for red in self._red_entities:
            if not red.active:
                continue

            collision_r = (
                COLLISION_RADIUS_MISSILE if red.entity_type == "missile"
                else COLLISION_RADIUS_DRONE
            )

            hit_target = None
            for agent_id, blue in self._blue_entities.items():
                if not blue.active:
                    continue
                if red.distance_to(blue) <= collision_r:
                    hit_target = (agent_id, blue)
                    break

            if hit_target is None and self._radar and self._radar.active:
                if red.distance_to(self._radar) <= collision_r:
                    hit_target = ("radar", self._radar)

            if hit_target is not None:
                tid, target_ent = hit_target
                target_ent.take_damage(red.damage)
                red.active = False

                self._effects.append(VisualEffect(
                    x=red.x, y=red.y,
                    effect_type="explosion",
                    color=(255, 60, 20),
                    radius=8, max_radius=45, lifetime=18, max_lifetime=18,
                ))

                if tid != "radar":
                    damage_events[tid] = damage_events.get(tid, 0.0) + red.damage
                    if self._renderer:
                        self._renderer.log_event(
                            f"HIT: {tid.upper()} took {red.damage} damage from {red.entity_type}"
                        )
                else:
                    for a in self.possible_agents:
                        damage_events[a] = damage_events.get(a, 0.0) + red.damage * 0.5
                    if self._renderer:
                        hp_left = self._radar.hp if self._radar.active else 0
                        self._renderer.log_event(
                            f"RADAR HIT! HP: {hp_left}/{self._radar.max_hp}"
                        )

        return damage_events


    def _tick_red_ai(self):
        for red in self._red_entities:
            if not red.active:
                continue

            if red.entity_type == "drone":
                target = self._pick_drone_target(red)
                if target:
                    red.move_toward(target.x, target.y)

            elif red.entity_type == "missile":
                if self._radar and self._radar.active:
                    red.target_x = self._radar.x
                    red.target_y = self._radar.y
                elif random.random() < MISSILE_RETARGET_PROB:
                    nearest = self._find_nearest_blue(red)
                    if nearest:
                        red.target_x = nearest.x
                        red.target_y = nearest.y
                red.move_toward_target()

    def _pick_drone_target(self, drone: Drone) -> Optional[Entity]:
        if random.random() < DRONE_RADAR_PRIORITY and self._radar and self._radar.active:
            return self._radar
        return self._find_nearest_blue(drone)


    def _check_wave_spawn(self):
        for trigger_step, n_drones, n_missiles in WAVE_SCHEDULE:
            if self.current_step == trigger_step:
                self._spawn_wave(n_drones, n_missiles)
                self._wave_number += 1
                if self._renderer:
                    self._renderer.log_event(
                        f"WAVE {self._wave_number}: {n_drones} DRONES, {n_missiles} BALLISTIC INBOUND"
                    )
                    self._renderer.trigger_wave_warning(self._wave_number)

    def _spawn_wave(self, n_drones: int, n_missiles: int, initial: bool = False):
        spawn = RED_SPAWN_REGION

        for _ in range(n_drones):
            x = np.random.uniform(spawn["x_min"], spawn["x_max"])
            y = np.random.uniform(spawn["y_min"], spawn["y_max"])
            eid = f"drone_{self._red_entity_counter}"
            self._red_entity_counter += 1
            self._red_entities.append(Drone(x=x, y=y, entity_id=eid))

        for _ in range(n_missiles):
            x = np.random.uniform(spawn["x_min"] + 100, spawn["x_max"] - 100)
            y = np.random.uniform(spawn["y_min"], spawn["y_max"] - 30)
            eid = f"missile_{self._red_entity_counter}"
            self._red_entity_counter += 1
            self._red_entities.append(BallisticMissile(
                x=x, y=y,
                target_x=RADAR_POSITION[0],
                target_y=RADAR_POSITION[1],
                entity_id=eid,
            ))


    def _find_nearest_red_in_range(
        self,
        blue: Entity,
        target_type: str,
        max_range: float,
    ) -> Optional[Entity]:
        best = None
        best_dist = max_range

        for red in self._red_entities:
            if not red.active or red.entity_type != target_type:
                continue
            d = blue.distance_to(red)
            if d < best_dist:
                best_dist = d
                best = red

        return best

    def _find_nearest_blue(self, red_entity: Entity) -> Optional[Entity]:
        nearest = None
        min_dist = float("inf")
        for ent in self._blue_entities.values():
            if ent.active:
                d = red_entity.distance_to(ent)
                if d < min_dist:
                    min_dist = d
                    nearest = ent
        if self._radar and self._radar.active:
            d = red_entity.distance_to(self._radar)
            if d < min_dist:
                nearest = self._radar
        return nearest


    def _build_observation(self, agent_id: str) -> np.ndarray:
        obs = np.zeros((GRID_DIM, GRID_DIM, NUM_OBS_CHANNELS), dtype=np.float32)

        for ent in self._blue_entities.values():
            if ent.active:
                gx, gy = self._world_to_grid(ent.x, ent.y)
                obs[gy, gx, 0] = min(1.0, ent.hp / ent.max_hp)
        if self._radar and self._radar.active:
            gx, gy = self._world_to_grid(self._radar.x, self._radar.y)
            obs[gy, gx, 0] = 1.0

        for ent in self._red_entities:
            if ent.active and ent.entity_type == "drone":
                gx, gy = self._world_to_grid(ent.x, ent.y)
                obs[gy, gx, 1] = 1.0

        for ent in self._red_entities:
            if ent.active and ent.entity_type == "missile":
                gx, gy = self._world_to_grid(ent.x, ent.y)
                obs[gy, gx, 2] = 1.0

        if self._radar and self._radar.operational:
            r_gx, r_gy = self._world_to_grid(self._radar.x, self._radar.y)
            r_cells = int(self._radar.detection_range / CELL_SIZE)
            for dy in range(-r_cells, r_cells + 1):
                for dx in range(-r_cells, r_cells + 1):
                    nx, ny = r_gx + dx, r_gy + dy
                    if 0 <= nx < GRID_DIM and 0 <= ny < GRID_DIM:
                        dist = np.sqrt(dx**2 + dy**2)
                        if dist <= r_cells:
                            obs[ny, nx, 3] = max(obs[ny, nx, 3], 1.0 - dist / (r_cells + 1))

        for ent in self._blue_entities.values():
            if ent.active and hasattr(ent, "radar_active") and ent.radar_active:
                radar_range = getattr(ent, "radar_range", getattr(ent, "engagement_range", 0))
                e_gx, e_gy = self._world_to_grid(ent.x, ent.y)
                r_cells = int(radar_range / CELL_SIZE)
                for dy in range(-r_cells, r_cells + 1):
                    for dx in range(-r_cells, r_cells + 1):
                        nx, ny = e_gx + dx, e_gy + dy
                        if 0 <= nx < GRID_DIM and 0 <= ny < GRID_DIM:
                            dist = np.sqrt(dx**2 + dy**2)
                            if dist <= r_cells:
                                obs[ny, nx, 3] = max(obs[ny, nx, 3], 0.7 * (1.0 - dist / (r_cells + 1)))

        for ent in self._red_entities:
            if ent.active:
                e_gx, e_gy = self._world_to_grid(ent.x, ent.y)
                threat_radius = 8
                threat_val = 1.0 if ent.entity_type == "missile" else 0.5
                for dy in range(-threat_radius, threat_radius + 1):
                    for dx in range(-threat_radius, threat_radius + 1):
                        nx, ny = e_gx + dx, e_gy + dy
                        if 0 <= nx < GRID_DIM and 0 <= ny < GRID_DIM:
                            dist = np.sqrt(dx**2 + dy**2)
                            if dist <= threat_radius:
                                heat = threat_val * (1.0 - dist / (threat_radius + 1))
                                obs[ny, nx, 4] = max(obs[ny, nx, 4], heat)

        if agent_id in self._blue_entities:
            ent = self._blue_entities[agent_id]
            if ent.active:
                gx, gy = self._world_to_grid(ent.x, ent.y)
                obs[gy, gx, 5] = ent.hp / ent.max_hp

                total_ammo, max_ammo = 1, 1
                if ent.entity_type == "thaad":
                    total_ammo = ent.ammo_expensive + ent.ammo_cheap
                    max_ammo = 26
                elif ent.entity_type == "aegis":
                    total_ammo = ent.ammo_sm3
                    max_ammo = 8
                elif ent.entity_type == "armor":
                    total_ammo = ent.ammo_airburst
                    max_ammo = 30
                elif ent.entity_type == "bomber":
                    total_ammo = ent.ammo_bombs
                    max_ammo = 4

                ammo_frac = min(1.0, total_ammo / max(1, max_ammo))
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        nx, ny = gx + dx, gy + dy
                        if 0 <= nx < GRID_DIM and 0 <= ny < GRID_DIM:
                            obs[ny, nx, 5] = max(obs[ny, nx, 5], ammo_frac * 0.5)

        return obs

    def _world_to_grid(self, wx: float, wy: float) -> tuple[int, int]:
        gx = int(np.clip(wx / CELL_SIZE, 0, GRID_DIM - 1))
        gy = int(np.clip(wy / CELL_SIZE, 0, GRID_DIM - 1))
        return gx, gy


    def render(self):
        if self.render_mode is None:
            return None

        if self._renderer is None:
            self._renderer = MilitaryRadarRenderer(self)

        env_state = {
            "blue_entities": list(self._blue_entities.values()),
            "red_entities": self._red_entities,
            "radar": self._radar,
            "effects": self._effects,
            "step": self.current_step,
            "max_steps": MAX_STEPS,
            "radar_alive": self._radar.operational if self._radar else False,
            "kills": self._total_kills,
            "score": self._total_score,
            "wave": self._wave_number,
        }

        if self.render_mode == "human":
            return self._renderer.render(env_state)
        elif self.render_mode == "rgb_array":
            self._renderer.render(env_state)
            import pygame
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self._renderer.screen)),
                axes=(1, 0, 2),
            )

    def close(self):
        if self._renderer:
            self._renderer.close()
            self._renderer = None


    def state(self) -> np.ndarray:
        all_obs = [self._build_observation(a) for a in self.possible_agents]
        return np.concatenate(all_obs, axis=-1)
