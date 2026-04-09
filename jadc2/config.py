"""
JADC2 Configuration — All tunable constants for the simulation.
================================================================
Edit this file to adjust game balance, visual appearance, and
training hyperparameters without touching environment logic.
"""

import numpy as np

# ─────────────────────────── WORLD GEOMETRY ───────────────────────────
WORLD_SIZE = 1000.0        # Continuous coordinate range [0, WORLD_SIZE]
GRID_DIM = 64              # Observation grid: GRID_DIM × GRID_DIM
CELL_SIZE = WORLD_SIZE / GRID_DIM  # ≈ 15.6 units per cell

# ─────────────────────────── EPISODE ──────────────────────────────────
MAX_STEPS = 500            # Timesteps per episode

# ─────────────────────────── BLUE TEAM COMPOSITION ────────────────────
NUM_THAAD = 2              # THAAD / Patriot air defense batteries
NUM_AEGIS = 2              # Aegis destroyers (Navy)
NUM_ARMOR = 2              # Armored columns / Tanks (Army)
NUM_BOMBER = 1             # Stealth bombers (Air Force)
NUM_RADAR = 1              # Central early-warning radar station

TOTAL_BLUE = NUM_THAAD + NUM_AEGIS + NUM_ARMOR + NUM_BOMBER

# ─────────────────────────── ENTITY PROPERTIES ────────────────────────
# Air Defense (THAAD/Patriot) — stationary
THAAD_HP = 5
THAAD_AMMO_EXPENSIVE = 6   # SM-3 class interceptors
THAAD_AMMO_CHEAP = 20      # PAC-3 class interceptors
THAAD_RANGE = 250.0        # Engagement radius
THAAD_COOLDOWN = 2         # Steps between shots

# Aegis Destroyer — mobile
AEGIS_HP = 8
AEGIS_AMMO_SM3 = 8
AEGIS_SPEED = 2.0
AEGIS_RANGE = 300.0
AEGIS_RADAR_RANGE = 350.0
AEGIS_COOLDOWN = 2

# Armored Column — mobile
ARMOR_HP = 10
ARMOR_AMMO_AIRBURST = 30
ARMOR_SPEED = 1.5
ARMOR_RANGE = 80.0         # Short range anti-drone
ARMOR_COOLDOWN = 1

# Stealth Bomber — mobile
BOMBER_HP = 3
BOMBER_AMMO_BOMBS = 4
BOMBER_SPEED = 4.0
BOMBER_RANGE = 60.0        # Bomb drop radius
BOMBER_STEALTH = 0.3       # Detection reduction factor
BOMBER_COOLDOWN = 3

# Central Radar
RADAR_HP = 5
RADAR_DETECTION_RANGE = 500.0  # Huge detection range

# ─────────────────────────── RED TEAM (Phase 3) ───────────────────────
DRONE_HP = 1
DRONE_SPEED = 3.0
DRONE_DAMAGE = 1
DRONE_COST = 0.5           # Low-cost decoy

MISSILE_HP = 3
MISSILE_SPEED = 6.0
MISSILE_DAMAGE = 5
MISSILE_COST = 8.0          # High-value threat

# ─────────────────────────── WEAPON COSTS (for reward) ────────────────
COST_THAAD_EXPENSIVE = 10.0   # Firing SM-3 at a drone = terrible trade
COST_THAAD_CHEAP = 2.0
COST_AEGIS_SM3 = 8.0
COST_ARMOR_AIRBURST = 1.0
COST_BOMBER_BOMB = 5.0

# ─────────────────────────── REWARD WEIGHTS ───────────────────────────
W_HIT = 10.0               # Successful interception of a target
W_COST = 1.0               # Weapon cost penalty
W_RADAR = 5.0              # Continuous reward for radar being alive
W_DMG = 8.0                # Damage sustained penalty

# ─────────────────────────── OBSERVATION CHANNELS ─────────────────────
NUM_OBS_CHANNELS = 6
# Channel 0: Friendly positions (Blue Team)
# Channel 1: Enemy drone positions
# Channel 2: Enemy missile positions
# Channel 3: Radar coverage map
# Channel 4: Threat heatmap
# Channel 5: Self position + status (ammo/hp encoded as intensity)

# ─────────────────────────── RENDERING — DARK MILITARY HUD ───────────
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 900
BATTLEFIELD_SIZE = 750      # Square battlefield viewport in pixels
HUD_WIDTH = SCREEN_WIDTH - BATTLEFIELD_SIZE  # Right panel width
FPS = 30

# ── Color Palette (RGB) — Tactical dark theme ──
class Colors:
    # Backgrounds
    BG_DEEP         = (10, 14, 20)       # #0A0E14 — main background
    BG_PANEL        = (15, 22, 35)       # #0F1623 — HUD panel bg
    BG_GRID         = (26, 35, 50)       # #1A2332 — grid lines

    # Tactical greens (radar / friendly)
    RADAR_GREEN     = (0, 255, 65)       # #00FF41 — classic radar
    RADAR_DIM       = (0, 100, 30)       # Dimmed sweep trail
    RADAR_GLOW      = (0, 200, 50, 80)   # Translucent glow

    # Friendly (Blue Team)
    FRIENDLY_CYAN   = (0, 170, 255)      # #00AAFF
    FRIENDLY_DIM    = (0, 80, 140)       # Dimmed friendly
    THAAD_COLOR     = (100, 200, 255)    # Light blue
    AEGIS_COLOR     = (0, 180, 220)      # Deep cyan
    ARMOR_COLOR     = (60, 200, 160)     # Teal green
    BOMBER_COLOR    = (180, 140, 255)    # Purple

    # Threats (Red Team)
    THREAT_RED      = (255, 51, 51)      # #FF3333
    THREAT_PULSE    = (255, 100, 100)    # Pulsing highlight
    MISSILE_AMBER   = (255, 170, 0)      # #FFAA00
    MISSILE_GLOW    = (255, 200, 50, 100)

    # HUD elements
    HUD_TEXT        = (180, 200, 220)    # Light gray-blue
    HUD_BRIGHT      = (220, 240, 255)    # White-blue highlights
    HUD_ACCENT      = (0, 255, 160)      # Neon teal accent
    HUD_WARNING     = (255, 200, 0)      # Amber warning
    HUD_CRITICAL    = (255, 60, 60)      # Red critical
    HUD_BORDER      = (40, 60, 80)       # Panel borders

    # Effects
    INTERCEPT_FLASH = (255, 255, 255)    # White flash
    EXPLOSION       = (255, 180, 50)     # Orange explosion
    SCANLINE        = (255, 255, 255, 8) # Very subtle scanline

    # Grid overlay
    GRID_LINE       = (20, 35, 55)       # Subtle grid
    GRID_MAJOR      = (30, 50, 75)       # Major grid lines (every 8 cells)

# ─────────────────────────── SPAWN ZONES ──────────────────────────────
# Blue Team spawns in the southern half, Red attacks from the north
BLUE_SPAWN_REGION = {
    "x_min": 100.0, "x_max": 900.0,
    "y_min": 500.0, "y_max": 900.0,
}
RED_SPAWN_REGION = {
    "x_min": 100.0, "x_max": 900.0,
    "y_min": 0.0,   "y_max": 150.0,
}
RADAR_POSITION = (500.0, 700.0)  # Central rear position

# ─────────────────────────── ACTION SPACE SIZES ───────────────────────
# THAAD: Discrete(4) → 0=noop, 1=fire_expensive, 2=fire_cheap, 3=toggle_radar
THAAD_NUM_ACTIONS = 4

# Aegis: MultiDiscrete([5, 9])
# Movement: 0=noop, 1=N, 2=S, 3=E, 4=W
# Combat: 0=noop, 1-4=fire_SM3_at_target, 5-8=share_telemetry
AEGIS_MOVE_ACTIONS = 5
AEGIS_COMBAT_ACTIONS = 9

# Armor: MultiDiscrete([5, 3])
# Movement: 0=noop, 1=N, 2=S, 3=E, 4=W
# Combat: 0=noop, 1=fire_airburst, 2=secure_position
ARMOR_MOVE_ACTIONS = 5
ARMOR_COMBAT_ACTIONS = 3

# Bomber: MultiDiscrete([9, 2])
# Movement: 0=noop, 1-8=8 compass directions
# Combat: 0=noop, 1=drop_bomb
BOMBER_MOVE_ACTIONS = 9
BOMBER_COMBAT_ACTIONS = 2

# Movement direction vectors (dx, dy) for 4-dir and 8-dir
MOVE_4DIR = {
    0: (0, 0),    # noop
    1: (0, -1),   # North (up)
    2: (0, 1),    # South (down)
    3: (1, 0),    # East
    4: (-1, 0),   # West
}

MOVE_8DIR = {
    0: (0, 0),     # noop
    1: (0, -1),    # N
    2: (0, 1),     # S
    3: (1, 0),     # E
    4: (-1, 0),    # W
    5: (1, -1),    # NE
    6: (-1, -1),   # NW
    7: (1, 1),     # SE
    8: (-1, 1),    # SW
}

# ─────────────────────────── PHASE 2 — INTERCEPT PROBABILITIES ────────
HIT_PROB_SM3_MISSILE  = 0.88    # THAAD SM-3 vs ballistic missile
HIT_PROB_SM3_DRONE    = 0.62    # THAAD SM-3 vs drone (inefficient)
HIT_PROB_PAC3_DRONE   = 0.82    # THAAD PAC-3 vs drone
HIT_PROB_PAC3_MISSILE = 0.52    # THAAD PAC-3 vs missile (marginal)
HIT_PROB_AEGIS_SM3    = 0.85    # Aegis SM-3 hit probability
HIT_PROB_AIRBURST     = 0.78    # Armor airburst per drone in AoE
HIT_PROB_BOMB         = 0.88    # Bomber bomb per entity in radius

# Collision radii (when Red entity physically reaches a Blue asset)
COLLISION_RADIUS_DRONE   = 12.0
COLLISION_RADIUS_MISSILE = 22.0

# ─────────────────────────── PHASE 3 — RED TEAM AI ────────────────────
DRONE_RADAR_PRIORITY = 0.45     # Probability drone retargets radar
MISSILE_RETARGET_PROB = 0.15    # Probability missile re-targets each step

# Wave schedule: (trigger_step, num_drones, num_missiles)
WAVE_SCHEDULE = [
    (60,  3, 0),
    (120, 5, 1),
    (180, 4, 0),
    (240, 6, 1),
    (300, 5, 2),
    (360, 8, 1),
    (400, 9, 2),
    (450, 6, 3),
]
