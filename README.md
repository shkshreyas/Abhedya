# 🛡️ Project Abhedya: JADC2 Multi-Agent Tactical Defense Simulation

<div align="center">

**Project Abhedya (अभेद्य) — "The Impenetrable Shield"**

**Joint All-Domain Command & Control (JADC2) Multi-Agent Reinforcement Learning Simulation**

[Repository: shkshreyas/Abhedya](https://github.com/shkshreyas/Abhedya)

*A PettingZoo-based MARL environment for training AI agents to solve the Cost-Exchange Crisis in modern anti-access/area denial (A2/AD) defense scenarios.*

[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![PettingZoo](https://img.shields.io/badge/PettingZoo-1.24+-green.svg)](https://pettingzoo.farama.org)
[![Ray RLlib](https://img.shields.io/badge/Ray_RLlib-2.9+-orange.svg)](https://docs.ray.io/en/latest/rllib/)
[![PyTorch CPU](https://img.shields.io/badge/PyTorch-CPU_Only-red.svg)](https://pytorch.org)

</div>

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [The Problem: Cost-Exchange Crisis](#-the-problem-cost-exchange-crisis)
- [Architecture & Design](#-architecture--design)
- [Environment Details](#-environment-details)
- [Entity Specifications](#-entity-specifications)
- [Observation Space](#-observation-space)
- [Action Spaces](#-action-spaces)
- [Reward Function](#-reward-function)
- [Rendering: Military Radar HUD](#-rendering-military-radar-hud)
- [Installation & Setup](#-installation--setup)
- [Quick Start](#-quick-start)
- [Project Structure](#-project-structure)
- [Development Phases](#-development-phases)
- [Configuration](#-configuration)
- [Technical Notes](#-technical-notes)

---

## 🎯 Project Overview

This project implements a **Multi-Agent Reinforcement Learning (MARL)** simulation where AI-controlled **Blue Team** assets must defend against an asymmetric **Red Team** attack consisting of low-cost drone swarms, decoys, and precision ballistic missiles.

The simulation is modeled on real-world Joint All-Domain Command & Control (JADC2) concepts, where multiple military branches (Army, Navy, Air Force) must coordinate across domains to achieve effective defense.

### Key Objectives

| Objective | Description |
|-----------|-------------|
| **Cost-Exchange Optimization** | Train AI to avoid wasting expensive interceptors (SM-3, $10M+) on cheap decoy drones ($500) |
| **Sensor Preservation** | Protect early-warning radar from targeted anti-radiation missiles |
| **Cross-Domain Coordination** | Enable Navy (Aegis), Army (Armor), and Air Force (Bombers) to share data and coordinate |
| **Ammo Conservation** | Learn when to fire and when to hold, balancing aggression vs. sustainability |

### Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Environment | **PettingZoo** (Parallel API) | Multi-agent environment framework |
| RL Algorithm | **Ray RLlib** (MAPPO) | Multi-Agent Proximal Policy Optimization |
| Neural Network | **PyTorch** (CPU) | CNN policy for spatial observations |
| Rendering | **Pygame** | Military-grade tactical HUD |
| Observations | **NumPy** | Multi-channel grid rasterization |

---

## 💥 The Problem: Cost-Exchange Crisis

In modern A2/AD warfare, adversaries exploit an asymmetric cost advantage:

```
Attacker launches:    100 × cheap drones         ($50,000 total)
                     +  5 × ballistic missiles   ($40,000,000 total)

Defender must spend:  100 × SM-3 interceptors    ($1,200,000,000)
                     ... to stop a $40M attack
```

**The defender loses economically even when winning tactically.**

This simulation trains AI to solve this by learning:
1. **Discriminate threats** — Don't fire SM-3s at decoy drones
2. **Use cheap interceptors** — PAC-3 and airburst rounds for drones
3. **Protect sensors** — Keep radar alive to maintain battlefield awareness
4. **Coordinate fires** — Share telemetry so the right weapon engages the right target

---

## 🏗️ Architecture & Design

### System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    JADC2_Env (ParallelEnv)               │
│                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │   Blue Team   │  │   Red Team   │  │   Renderer   │  │
│  │   (7 Agents)  │  │  (Scripted)  │  │ (Pygame HUD) │  │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  │
│         │                  │                  │          │
│  ┌──────▼──────────────────▼──────────────────▼───────┐  │
│  │              World State (1000×1000 grid)           │  │
│  │     Entities, Positions, HP, Ammo, Radar Status     │  │
│  └──────┬─────────────────────────────────────────────┘  │
│         │                                                │
│  ┌──────▼─────────────────────────────────────────────┐  │
│  │           Observation Builder (6-Channel CNN)       │  │
│  │   [Friendly|Drones|Missiles|Radar|Threat|Self]     │  │
│  └────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
                          │
                    PettingZoo API
                          │
               ┌──────────▼──────────┐
               │   Ray RLlib MAPPO    │
               │   (Phase 4)          │
               │   Centralized Critic │
               │   Decentralized Exec │
               └─────────────────────┘
```

### Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| API | **Parallel** (not AEC) | All Blue Team agents act simultaneously; matches RLlib's `ParallelPettingZooEnv` |
| Observations | **Multi-channel 64×64 grid** | CNN-friendly spatial representation (like YOLO's grid cells) |
| Actions | **Heterogeneous MultiDiscrete** | Each agent type has unique capabilities; MultiDiscrete decouples movement from combat |
| Coordinate system | **Continuous** (1000×1000) | Realistic; discretized to 64×64 only for observations |
| Training | **CPU-only** | Optimized for laptop without GPU; batch sizes and model complexity adjusted |

---

## 🌍 Environment Details

### World

| Parameter | Value |
|-----------|-------|
| World size | 1000 × 1000 continuous units |
| Grid resolution | 64 × 64 cells (≈15.6 units/cell) |
| Max episode length | 500 timesteps |
| Blue spawn zone | Southern half (y: 500–900) |
| Red spawn zone | Northern edge (y: 0–150) |
| Radar position | Center-rear (500, 700) |

### Episode Flow

```
Reset → Spawn Blue Team (7 agents) + Radar
      → Spawn initial Red threats (5 drones + 1 missile)
      → Loop for 500 steps:
          ├─ All Blue agents choose actions simultaneously
          ├─ Red Team moves toward targets (scripted)
          ├─ Collision detection + interception checks
          ├─ Reward calculation (joint objective)
          ├─ New Red waves spawn periodically
          └─ Render (if human mode)
      → Episode ends when: max steps reached OR all Blue destroyed
```

---

## 🎖️ Entity Specifications

### Blue Team (Learning Agents)

| Entity | Type | HP | Speed | Ammo | Range | Role |
|--------|------|-----|-------|------|-------|------|
| **THAAD Battery** | Air Defense | 5 | Static | 6 SM-3 + 20 PAC-3 | 250 | High-altitude missile defense |
| **Aegis Destroyer** | Navy | 8 | 2.0 | 8 SM-3 | 300 | Mobile naval interceptor + radar |
| **Armored Column** | Army | 10 | 1.5 | 30 Airburst | 80 | Close-range drone defense |
| **Stealth Bomber** | Air Force | 3 | 4.0 | 4 Bombs | 60 | Offensive strike on launch sites |
| **Radar Station** | Sensor | 5 | Static | — | 500 | Early-warning detection |

### Red Team (Scripted Adversaries)

| Entity | HP | Speed | Damage | Cost | Behavior |
|--------|-----|-------|--------|------|----------|
| **Drone** | 1 | 3.0 | 1 | $0.5 | Swarm toward nearest Blue unit |
| **Ballistic Missile** | 3 | 6.0 | 5 | $8.0 | Precision strike radar station |

---

## 👁️ Observation Space

Each agent receives a **6-channel 64×64 grid image** (shape: `(64, 64, 6)`, values in `[0, 1]`):

```
Channel 0  ████████  Friendly unit positions (HP as intensity)
Channel 1  ████████  Detected enemy drones
Channel 2  ████████  Detected enemy missiles
Channel 3  ████████  Radar coverage map (distance-weighted)
Channel 4  ████████  Threat heatmap (proximity to enemies)
Channel 5  ████████  Self position + status (ammo/HP encoded)
```

This structure is designed to be fed directly into a **Convolutional Neural Network (CNN)**, similar to how object detection models (YOLO, SSD) process multi-channel spatial inputs.

---

## 🎮 Action Spaces

Actions are **heterogeneous** — each agent type has a different action space tailored to its capabilities:

### THAAD / Patriot Battery — `Discrete(4)`

| Action | Effect | Weapon Cost |
|--------|--------|-------------|
| 0 | No-op (hold fire) | — |
| 1 | Fire expensive interceptor (SM-3) | $10.0 |
| 2 | Fire cheap interceptor (PAC-3) | $2.0 |
| 3 | Toggle radar (active ↔ passive) | — |

### Aegis Destroyer — `MultiDiscrete([5, 9])`

| Movement (0-4) | Combat (0-8) |
|----------------|--------------|
| 0: Hold position | 0: Hold fire |
| 1: North | 1-4: Fire SM-3 at target 1–4 |
| 2: South | 5-8: Share telemetry with ally 1–4 |
| 3: East | — |
| 4: West | — |

### Armored Column — `MultiDiscrete([5, 3])`

| Movement (0-4) | Combat (0-2) |
|----------------|--------------|
| 0: Hold | 0: Hold fire |
| 1-4: N/S/E/W | 1: Fire airburst round |
| — | 2: Secure position (defensive bonus) |

### Stealth Bomber — `MultiDiscrete([9, 2])`

| Movement (0-8) | Combat (0-1) |
|----------------|--------------|
| 0: Hold | 0: Hold fire |
| 1-8: 8 compass directions | 1: Drop bomb on launch site |

---

## 📊 Reward Function

The environment uses a **joint team reward** to encourage cooperation. The reward at timestep $t$ is:

$$R_t = w_{hit} \sum_{k=1}^{K} H_{k,t} - w_{cost} \sum_{j=1}^{M} C_{j,t} + w_{radar} \cdot S_t - w_{dmg} \cdot D_t$$

### Variables

| Symbol | Meaning | Default Weight |
|--------|---------|----------------|
| $H_{k,t}$ | Successful interception of target $k$ | $w_{hit} = 10.0$ |
| $C_{j,t}$ | Cost of weapon $j$ fired (penalizes expensive-on-cheap) | $w_{cost} = 1.0$ |
| $S_t$ | Binary radar operational status (1=alive, 0=destroyed) | $w_{radar} = 5.0$ |
| $D_t$ | Damage sustained by Blue Team infrastructure | $w_{dmg} = 8.0$ |

### Cost-Exchange Penalty Examples

| Scenario | Reward Component |
|----------|-----------------|
| SM-3 ($10) intercepts ballistic missile ($8) | +10.0 (hit) − 10.0 (cost) = **0.0** |
| SM-3 ($10) intercepts drone ($0.5) | +10.0 (hit) − 10.0 (cost) = **0.0** (wasteful!) |
| PAC-3 ($2) intercepts drone ($0.5) | +10.0 (hit) − 2.0 (cost) = **+8.0** (smart!) |
| Airburst ($1) intercepts drone ($0.5) | +10.0 (hit) − 1.0 (cost) = **+9.0** (optimal!) |

The AI should learn that **cheap weapons on cheap targets** yields the best reward.

---

## 🖥️ Rendering: Military Radar HUD

The simulation features a premium **dark-themed tactical radar display** built with Pygame:

### HUD Layout

```
┌─────────────────────────────┬─────────────────┐
│                             │  ▌ FORCE STATUS  │
│  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓  │  ● THAAD_0  5/5 │
│  ▓  BATTLEFIELD VIEW     ▓  │  ● AEGIS_0  8/8 │
│  ▓                       ▓  │  ■ ARMOR_0 10/10 │
│  ▓    ◆ THAAD   ▽ AEGIS ▓  │  ▲ BOMBER_0 3/3 │
│  ▓    ■ ARMOR   ▲ BOMBER▓  │  ◉ RADAR-1  5/5 │
│  ▓    • Drones  ◇ Missile▓ │──────────────────│
│  ▓    ◉ Radar   ~ Sweep  ▓ │  ▌ THREAT BOARD  │
│  ▓                       ▓  │  DRONES:     5   │
│  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓  │  BALLISTIC:  1   │
│                             │  THREAT: ████ 30% │
│  ◆ ABHEDYA TACTICAL CMD    │──────────────────│
│                 T+0042/500  │  ▌ AMMUNITION    │
│  ● RADAR ONLINE            │  THAAD_0 SM3:6   │
│                             │  AEGIS_0 SM3:8   │
│                             │──────────────────│
│                             │  ▌ EVENT LOG     │
│                             │  [T+0001] INIT   │
└─────────────────────────────┴─────────────────┘
```

### Visual Features

| Feature | Implementation |
|---------|---------------|
| **Radar Sweep** | Rotating phosphor-green line with fading trail (40-frame persistence) |
| **CRT Scanlines** | Horizontal semi-transparent lines every 3px for retro military feel |
| **Entity Glow** | Colored translucent circles behind each unit icon |
| **Pulsing Threats** | Drone dots pulse using sine-wave intensity modulation |
| **Blinking Missiles** | Ballistic missile diamonds alternate amber/red with "BALLISTIC" label |
| **Range Rings** | Anti-aliased circles showing engagement/detection ranges |
| **HP Bars** | Color-coded (green → amber → red) below each entity |
| **Vignette** | Dark corner overlays for cinematic framing |
| **HUD Panels** | Force status, threat board, ammunition, event log with neon accents |

---

## ⚙️ Installation & Setup

### Prerequisites

- **Python 3.11+** (Anaconda or standard CPython)
- **No GPU required** — fully CPU-optimized

### Install

```bash
# Clone the repository
git clone https://github.com/shkshreyas/Abhedya.git
cd Abhedya

# Install dependencies (CPU-only PyTorch)
pip install -r requirements.txt
```

### Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `pettingzoo` | ≥1.24.0 | Multi-agent environment API |
| `gymnasium` | ≥0.29.0 | Stable RL environment interface |
| `pygame` | ≥2.5.0 | Tactical HUD rendering |
| `numpy` | ≥1.24.0 | Array operations & rasterization |
| `ray[rllib]` | ≥2.9.0 | MAPPO training framework |
| `torch` | ≥2.0.0 (CPU) | Neural network backend |
| `supersuit` | ≥3.9.0 | Environment preprocessing wrappers |

---

## 🚀 Quick Start

### Run the Visual Demo

```bash
python demo.py
```

This will:
1. Create the JADC2 environment
2. Spawn Blue Team assets and initial Red threats
3. Run 300 steps with random actions
4. Display the military radar HUD in real-time

**Controls:**
- `ESC` or close window to exit

### Use as a PettingZoo Environment

```python
from jadc2.env import JADC2_Env

# Create environment
env = JADC2_Env(render_mode="human")  # or None for headless

# Reset
observations, infos = env.reset(seed=42)

# Game loop
while env.agents:
    # Your policy here — sample random actions for now
    actions = {
        agent: env.action_space(agent).sample()
        for agent in env.agents
    }
    
    observations, rewards, terminations, truncations, infos = env.step(actions)
    env.render()

env.close()
```

### Run PettingZoo API Validation

```bash
python -c "
from pettingzoo.test import parallel_api_test
from jadc2.env import JADC2_Env
env = JADC2_Env()
parallel_api_test(env, num_cycles=50)
print('All API tests passed!')
"
```

---

## 📁 Project Structure

```
RL Project/
│
├── 📄 requirements.txt        # Python dependencies (CPU-only)
├── 📄 demo.py                  # Visual demo script
├── 📄 train.py                 # (Phase 4) RLlib MAPPO training
├── 📄 README.md                # This file
│
└── 📦 jadc2/                   # Main package
    ├── 📄 __init__.py           # Package init & exports
    ├── 📄 config.py             # All tunable constants
    │                            #   → World geometry, entity stats
    │                            #   → Reward weights, color palette
    │                            #   → Action space sizes, spawn zones
    ├── 📄 entities.py           # Entity dataclass hierarchy
    │                            #   → Blue: THAAD, Aegis, Armor, Bomber, Radar
    │                            #   → Red: Drone, BallisticMissile
    │                            #   → Visual effects (explosions)
    ├── 📄 env.py                # JADC2_Env (PettingZoo ParallelEnv)
    │                            #   → Spaces, reset(), step(), observations
    │                            #   → Multi-channel grid rasterization
    │                            #   → Global state for centralized critic
    └── 📄 renderer.py           # MilitaryRadarRenderer (pygame)
                                 #   → Dark HUD, radar sweep, scanlines
                                 #   → Entity icons, threat pulsing
                                 #   → HUD panels, event log
```

---

## 📅 Development Phases

| Phase | Status | Description |
|-------|--------|-------------|
| **Phase 1** | ✅ Complete | Environment structure, entity classes, observation/action spaces, renderer, `reset()` |
| **Phase 2** | ✅ Complete | `step()` logic — collision detection, intercept mechanics, full reward function |
| **Phase 3** | ✅ Complete | Red Team AI — dynamic spawning heuristics, wave escalation, radar-seeking behavior |
| **Phase 4** | ✅ Complete | Ray RLlib MAPPO training script, CNN policy, centralized critic, training pipeline |

### Phase 4 Deliverables (Completed)

- [x] `JADC2_Env` class inheriting from `ParallelEnv`
- [x] 7 heterogeneous Blue Team agents with correct spaces
- [x] Entity dataclasses with action methods (fire, move, toggle)
- [x] 6-channel 64×64 CNN observation builder
- [x] Multi-channel grid rasterization (friendlies, enemies, radar, threat heatmap, self)
- [x] Basic movement processing for all agent types
- [x] Red Team entities with movement-toward-target
- [x] Military radar HUD with sweep, scanlines, HUD panels
- [x] PettingZoo Parallel API test: **PASSED** ✅
- [x] Full configuration module for easy tuning
- [x] Complete reward function logic and intercept mechanics
- [x] Scripted Adversary behaviors and automated event evaluation
- [x] Centralized Ray RLlib MAPPO training capability
- [x] Kaggle GPU-ready Training workflows
- [x] PettingZoo Parallel API test: **PASSED** ✅
- [x] Full configuration module for easy tuning

---

## 🔧 Configuration

All game parameters are centralized in [`jadc2/config.py`](jadc2/config.py). Key tunable groups:

### World & Episode
```python
WORLD_SIZE = 1000.0     # Battlefield dimensions
GRID_DIM = 64           # Observation grid resolution  
MAX_STEPS = 500         # Episode length
```

### Entity Balance
```python
THAAD_HP = 5            # Air defense durability
AEGIS_SPEED = 2.0       # Naval movement speed
BOMBER_STEALTH = 0.3    # Detection reduction factor
RADAR_DETECTION_RANGE = 500.0  # Early-warning radius
```

### Reward Weights
```python
W_HIT = 10.0            # Interception reward
W_COST = 1.0            # Weapon cost penalty multiplier
W_RADAR = 5.0           # Radar alive bonus
W_DMG = 8.0             # Damage taken penalty
```

### Weapon Costs (Cost-Exchange Tuning)
```python
COST_THAAD_EXPENSIVE = 10.0   # SM-3 — should NOT be used on drones
COST_THAAD_CHEAP = 2.0        # PAC-3 — acceptable for drones
COST_ARMOR_AIRBURST = 1.0     # Best cost-exchange for drone defense
```

---

## 📝 Technical Notes

### CPU Optimization

This project is designed to run on laptops without GPUs:

- **PyTorch CPU**: Installed from `https://download.pytorch.org/whl/cpu` — no CUDA overhead
- **Small observation grid**: 64×64×6 = 24,576 values per obs — lightweight for CNN inference
- **Efficient rendering**: Static surfaces (grid, scanlines, HUD background) are pre-rendered and cached
- **Ray RLlib**: Training batch sizes will be adjusted for CPU throughput in Phase 4

### PettingZoo Compatibility

- Implements the **Parallel API** (`ParallelEnv`)
- Passes `parallel_api_test` with 50+ cycles
- Compatible with `pettingzoo.utils.conversions.parallel_to_aec` if AEC is needed
- Works with RLlib's `ParallelPettingZooEnv` wrapper

### MAPPO Architecture (Phase 4)

The Multi-Agent PPO training setup will use:
- **Centralized critic**: `state()` method provides full global state
- **Decentralized actors**: Each agent uses only its own 64×64×6 observation
- **Shared parameters**: Agents of the same type share policy weights
- **4 policy groups**: THAAD, Aegis, Armor, Bomber

---

<div align="center">

**Built with 🧠 Reinforcement Learning + 🎯 Tactical AI**

*All Phases Complete — Ready for GPU Deployment*

</div>
