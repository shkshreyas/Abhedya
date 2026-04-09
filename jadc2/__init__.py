"""
JADC2 Multi-Agent Tactical Defense Simulation
=============================================
A PettingZoo-based MARL environment simulating multi-domain
battlefield defense using the MAPPO algorithm.
"""

from jadc2.env import JADC2_Env, env as make_env

__version__ = "0.1.0"
__all__ = ["JADC2_Env", "make_env"]
