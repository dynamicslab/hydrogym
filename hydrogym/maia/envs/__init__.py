"""
HydroGym Maia Environment Implementations
==========================================

This package contains specific CFD environment implementations for the Maia solver.
All environments are automatically registered with the factory system.
"""

from .cavity import Cavity, Cavity3Jet
from .cube import Cube

# Import all environment classes to trigger registration
from .cylinder import Cylinder, RotaryCylinder
from .dra2303 import DRA2303Base, DRA2303Jet, DRA2303SurfaceWave
from .naca0012 import NACA0012, NACA0012Gust
from .pinball import JetPinball, Pinball
from .sphere import Sphere
from .square_cylinder import SquareCylinder
from .turbulent_boundary_layer import ZPGTBLBase, ZPGTBLJet, ZPGTBLSurfaceWave

__all__ = [
    "Cylinder",
    "RotaryCylinder",
    "Cavity",
    "Cavity3Jet",
    "Pinball",
    "JetPinball",
    "NACA0012",
    "NACA0012Gust",
    "SquareCylinder",
    "Cube",
    "Sphere",
    "ZPGTBLBase",
    "ZPGTBLJet",
    "ZPGTBLSurfaceWave",
    "DRA2303Base",
    "DRA2303Jet",
    "DRA2303SurfaceWave",
]
