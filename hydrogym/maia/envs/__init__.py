"""
HydroGym Maia Environment Implementations
==========================================

This package contains specific CFD environment implementations for the Maia solver.
All environments are automatically registered with the factory system.
"""

# Import all environment classes to trigger registration
from .cylinder import Cylinder, RotaryCylinder
from .cavity import Cavity, Cavity3Jet
from .pinball import Pinball, JetPinball
from .naca0012 import NACA0012, NACA0012Gust
from .square_cylinder import SquareCylinder
from .cube import Cube
from .sphere import Sphere

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
]
