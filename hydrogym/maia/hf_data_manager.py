"""
Backwards-compatibility shim.

``HFDataManager`` and ``SOLVER_PROFILES`` have moved to :mod:`hydrogym.data_manager`.
This module re-exports them so that existing imports of the form::

    from hydrogym.maia.hf_data_manager import HFDataManager
    from hydrogym.maia import HFDataManager

continue to work without modification.
"""

from hydrogym.data_manager import HFDataManager, SOLVER_PROFILES  # noqa: F401

__all__ = ['HFDataManager', 'SOLVER_PROFILES']
