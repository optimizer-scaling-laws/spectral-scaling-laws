"""Compatibility namespace for lightweight scaling helpers.

Paper-facing fitting utilities live under :mod:`optimizer_ssl.analysis`.
"""

from optimizer_ssl.scaling.fit_power_law import fit_power_law

__all__ = ["fit_power_law"]
