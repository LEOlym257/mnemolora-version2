"""FD-PSC continual-memory components for AdaJEPA.

The package deliberately has no import-time side effects.  In particular,
adapter injection only happens after :class:`FDPSCConfig` has been validated
and an enabled :class:`FDPSCSystem` is constructed.
"""

from .config import FDPSCConfig, FDPSCConfigError

__all__ = ["FDPSCConfig", "FDPSCConfigError"]
