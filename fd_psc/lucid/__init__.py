"""Publication-facing API for LUCID continual adaptation.

The implementation remains in :mod:`fd_psc.v2` for checkpoint compatibility;
new integrations should import the stable LUCID names from this module.
"""

from fd_psc.v2.budget_controller import AdaptiveBudgetController
from fd_psc.v2.deep_sleep import DeepSleepController
from fd_psc.v2.state_machine import FSDV2State as LUCIDState
from fd_psc.v2.state_machine import FSDV2StateMachine as LUCIDStateMachine
from fd_psc.v2.system import FSDV2IntegrationError as LUCIDIntegrationError
from fd_psc.v2.system import FSDV2System as LUCIDSystem

__all__ = [
    "AdaptiveBudgetController",
    "DeepSleepController",
    "LUCIDIntegrationError",
    "LUCIDState",
    "LUCIDStateMachine",
    "LUCIDSystem",
]
