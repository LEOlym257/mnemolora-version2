"""Publication-facing LUCID aliases remain compatible with schema-2 internals."""

import unittest

from fd_psc.lucid import (
    LUCIDIntegrationError,
    LUCIDState,
    LUCIDStateMachine,
    LUCIDSystem,
)
from fd_psc.v2.state_machine import FSDV2State, FSDV2StateMachine
from fd_psc.v2.system import FSDV2IntegrationError, FSDV2System


class LUCIDPublicAPITests(unittest.TestCase):
    def test_public_aliases_preserve_checkpoint_compatible_implementations(self):
        self.assertIs(LUCIDSystem, FSDV2System)
        self.assertIs(LUCIDIntegrationError, FSDV2IntegrationError)
        self.assertIs(LUCIDState, FSDV2State)
        self.assertIs(LUCIDStateMachine, FSDV2StateMachine)


if __name__ == "__main__":
    unittest.main()
