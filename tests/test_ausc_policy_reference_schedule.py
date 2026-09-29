"""A missing historical mode field is accepted only for its pinned revision."""
import unittest
from sim.ausc_policy.reference_schedule import zero_reference_contract


class ReferenceSchemaTests(unittest.TestCase):
    def test_pre_hint_revision_is_explicitly_bound(self):
        owner=dict(code_commit='7b5802f9724ed8b7702e5300ff8db6d339681caa')
        self.assertTrue(zero_reference_contract({},owner))
        self.assertFalse(zero_reference_contract({},dict(code_commit='unknown')))
        self.assertFalse(zero_reference_contract(dict(motion_velocity='quadratic5'),owner))
        self.assertFalse(zero_reference_contract(dict(motion_velocity=None),owner))
        self.assertTrue(zero_reference_contract(dict(motion_velocity='zero'),{}))


if __name__=='__main__':unittest.main()
