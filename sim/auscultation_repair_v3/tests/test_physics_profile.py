import pytest
from sim.auscultation_repair_v3.physics_profile import PhysicsProfile


@pytest.mark.parametrize('options',[dict(position_iterations=0),dict(position_iterations=True),
    dict(velocity_iterations=-1),dict(armature_kg_m2=float('nan')),dict(armature_kg_m2=.03)])
def test_invalid_candidate_parameters_cannot_reach_isaac(options):
    with pytest.raises(ValueError):PhysicsProfile(**options).validate()
