from types import SimpleNamespace

from sim.ausc_policy.seed_probe import invalidate_loaded_cache


class CacheProbe:
    def __init__(self):
        self.resets = 0

    def modulation_cache_state(self, expert):
        return {'populated': expert.populated, 'entries': 10 if expert.populated else 0}

    def reset_modulation_cache(self, expert):
        assert expert.populated, 'Legacy reset requires an existing eval cache'
        self.resets += 1
        expert.populated = False
        return {'reset': True}


def test_fresh_checkpoint_has_no_stale_cache():
    probe, expert = CacheProbe(), SimpleNamespace(populated=False)
    assert invalidate_loaded_cache(probe, expert)['first_evaluation']
    assert probe.resets == 0


def test_existing_cache_is_invalidated_after_parameter_load():
    probe, expert = CacheProbe(), SimpleNamespace(populated=True)
    assert invalidate_loaded_cache(probe, expert)['reset']
    assert probe.resets == 1 and not expert.populated
