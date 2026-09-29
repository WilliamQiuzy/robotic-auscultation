"""Explicit versions of actual recorded origins; no arbitrary replay selection."""


def selected_origin(config):
    protocol=config.get('moving_origin','legacy_approach609_v1')
    declared={'legacy_approach609_v1':(609,'APPROACH','act609'),
              'history_free20_v1':(20,'FREE','history20')}
    if protocol not in declared or type(config.get('source_action_index')) is not int:
        raise ValueError('Unknown declared moving origin')
    selected=declared[protocol]
    if config['source_action_index']!=selected[0]:
        raise ValueError('Moving source index differs from fixed protocol')
    return selected
