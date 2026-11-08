"""Explicit versions of actual recorded origins; no arbitrary replay selection."""


def selected_origin(config):
    protocol=config.get('moving_origin','legacy_approach609_v1')
    declared={'legacy_approach609_v1':(609,'APPROACH','act609'),
              'history_free20_v1':(20,'FREE','history20'),
              'history_free120_v1':(120,'FREE','history120'),
              'history_free220_v1':(220,'FREE','history220')}
    if protocol not in declared or type(config.get('source_action_index')) is not int:
        raise ValueError('Unknown declared moving origin')
    selected=declared[protocol]
    if config['source_action_index']!=selected[0]:
        raise ValueError('Moving source index differs from fixed protocol')
    return selected


def timing_validation_mode(config):
    index,phase,_=selected_origin(config)
    mode=config.get('timing_validation','paired_retest')
    expected='paired_retest' if index==20 else 'validated_algorithm_new_origin'
    if phase!='FREE' or index not in (20,120,220) or mode!=expected:
        raise ValueError('Timing repair validation does not match the predeclared origin')
    return mode
