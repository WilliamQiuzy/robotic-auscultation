"""Explicit operational episode names; not imported by acquisition or training."""
import re

CANDIDATE_EPISODE='R5_expert_observed_redundancy30_v1'


def qualification_id(episode):
    if not isinstance(episode,str) or not (re.fullmatch(r'[RL][1-5]_expert_observed_v1',episode) or episode==CANDIDATE_EPISODE):
        raise ValueError('Unexpected V6 operational episode')
    return episode.replace('_expert_','_')
