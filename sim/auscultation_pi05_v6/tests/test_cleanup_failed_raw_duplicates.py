import json
from pathlib import Path
import pytest
from sim.auscultation_pi05_v6.cleanup_failed_raw_duplicates import validate_archive,validate_completed_failure,safe_file

def test_failed_completed_only():
    validate_completed_failure({'passed':False},{'close_returned':True,'errors':[]})
    for c,t in [({'passed':True},{'close_returned':True,'errors':[]}),({'passed':False},{'close_returned':False,'errors':[]}),({'passed':False},{'close_returned':True,'errors':['bad']})]:
        with pytest.raises(ValueError):validate_completed_failure(c,t)

def test_archive_requires_exact_failed_identity_and_whitelist():
    data=dict(verified_identical=True,acceptance_passed=False,source='linbox:/src',archive='qwen-agent-l40s:/dst',files={'sensors/00001.npz':'a'*64,'sensors/../outside.npz':'b'*64,'expert_trace.jsonl':'c'*64})
    assert validate_archive(data,'/src','/dst')=={'sensors/00001.npz':'a'*64}
    for field,value in [('verified_identical',False),('acceptance_passed',True),('source','other')]:
        with pytest.raises(ValueError):validate_archive({**data,field:value},'/src','/dst')

def test_source_paths_reject_symlinks_and_escape(tmp_path):
    (tmp_path/'sensors').mkdir();(tmp_path/'sensors/00001.npz').write_bytes(b'raw')
    assert safe_file(tmp_path,'sensors/00001.npz')==tmp_path/'sensors/00001.npz'
    for name in ('../00001.npz','sensors/../00001.npz','expert_trace.jsonl'):
        with pytest.raises(ValueError):safe_file(tmp_path,name)
    (tmp_path/'sensors/00002.npz').symlink_to(tmp_path/'sensors/00001.npz')
    with pytest.raises(ValueError):safe_file(tmp_path,'sensors/00002.npz')
