"""Offline representation audit, not a policy generalization evaluation."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from .state_encoding import StateCodec


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--tokenizer',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    import sentencepiece
    from sim.auscultation_expert_layout_v1.kinematics import OpenArmKinematics
    from sim.auscultation_vla_v2.train_openpi import load_dataset,training_instruction
    from sim.auscultation_vla_v2.training_data import normalize
    kin=OpenArmKinematics();codec=StateCodec(kin.lower_limits,kin.upper_limits)
    ds=load_dataset(args.manifest);tok=sentencepiece.SentencePieceProcessor(model_file=str(args.tokenizer))
    report=dict(schema='wam_state_representation_audit_v1',data_hash=ds.data_hash,codec=codec.spec(),
        codec_sha256=hashlib.sha256(Path(__file__).with_name('state_encoding.py').read_bytes()).hexdigest(),
        urdf_sha256=hashlib.sha256(kin.urdf_path.read_bytes()).hexdigest(),episodes=[],
        scope='archived data coverage and quantization only; no generalization claim')
    for episode in ds.episodes:
        raw=episode['state'];old=normalize(raw,ds.normalization['state'])
        new=codec.discrete(raw);restored=codec.decode(new)
        lengths=[len(tok.encode(codec.prompt(training_instruction(str(phase),str(target)),s),add_bos=True))
                 for phase,target,s in zip(episode['phases'],episode['targets'],raw)]
        report['episodes'].append(dict(id=episode['episode_id'],split=episode['split'],frames=len(raw),
            v2_fraction_outside_range=np.mean(abs(old)>1,axis=0).tolist(),new_rejected=0,
            max_abs_quantization_error=np.max(abs(restored-raw),axis=0).tolist(),max_token_length=max(lengths)))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(episodes=len(report['episodes']),max_token_length=max(x['max_token_length'] for x in report['episodes']),
        max_joint_quantization_error_rad=max(max(x['max_abs_quantization_error'][:7]) for x in report['episodes']),
        new_rejected=sum(x['new_rejected'] for x in report['episodes']))),flush=True)


if __name__=='__main__':main()
