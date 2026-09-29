"""Official selected V11 pi05 actor, admitted by the original external producer."""
import copy
import hashlib
from pathlib import Path
import time
import numpy as np
from . import admission_bridge as bridge, policy_contract as contract
from . import action_encoding as codec


def read_bound_json(path,expected_sha):return bridge.read_bound(path,expected_sha)


def input_contract(tree):
    """Exact original pure input leaf contract; does not import scientific modules."""
    out=[]
    def visit(value,path):
        if isinstance(value,dict):
            for k in sorted(value):
                contract.need(type(k) is str,'String input leaf name required');visit(value[k],path+[k])
        else:
            a=np.asarray(value);contract.need(a.dtype.kind in 'biuf' and np.isfinite(a).all(),'Finite numeric model input required')
            out.append(dict(path='/'.join(path),shape=list(a.shape),dtype=str(a.dtype),sha256=hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()))
    visit(tree,[]);contract.need(out,'Model inputs required');return out


def load_actor(admission_path,admission_sha256,*,verifier_binding,sampling_steps=10,seed=917):
    contract.need(type(sampling_steps) is int and sampling_steps==10 and type(seed) is int and seed==917,'Original sampler917/10 required')
    admitted,bridge_receipt=bridge.selected_admission_with_receipt(admission_path,admission_sha256,verifier_binding=verifier_binding)
    contract.need(admitted.get('candidate_admitted') is True and admitted.get('service_eligible') is True,'Positive reconstructed V11 admission required')
    bridge.verify_artifacts(admitted)
    cp=admitted['checkpoint_binding'];checkpoint=Path(cp['checkpoint_path'])
    metadata=read_bound_json(checkpoint/'metadata.json',cp['metadata_sha256']);saved=read_bound_json(admitted['configuration_binding']['path'],admitted['configuration_binding']['sha256']);config=metadata['config'];norm=metadata['normalization']
    contract.need(saved['config']==config and saved['normalization']==norm and saved['data_hash']==cp['data_hash'],'Admitted config/norm/data changed')
    contract.need(bridge.digest(config)==cp['config_sha256'] and bridge.digest(norm)==cp['normalization_sha256'] and config['source_sha256']==cp['source_sha256'],'Selected checkpoint identity differs')
    codec.validate_stats(norm['residual_actions'])
    contract.need(norm['action_encoding_binding']==contract.action_encoding_binding(),'Admitted interval codec binding differs')
    info=contract.make_binding(admitted,admission_sha256,metadata);contract.verify_runtime_sources(info)
    # Only this explicit admitted entry imports/initializes the actual model.
    import jax
    from openpi.models import model as model_api
    from sim.auscultation_pi05_v7 import model as implementation
    from sim.auscultation_pi05_v7.training_data import observation,stack_observations
    runtime=implementation.initialize(61,5e-5)
    contract.need(runtime['assets']==config['assets'],'Official public assets changed')
    implementation.restore(runtime,checkpoint,cp['data_hash'],config,norm)
    bridge.verify_artifacts(admitted);contract.verify_runtime_sources(info)
    sample=implementation.sampler(runtime,steps=10)

    def predict(rgb,state,phase,target):
        contract.need(type(phase) is str and phase in codec.PHASES and type(target) is str and target in codec.TARGETS,'Explicit canonical phase/target required')
        raw=np.asarray(state,np.float32);pixels=np.asarray(rgb)
        contract.need(raw.shape==(41,) and np.isfinite(raw).all(),'Finite original41 state required')
        contract.need(pixels.dtype==np.uint8 and pixels.ndim==3 and pixels.shape[2]==3,'Actual uint8 RGB required')
        row=observation(pixels,raw,phase,target,runtime['tokenizer'],norm);batch=stack_observations([row]);leaves=input_contract(batch)
        obs=model_api.Observation(**jax.device_put(batch));started=time.monotonic()
        normalized=np.asarray(sample(runtime['trainable'],runtime['frozen'],obs,jax.random.key(917)))
        contract.need(normalized.shape==(1,10,7) and normalized.dtype==np.float32 and np.isfinite(normalized).all(),'Finite original float32 normalized10x7 required')
        latent=codec.denormalize_chunk(normalized[0],phase,target,norm['residual_actions'])
        absolute=codec.decode_chunk(normalized[0],raw[:7],phase,target,norm['residual_actions'])
        contract.need(absolute.shape==(10,7) and absolute.dtype==np.float32 and np.isfinite(absolute).all(),'Finite original raw float32 absolute10x7 required')
        contract.need(input_contract(batch)==leaves,'Actual model input changed during consumption')
        return absolute,dict(inference_seconds=time.monotonic()-started,observation_anchor_q=raw[:7].tolist(),model_normalized_residual=normalized[0].tolist(),model_latent_residual=latent.tolist(),codec_phase=phase,codec_target=target,model_input_leaves=leaves,decoder='V11 target interval latent coordinate; one raw measured q anchor for all70; no clipping or posthoc projection')
    predict.runtime_admission=copy.deepcopy(admitted)
    predict.admission_bridge_receipt=copy.deepcopy(bridge_receipt)
    return predict,info
