"""CPU-side adapter for pinned openpi π0.5; no model or robot is run here.

The official flow head is instantiated at action_dim=7. The 32-D measured
state is independent and enters the official pi05-style discrete prompt.
"""
import numpy as np
from PIL import Image

OPENPI_COMMIT='215abfb217dbac7d5f1273282331b9b1866c0479'
PROJECTION_KEYS=(('action_in_proj','kernel'),('action_out_proj','kernel'),('action_out_proj','bias'))
# Exact paths emitted by the pinned gemma_2b_lora + gemma_300m_lora model.
# Attention Einsum uses lora_a/b; FeedForward embeds the factor in its leaf
# name. Keep initialization and optimizer selection on the same allowlist.
LORA_PATHS=frozenset(
    [('PaliGemma','llm','layers','attn',layer+suffix,'lora_'+factor)
     for layer in ('attn_vec_einsum','kv_einsum','q_einsum')
     for suffix in ('','_1') for factor in ('a','b')]
    + [('PaliGemma','llm','layers','mlp'+suffix,layer+'_lora_'+factor)
       for suffix in ('','_1') for layer in ('gating_einsum','linear') for factor in ('a','b')]
)
TASK_PROJECTION_PATHS=frozenset((module,leaf) for module in ('action_in_proj','action_out_proj')
                               for leaf in ('kernel','bias'))


def adapt_projection_params(flat_params):
    """Copy the parameter mapping; slice only the three declared native tensors."""
    try:
        incoming,outgoing,bias=[flat_params[k] for k in PROJECTION_KEYS]
        valid=(incoming.ndim==2 and incoming.shape[0]==32 and outgoing.ndim==2
            and outgoing.shape==(incoming.shape[1],32) and bias.shape==(32,))
    except (KeyError,AttributeError):valid=False
    if not valid:raise ValueError('Pretrained action projection shape mismatch')
    result=dict(flat_params)
    result[PROJECTION_KEYS[0]]=incoming[:7,:]
    result[PROJECTION_KEYS[1]]=outgoing[:,:7]
    result[PROJECTION_KEYS[2]]=bias[:7]
    return result


def lora_path(path):
    return tuple(map(str,path)) in LORA_PATHS


def trainable_path(path):
    names=tuple(map(str,path))
    return names in TASK_PROJECTION_PATHS or lora_path(names)


def state_prompt(instruction,state):
    state=np.asarray(state,float)
    if state.shape!=(32,) or not np.isfinite(state).all():
        raise ValueError('Pi05 task requires all 32 finite state features')
    if not isinstance(instruction,str) or not instruction.strip():raise ValueError('Missing task instruction')
    # Task-specific 4096-bin physical/asinh state recipe; requires retraining.
    if np.any(abs(state)>1):raise ValueError('State outside declared physical domain')
    discrete=np.rint((state+1)*(4095/2)).astype(np.int32)
    cleaned=instruction.strip().replace('_',' ').replace('\n',' ')
    return f"Task: {cleaned}, State: {' '.join(map(str,discrete))};\nAction: "


def tokenize_state(sentencepiece_model,instruction,state,*,max_len=200):
    if type(max_len) is not int or max_len<1:raise ValueError('Invalid token budget')
    ids=sentencepiece_model.encode(state_prompt(instruction,state),add_bos=True)
    if not ids or len(ids)>max_len:
        raise ValueError('Task/state token budget exceeded; do not truncate modalities')
    mask=np.arange(max_len)<len(ids)
    return np.pad(np.asarray(ids,np.int32),(0,max_len-len(ids))),mask


def prepare_images(rgb):
    if not isinstance(rgb,np.ndarray) or rgb.dtype!=np.uint8 or rgb.ndim!=3 or rgb.shape[-1]!=3 or min(rgb.shape[:2])<2:
        raise ValueError('One uint8 RGB camera frame is required')
    height,width=rgb.shape[:2];scale=224/max(height,width)
    w,h=max(1,round(width*scale)),max(1,round(height*scale))
    resized=np.asarray(Image.fromarray(rgb).resize((w,h),Image.Resampling.BILINEAR))
    # Letterbox RGB only. Metric depth/target calibration is processed before
    # this model image transform and never resized as a false color camera.
    pixels=np.zeros((224,224,3),np.uint8)
    y,x=(224-h)//2,(224-w)//2;pixels[y:y+h,x:x+w]=resized
    image=(pixels.astype(np.float32)/127.5)-1
    images={'base_0_rgb':image,'left_wrist_0_rgb':np.zeros_like(image),'right_wrist_0_rgb':np.zeros_like(image)}
    return images,dict(base_0_rgb=True,left_wrist_0_rgb=False,right_wrist_0_rgb=False)
