import os
import sys
import torch

nor_dir = os.path.dirname(__file__)
Matcha_path = os.path.join(nor_dir, 'third_party/Matcha-TTS')
sys.path.append(nor_dir)
sys.path.append(Matcha_path)

from cosyvoice.cli.cosyvoice import CosyVoice2

def nt_load_wav(waveform, sample_rate, target_sample_rate):
    # 如果采样率不同，需要重采样
    if sample_rate != target_sample_rate:
        # 使用线性插值进行重采样
        waveform = torch.nn.functional.interpolate(waveform.unsqueeze(0), 
                                                  size=int(waveform.shape[-1] * target_sample_rate / sample_rate), 
                                                  mode='linear', 
                                                  align_corners=False).squeeze(0)
    return waveform

# 导入拆分后的节点类
from .nodes.zero_shot_sampler import NTCosyVoiceZeroShotSampler
from .nodes.clone_speaker import NTCosyVoiceCloneSpeaker
from .nodes.select_speaker import NTCosyVoiceSelectSpeaker
from .nodes.cross_lingual_sampler import NTCosyVoiceCrossLingualSampler
from .nodes.instruct2_sampler import NTCosyVoiceInstruct2Sampler
from .nodes.delete_speaker import NTCosyVoiceDeleteSpeaker
from .nodes.refresh_speakers import NTCosyVoiceRefreshSpeakers

NODE_CLASS_MAPPINGS = {
    "NTCosyVoiceZeroShotSampler": NTCosyVoiceZeroShotSampler,
    "NTCosyVoiceInstruct2Sampler": NTCosyVoiceInstruct2Sampler,
    "NTCosyVoiceCrossLingualSampler": NTCosyVoiceCrossLingualSampler,
    "NTCosyVoiceCloneSpeaker": NTCosyVoiceCloneSpeaker,
    "NTCosyVoiceSelectSpeaker": NTCosyVoiceSelectSpeaker,
    "NTCosyVoiceDeleteSpeaker": NTCosyVoiceDeleteSpeaker,
    "NTCosyVoiceRefreshSpeakers": NTCosyVoiceRefreshSpeakers
}

__all__ = ['NODE_CLASS_MAPPINGS']