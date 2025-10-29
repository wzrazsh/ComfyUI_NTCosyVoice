import os
import sys
import torch
import time

# 添加必要的路径
nor_dir = os.path.dirname(os.path.dirname(__file__))
Matcha_path = os.path.join(nor_dir, 'third_party/Matcha-TTS')
sys.path.append(nor_dir)
sys.path.append(Matcha_path)

# 导入共享的 CosyVoice 实例
from .shared_cosyvoice import get_shared_cosyvoice, cleanup_shared_cosyvoice

# 导入共享函数
from .utils import nt_load_wav, get_speakers_with_default


class NTCosyVoiceCloneSpeaker:
    def __init__(self):
        pass

    @property
    def cosyvoice(self):
        """使用共享的 CosyVoice 实例"""
        return get_shared_cosyvoice()

    @classmethod
    def INPUT_TYPES(cls):
        # 使用封装的函数获取可用说话人列表
        # available_speakers = get_speakers_with_default()
        
        return {
            "required": {
                "audio": ("AUDIO",),
                "prompt_text": ("STRING", {"multiline": True}),
                "speaker_name": ("STRING", {"default": "new_speaker","multiline": False}),
                 },
        }
    

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("speaker_name",)
    FUNCTION = "clone_speaker"
    CATEGORY = "Nineton Nodes"
    OUTPUT_NODE = True

    def clone_speaker(self, audio, prompt_text, speaker_name):
        waveform = audio["waveform"].squeeze(0)
        sample_rate = audio["sample_rate"]
        
        try:
            # 将音频转换为16kHz采样率
            prompt_speech_16k = nt_load_wav(waveform, sample_rate, 16000)
            
            # 添加零样本说话人
            success = self.cosyvoice.add_zero_shot_spk(prompt_text, prompt_speech_16k, speaker_name)
            
            # 保存说话人信息到spk2info.pt文件
            if success:
                self.cosyvoice.save_spkinfo()
            else:
                print(f"警告: 说话人克隆失败，说话人名称: {speaker_name}")
                
            return (speaker_name,)
        finally:
            # 执行完成后释放GPU内存
            cleanup_shared_cosyvoice()
    


# 导出节点类
NODE_CLASS_MAPPINGS = {
    "NTCosyVoiceCloneSpeaker": NTCosyVoiceCloneSpeaker,
}

__all__ = ['NTCosyVoiceCloneSpeaker']