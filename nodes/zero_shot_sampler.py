import os
import sys
import torch

# 添加必要的路径
nor_dir = os.path.dirname(os.path.dirname(__file__))
Matcha_path = os.path.join(nor_dir, 'third_party/Matcha-TTS')
sys.path.append(nor_dir)
sys.path.append(Matcha_path)

# 导入共享的 CosyVoice 实例
from .shared_cosyvoice import get_shared_cosyvoice

# 导入共享函数
from .utils import nt_load_wav


class NTCosyVoiceZeroShotSampler:
    def __init__(self):
        pass

    @property
    def cosyvoice(self):
        """使用共享的 CosyVoice 实例"""
        return get_shared_cosyvoice()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "forceInput": True}),
                "speed": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 1.5, "step": 0.1}),
            },
            "optional": {
                "prompt_text": ("STRING", {"multiline": True, "default": "", "forceInput": True}),
                "prompt_speech": ("AUDIO", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("tts_speech",)
    FUNCTION = "main_func"
    CATEGORY = "Nineton Nodes"

    def main_func(self, audio, speed, text, prompt_text):
        waveform = audio["waveform"].squeeze(0)
        sample_rate = audio["sample_rate"]
        print(f"waveform:{waveform}, sample_rate:{sample_rate}")
        prompt_speech_16k = nt_load_wav(waveform, sample_rate, 16000)
        speechs = []
        
        try:
            for i, j in enumerate(self.cosyvoice.inference_zero_shot(tts_text=text, prompt_text=prompt_text, prompt_speech_16k=prompt_speech_16k, stream=False, speed=speed)):
                speechs.append(j['tts_speech'])

            tts_speech = torch.cat(speechs, dim=1)
            tts_speech = tts_speech.unsqueeze(0)
            outaudio = {"waveform": tts_speech, "sample_rate": self.cosyvoice.sample_rate}

            return (outaudio,)
        finally:
            # 执行完成后释放GPU内存
            self._cleanup_gpu_memory()
    
    def _cleanup_gpu_memory(self):
        """清理GPU内存（仅清理PyTorch缓存，不清理共享的CosyVoice实例）"""
        if torch.cuda.is_available():
            # 仅清理PyTorch缓存，共享的CosyVoice实例由共享管理器统一管理
            torch.cuda.empty_cache()
            print("GPU缓存已清理")
        else:
            print("GPU不可用，跳过内存清理")


# 导出节点类
NODE_CLASS_MAPPINGS = {
    "NTCosyVoiceZeroShotSampler": NTCosyVoiceZeroShotSampler,
}

__all__ = ['NTCosyVoiceZeroShotSampler']