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


class NTCosyVoiceCrossLingualSampler:
    def __init__(self):
        pass

    @property
    def cosyvoice(self):
        """使用共享的 CosyVoice 实例"""
        return get_shared_cosyvoice()

    @classmethod
    def INPUT_TYPES(cls):
        # 创建节点实例以获取可用说话人列表
        node_instance = cls()
        available_speakers = node_instance.cosyvoice.list_available_spks()
        
        # 如果没有可用说话人，提供一个默认选项
        if not available_speakers:
            available_speakers = ["no_speakers_available"]
        
        return {
            "required": {
                "text": ("STRING", {"multiline": True}),
                "speaker_name": (available_speakers, {
                    "default": available_speakers[0] if available_speakers else "no_speakers_available",
                    "multiline": False
                }),
                "target_language": (["auto", "en", "zh", "ja", "ko", "fr", "de", "es", "it", "ru", "pt", "ar", "hi", "th", "vi", "id", "ms", "fil", "tr", "nl", "pl", "sv", "da", "fi", "no", "cs", "hu", "ro", "bg", "el", "he", "fa", "ur", "bn", "ta", "te", "ml", "kn", "mr", "gu", "pa", "or", "as", "mai", "ne", "si", "my", "km", "lo", "bo", "dz", "ug", "mn", "jv", "su", "ace", "ban", "bug", "mad", "min", "bjn", "ceb", "hil", "ilo", "pam", "war", "bcl", "pag", "tsg", "cbk", "krj", "sgd", "mdh", "mbb", "krj", "akl", "bbc", "bto", "cts", "fbl", "lbl", "mta", "msb", "mwm", "pag", "plv", "rbl", "sgb", "tgl", "tsg", "war", "yka", "kaa", "krc", "kum", "lez", "nog", "tab", "tkr", "tly", "udi", "xal", "yrk", "abq", "ady", "alt", "ava", "bak", "bua", "chv", "crh", "dng", "evn", "gld", "inh", "kbd", "kca", "kjh", "koi", "kpy", "kum", "lbe", "lez", "mdf", "mhr", "mkd", "mns", "mww", "myv", "neg", "nio", "nog", "oaa", "sel", "tab", "tut", "tyv", "ude", "udm", "uum", "xal", "ykg", "yux", "zul", "afr", "amh", "aze", "bel", "ben", "bos", "cat", "ces", "cym", "dan", "deu", "ell", "est", "eus", "fas", "fin", "fra", "gle", "glg", "guj", "hat", "hau", "heb", "hin", "hrv", "hun", "hye", "ibo", "ind", "isl", "jav", "kan", "kat", "kaz", "khm", "kin", "kir", "lao", "lav", "lit", "ltz", "mal", "mar", "mkd", "mlg", "mlt", "mon", "mri", "msa", "mya", "nep", "nld", "nor", "nya", "ori", "pan", "pol", "por", "ron", "run", "rus", "sin", "slk", "slv", "sna", "som", "sot", "spa", "sqi", "srp", "sun", "swa", "swe", "tam", "tel", "tgk", "tgl", "tha", "tur", "ukr", "urd", "uzb", "vie", "xho", "yor", "zho", "zul"], {
                    "default": "auto",
                    "multiline": False
                }),
                "speed": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "main_func"
    CATEGORY = "Nineton Nodes"

    def main_func(self, text, speaker_name, target_language, speed):
        try:
            # 调用跨语言合成
            # 注意：CosyVoice2 使用 inference_cross_lingual 方法而不是 cross_lingual_synthesis
            # 该方法返回一个生成器，我们需要获取第一个结果
            result_generator = self.cosyvoice.inference_cross_lingual(
                tts_text=text,
                prompt_speech_16k=None,  # 对于跨语言合成，prompt_speech_16k 可以为 None
                zero_shot_spk_id=speaker_name,
                stream=False,
                speed=speed,
                text_frontend=True
            )
            
            # 获取第一个结果
            result = next(result_generator)
            
            # 返回音频数据
            audio_output = {
                "waveform": torch.tensor(result["tts_speech"]).unsqueeze(0),
                "sample_rate": self.cosyvoice.sample_rate
            }
            
            return (audio_output,)
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
    "NTCosyVoiceCrossLingualSampler": NTCosyVoiceCrossLingualSampler,
}

__all__ = ['NTCosyVoiceCrossLingualSampler']