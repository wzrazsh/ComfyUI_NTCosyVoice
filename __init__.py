import os
import sys
import torch
nor_dir = os.path.dirname(__file__)
Matcha_path = os.path.join(nor_dir, 'third_party/Matcha-TTS')
sys.path.append(nor_dir)
sys.path.append(Matcha_path)

from comfy.utils import common_upscale
from comfy.sd import load_checkpoint_guess_config
import folder_paths
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

class NTCosyVoiceZeroShotSampler:
    def __init__(self):
        self.__cosyvoice = None

    @property
    def cosyvoice(self):
        if self.__cosyvoice is None:
            model_path = os.path.join(nor_dir, 'pretrained_models/CosyVoice2-0.5B')
            # 修复模型路径，使用绝对路径
            self.__cosyvoice = CosyVoice2(model_path, load_jit=False, load_trt=False, load_vllm=False, fp16=False)
        return self.__cosyvoice

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio": ("AUDIO",),
                "speed": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 1.5, "step": 0.1}),
                "text": ("STRING", {"multiline": True}),
                "prompt_text": ("STRING", {"multiline": True}),
            },
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
        for i, j in enumerate(self.cosyvoice.inference_zero_shot(tts_text=text, prompt_text=prompt_text, prompt_speech_16k=prompt_speech_16k, stream=False, speed=speed)):
            speechs.append(j['tts_speech'])

        tts_speech = torch.cat(speechs, dim=1)
        tts_speech = tts_speech.unsqueeze(0)
        outaudio = {"waveform": tts_speech, "sample_rate": self.cosyvoice.sample_rate}

        return (outaudio,)


class NTCosyVoiceCloneSpeaker:
    def __init__(self):
        self.__cosyvoice = None

    @property
    def cosyvoice(self):
        if self.__cosyvoice is None:
            model_path = os.path.join(nor_dir, 'pretrained_models/CosyVoice2-0.5B')
            # 修复模型路径，使用绝对路径
            self.__cosyvoice = CosyVoice2(model_path, load_jit=False, load_trt=False, load_vllm=False, fp16=False)
        return self.__cosyvoice

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio": ("AUDIO",),
                "prompt_text": ("STRING", {"multiline": True}),
                "speaker_name": ("STRING", {"multiline": False}),
            },
        }

    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ("success",)
    FUNCTION = "clone_speaker"
    CATEGORY = "Nineton Nodes"
    OUTPUT_NODE = True

    def clone_speaker(self, audio, prompt_text, speaker_name):
        waveform = audio["waveform"].squeeze(0)
        sample_rate = audio["sample_rate"]
        
        # 将音频转换为16kHz采样率
        prompt_speech_16k = nt_load_wav(waveform, sample_rate, 16000)
        
        # 添加零样本说话人
        success = self.cosyvoice.add_zero_shot_spk(prompt_text, prompt_speech_16k, speaker_name)
        
        # 保存说话人信息到spk2info.pt文件
        if success:
            self.cosyvoice.save_spkinfo()
            
        return (success,)


class NTCosyVoiceSelectSpeaker:
    def __init__(self):
        self.__cosyvoice = None

    @property
    def cosyvoice(self):
        if self.__cosyvoice is None:
            model_path = os.path.join(nor_dir, 'pretrained_models/CosyVoice2-0.5B')
            # 修复模型路径，使用绝对路径
            self.__cosyvoice = CosyVoice2(model_path, load_jit=False, load_trt=False, load_vllm=False, fp16=False)
        return self.__cosyvoice

    @classmethod
    def INPUT_TYPES(s):
        # 创建节点实例以获取可用说话人列表
        node_instance = s()
        available_speakers = node_instance.cosyvoice.list_available_spks()
        
        # 如果没有可用说话人，提供一个默认选项
        if not available_speakers:
            available_speakers = ["no_speakers_available"]
        
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "forceInput": True}),
                "speed": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 1.5, "step": 0.1}),
                "speaker_name": (available_speakers, {"default": available_speakers[0] if available_speakers else "no_speakers_available"}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("tts_speech",)
    FUNCTION = "generate_speech"
    CATEGORY = "Nineton Nodes"

    def generate_speech(self, text, speed, speaker_name):
        speechs = []
        
        # 检查说话人是否存在
        available_speakers = self.cosyvoice.list_available_spks()
        if speaker_name not in available_speakers:
            raise ValueError(f"说话人 '{speaker_name}' 不存在。可用的说话人: {available_speakers}")
        
        # 使用指定的说话人进行零样本推理
        for i, j in enumerate(self.cosyvoice.inference_zero_shot(
            tts_text=text, 
            prompt_text="", 
            prompt_speech_16k=torch.zeros(1, 16000),  # 占位符音频
            zero_shot_spk_id=speaker_name,
            stream=False, 
            speed=speed
        )):
            speechs.append(j['tts_speech'])

        tts_speech = torch.cat(speechs, dim=1)
        tts_speech = tts_speech.unsqueeze(0)
        outaudio = {"waveform": tts_speech, "sample_rate": self.cosyvoice.sample_rate}

        return (outaudio,)


class NTCosyVoiceCrossLingualSampler:
    def __init__(self):
        self.__cosyvoice = None

    @property
    def cosyvoice(self):
        if self.__cosyvoice is None:
            model_path = os.path.join(nor_dir, 'pretrained_models/CosyVoice2-0.5B')
            # 修复模型路径，使用绝对路径
            self.__cosyvoice = CosyVoice2(model_path, load_jit=False, load_trt=False, load_vllm=False, fp16=False)
        return self.__cosyvoice

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio": ("AUDIO",),
                "speed": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 1.5, "step": 0.1}),
                "text": ("STRING", {"multiline": True}),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("tts_speech",)
    FUNCTION = "main_func"
    CATEGORY = "Nineton Nodes"

    def main_func(self, audio, speed, text):
        waveform = audio["waveform"].squeeze(0)
        sample_rate = audio["sample_rate"]

        prompt_speech_16k = nt_load_wav(waveform, sample_rate, 16000)
        speechs = []
        for i, j in enumerate(self.cosyvoice.inference_cross_lingual(tts_text=text,
                prompt_speech_16k=prompt_speech_16k, stream=False, speed=speed)):
            speechs.append(j['tts_speech'])

        tts_speech = torch.cat(speechs, dim=1)
        tts_speech = tts_speech.unsqueeze(0)
        outaudio = {"waveform": tts_speech, "sample_rate": self.cosyvoice.sample_rate}

        return (outaudio,)


class NTCosyVoiceInstruct2Sampler:
    def __init__(self):
        self.__cosyvoice = None

    @property
    def cosyvoice(self):
        if self.__cosyvoice is None:
            model_path = os.path.join(nor_dir, 'pretrained_models/CosyVoice2-0.5B')
            # 修复模型路径，使用绝对路径
            self.__cosyvoice = CosyVoice2(model_path, load_jit=False, load_trt=False, load_vllm=False, fp16=False)
        return self.__cosyvoice

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio": ("AUDIO",),
                "speed": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 1.5, "step": 0.1}),
                "text": ("STRING", {"multiline": True}),
                "instruct": ("STRING", {"multiline": True}),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("tts_speech",)
    FUNCTION = "main_func"
    CATEGORY = "Nineton Nodes"

    def main_func(self, audio, speed, text, instruct):
        waveform = audio["waveform"].squeeze(0)
        sample_rate = audio["sample_rate"]

        prompt_speech_16k = nt_load_wav(waveform, sample_rate, 16000)

        speechs = []
        for i, j in enumerate(self.cosyvoice.inference_instruct2(tts_text=text, instruct_text=instruct, prompt_speech_16k=prompt_speech_16k, stream=False, speed=speed)):
            speechs.append(j['tts_speech'])

        tts_speech = torch.cat(speechs, dim=1)
        tts_speech = tts_speech.unsqueeze(0)
        outaudio = {"waveform": tts_speech, "sample_rate": self.cosyvoice.sample_rate}

        return (outaudio,)


NODE_CLASS_MAPPINGS = {
    "NTCosyVoiceZeroShotSampler": NTCosyVoiceZeroShotSampler,
    "NTCosyVoiceInstruct2Sampler": NTCosyVoiceInstruct2Sampler,
    "NTCosyVoiceCrossLingualSampler": NTCosyVoiceCrossLingualSampler,
    "NTCosyVoiceCloneSpeaker": NTCosyVoiceCloneSpeaker,
    "NTCosyVoiceSelectSpeaker": NTCosyVoiceSelectSpeaker
}

__all__ = ['NODE_CLASS_MAPPINGS']