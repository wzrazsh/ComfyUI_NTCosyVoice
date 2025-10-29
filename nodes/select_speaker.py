import os
import sys
import torch

# 添加必要的路径
nor_dir = os.path.dirname(os.path.dirname(__file__))
Matcha_path = os.path.join(nor_dir, 'third_party/Matcha-TTS')
sys.path.append(nor_dir)
sys.path.append(Matcha_path)

# 导入共享的 CosyVoice 实例
from .shared_cosyvoice import get_shared_cosyvoice, cleanup_shared_cosyvoice


class NTCosyVoiceSelectSpeaker:
    def __init__(self):
        pass

    @property
    def cosyvoice(self):
        """使用共享的 CosyVoice 实例"""
        return get_shared_cosyvoice()

    @classmethod
    def INPUT_TYPES(cls):
        # 使用封装的函数获取可用说话人列表
        from .utils import get_speakers_with_default
        available_speakers = get_speakers_with_default()
        
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "forceInput": True}),
                "speed": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 1.5, "step": 0.1}),
                "speaker_name": ("COMBO", {"options": available_speakers, "default": available_speakers[0]}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("tts_speech",)
    FUNCTION = "generate_speech"
    CATEGORY = "Nineton Nodes"

    def generate_speech(self, text, speed, speaker_name=None):
        speechs = []
        
        # 如果没有提供speaker_name，使用默认值
        if speaker_name is None:
            from .utils import get_available_speakers
            available_speakers = get_available_speakers()
            speaker_name = available_speakers[0] if available_speakers else "no_speakers_available"
        
        print(f"=== NTCosyVoiceSelectSpeaker 执行开始 ===")
        print(f"输入参数 - 文本长度: {len(text)}, 语速: {speed}, 说话人: '{speaker_name}'")
        
        # 第一次检查：运行时重新获取说话人列表
        from .utils import get_available_speakers
        available_speakers = get_available_speakers()
        print(f"第一次查询 - 可用说话人列表: {available_speakers}")
        
        # 如果用户输入的是精确匹配的说话人名称，直接使用
        if speaker_name in available_speakers:
            print(f"第一次检查 - 精确匹配找到说话人: {speaker_name}")
        else:
            # 尝试进行模糊匹配
            matched_speakers = [spk for spk in available_speakers if speaker_name.lower() in spk.lower()]
            print(f"第一次检查 - 模糊匹配结果: {matched_speakers}")
            
            if matched_speakers:
                # 如果找到匹配的说话人，使用第一个匹配项
                selected_speaker = matched_speakers[0]
                print(f"第一次检查 - 使用匹配的说话人: {selected_speaker}")
                speaker_name = selected_speaker
            else:
                print("第一次检查 - 未找到匹配，开始第二次检查")
                # 第二次检查：重新获取说话人列表，确保是最新的
                available_speakers_second_check = get_available_speakers()
                print(f"第二次查询 - 可用说话人列表: {available_speakers_second_check}")
                
                # 在第二次检查中再次尝试匹配
                if speaker_name in available_speakers_second_check:
                    print(f"第二次检查 - 精确匹配找到说话人: {speaker_name}")
                else:
                    # 第二次模糊匹配
                    matched_speakers_second = [spk for spk in available_speakers_second_check if speaker_name.lower() in spk.lower()]
                    print(f"第二次检查 - 模糊匹配结果: {matched_speakers_second}")
                    
                    if matched_speakers_second:
                        selected_speaker = matched_speakers_second[0]
                        print(f"第二次检查 - 使用匹配的说话人: {selected_speaker}")
                        speaker_name = selected_speaker
                    else:
                        # 如果两次检查都不存在，返回错误提示
                        error_msg = f"错误: 说话人 '{speaker_name}' 不存在。可用的说话人列表: {available_speakers_second_check}"
                        print(error_msg)
                        print("=== NTCosyVoiceSelectSpeaker 执行结束（错误） ===")
                        # 返回空音频和错误信息
                        empty_audio = {"waveform": torch.zeros(1, 16000), "sample_rate": self.cosyvoice.sample_rate}
                        return (empty_audio,)
        
        print(f"最终使用的说话人: {speaker_name}")
        
        try:
            # 使用指定的说话人进行零样本推理
            print("开始零样本推理...")
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
            
            print(f"推理完成 - 音频长度: {tts_speech.shape[1]} 采样点")
            print("=== NTCosyVoiceSelectSpeaker 执行结束（成功） ===")
            return (outaudio,)
        except Exception as e:
            # 捕获推理过程中的异常
            error_msg = f"推理过程中发生错误: {str(e)}"
            print(error_msg)
            print("=== NTCosyVoiceSelectSpeaker 执行结束（异常） ===")
            # 返回空音频
            empty_audio = {"waveform": torch.zeros(1, 16000), "sample_rate": self.cosyvoice.sample_rate}
            return (empty_audio,)
        finally:
            # 执行完成后释放GPU内存
            cleanup_shared_cosyvoice()
    
    

# 导出节点类
NODE_CLASS_MAPPINGS = {
    "NTCosyVoiceSelectSpeaker": NTCosyVoiceSelectSpeaker,
}

__all__ = ['NTCosyVoiceSelectSpeaker']