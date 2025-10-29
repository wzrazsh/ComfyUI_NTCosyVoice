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


class NTCosyVoiceDeleteSpeaker:
    def __init__(self):
        pass
 
    @classmethod
    def INPUT_TYPES(cls):
        # 直接从spk2info.pt文件中读取可用说话人列表，避免加载模型
        available_speakers = []
        try:           
            # 获取模型路径
            model_path = os.path.join(nor_dir, 'pretrained_models/CosyVoice2-0.5B')
            spk2info_path = os.path.join(model_path, "spk2info.pt")
            
            # 检查文件是否存在并加载
            if os.path.exists(spk2info_path):
                spk2info = torch.load(spk2info_path, map_location='cpu')
                available_speakers = list(spk2info.keys())
        except Exception as e:
            print(f"读取spk2info.pt文件时出错: {e}")
        
        # 如果没有可用说话人，提供一个默认选项
        if not available_speakers:
            available_speakers = ["no_speakers_available"]
        
        return {
            "required": {
                "speaker_name": (available_speakers, {
                    "default": available_speakers[0] if available_speakers else "no_speakers_available",
                }),
            }
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # 强制每次执行都重新加载输入类型，确保列表实时更新
        return float("inf")

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("deleted_speaker",)
    FUNCTION = "delete_speaker"
    CATEGORY = "Nineton Nodes"
    OUTPUT_NODE = True

    def delete_speaker(self, speaker_name):
        print(f"=== NTCosyVoiceDeleteSpeaker 执行开始 ===")
        print(f"要删除的说话人: '{speaker_name}'")      

        try:
            # 获取模型路径
            model_path = os.path.join(nor_dir, 'pretrained_models/CosyVoice2-0.5B')
            spk2info_path = os.path.join(model_path, "spk2info.pt")
            
            # 检查文件是否存在
            if not os.path.exists(spk2info_path):
                error_msg = f"错误: spk2info.pt 文件不存在于路径 {spk2info_path}"
                print(error_msg)
                print("=== NTCosyVoiceDeleteSpeaker 执行结束（错误） ===")
                return (error_msg,)
            
            # 加载说话人信息
            spk2info = torch.load(spk2info_path, map_location='cpu')
            available_speakers = list(spk2info.keys())
            print(f"删除前可用说话人列表: {available_speakers}")
            
            # 检查说话人是否存在
            if speaker_name not in available_speakers:
                error_msg = f"错误: 说话人 '{speaker_name}' 不存在。可用的说话人列表: {available_speakers}"
                print(error_msg)
                print("=== NTCosyVoiceDeleteSpeaker 执行结束（错误） ===")
                return (f"错误: {speaker_name} 不存在",)
            
            # 从spk2info字典中删除说话人
            if speaker_name in spk2info:
                del spk2info[speaker_name]
                print(f"已从内存中删除说话人: {speaker_name}")
                
                # 保存更新后的说话人信息到文件
                torch.save(spk2info, spk2info_path)
                print("已保存更新后的说话人信息到文件")
                
                # 重新加载说话人信息，确保文件更改生效
                if os.path.exists(spk2info_path):
                    # 重新加载spk2info文件
                    updated_spk2info = torch.load(spk2info_path, map_location='cpu')
                    updated_speakers = list(updated_spk2info.keys())
                    print("已重新加载说话人信息文件")
                else:
                    updated_speakers = []
                
                print(f"删除后可用说话人列表: {updated_speakers}")
                
                if speaker_name not in updated_speakers:
                    success_msg = f"成功删除说话人: {speaker_name}"
                    print(success_msg)
                    print("=== NTCosyVoiceDeleteSpeaker 执行结束（成功） ===")
                    return (success_msg,)
                else:
                    error_msg = f"警告: 说话人 '{speaker_name}' 删除失败，仍然存在于列表中"
                    print(error_msg)
                    print("=== NTCosyVoiceDeleteSpeaker 执行结束（警告） ===")
                    return (error_msg,)
            else:
                error_msg = f"错误: 说话人 '{speaker_name}' 在spk2info字典中不存在"
                print(error_msg)
                print("=== NTCosyVoiceDeleteSpeaker 执行结束（错误） ===")
                return (error_msg,)
                
        except Exception as e:
            # 捕获删除过程中的异常
            error_msg = f"删除过程中发生错误: {str(e)}"
            print(error_msg)
            print("=== NTCosyVoiceDeleteSpeaker 执行结束（异常） ===")
            return (error_msg,)
        finally:
            # 执行完成后释放GPU内存
            cleanup_shared_cosyvoice()
    


# 导出节点类
NODE_CLASS_MAPPINGS = {
    "NTCosyVoiceDeleteSpeaker": NTCosyVoiceDeleteSpeaker,
}

__all__ = ['NTCosyVoiceDeleteSpeaker']