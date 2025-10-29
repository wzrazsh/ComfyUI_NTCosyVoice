"""
刷新说话人列表节点
用于在删除说话人后强制刷新列表显示
"""
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


class NTCosyVoiceRefreshSpeakers:
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
                "force_refresh": (["true", "false"], {
                    "default": "true",
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("refresh_status",)
    FUNCTION = "refresh_speakers"
    CATEGORY = "Nineton Nodes"
    OUTPUT_NODE = True

    def refresh_speakers(self, force_refresh="true"):
        print(f"=== NTCosyVoiceRefreshSpeakers 执行开始 ===")
        
        try:
            # 强制重新加载说话人信息文件
            spk2info_path = f"{self.cosyvoice.model_dir}/spk2info.pt"
            if os.path.exists(spk2info_path):
                # 重新加载spk2info文件
                self.cosyvoice.frontend.spk2info = torch.load(spk2info_path, map_location=self.cosyvoice.frontend.device)
                print("已强制重新加载说话人信息文件")
            
            # 获取当前可用的说话人列表
            available_speakers = self.cosyvoice.list_available_spks()
            print(f"刷新后可用说话人列表: {available_speakers}")
            
            if force_refresh == "true":
                success_msg = f"说话人列表已强制刷新，当前可用说话人: {len(available_speakers)} 个"
            else:
                success_msg = f"说话人列表已刷新，当前可用说话人: {len(available_speakers)} 个"
            
            print(success_msg)
            print("=== NTCosyVoiceRefreshSpeakers 执行结束（成功） ===")
            return (success_msg,)
                
        except Exception as e:
            # 捕获刷新过程中的异常
            error_msg = f"刷新过程中发生错误: {str(e)}"
            print(error_msg)
            print("=== NTCosyVoiceRefreshSpeakers 执行结束（异常） ===")
            return (error_msg,)
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
    "NTCosyVoiceRefreshSpeakers": NTCosyVoiceRefreshSpeakers,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NTCosyVoiceRefreshSpeakers": "CosyVoice 刷新说话人列表",
}

__all__ = ['NTCosyVoiceRefreshSpeakers']