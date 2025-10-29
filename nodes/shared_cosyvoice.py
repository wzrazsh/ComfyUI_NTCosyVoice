"""
共享的 CosyVoice 对象管理模块
让所有节点可以共享同一个 CosyVoice 实例，避免重复初始化和内存浪费
"""

import os
import sys
import torch

# 添加必要的路径
nor_dir = os.path.dirname(os.path.dirname(__file__))
Matcha_path = os.path.join(nor_dir, 'third_party/Matcha-TTS')
sys.path.append(nor_dir)
sys.path.append(Matcha_path)

from cosyvoice.cli.cosyvoice import CosyVoice2


class SharedCosyVoiceManager:
    """共享的 CosyVoice 对象管理器"""
    
    _instance = None
    _cosyvoice = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SharedCosyVoiceManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self._initialized = True
    
    @property
    def cosyvoice(self):
        """获取共享的 CosyVoice 实例"""
        if self._cosyvoice is None:
            model_path = os.path.join(nor_dir, 'pretrained_models/CosyVoice2-0.5B')
            print(f"初始化共享 CosyVoice 实例，模型路径: {model_path}")
            self._cosyvoice = CosyVoice2(model_path, load_jit=False, load_trt=False, load_vllm=False, fp16=False)
        return self._cosyvoice
    
    def cleanup(self):
        """清理共享的 CosyVoice 实例"""
        if self._cosyvoice is not None:
            if torch.cuda.is_available():
                # 清理 GPU 内存
                torch.cuda.empty_cache()
                
                # 删除对象引用
                del self._cosyvoice
                self._cosyvoice = None
                
                # 强制垃圾回收
                import gc
                gc.collect()
                
                # 再次清理 GPU 缓存
                torch.cuda.empty_cache()
                
                print("共享 CosyVoice GPU 内存已释放")
            else:
                print("共享 CosyVoice 对象已清理")
    
    def reload(self):
        """重新加载 CosyVoice 实例"""
        self.cleanup()
        self._cosyvoice = None  # 重置实例，下次访问时会重新创建


# 创建全局共享管理器实例
shared_manager = SharedCosyVoiceManager()


def get_shared_cosyvoice():
    """获取共享的 CosyVoice 实例"""
    return shared_manager.cosyvoice


def cleanup_shared_cosyvoice():
    """清理共享的 CosyVoice 实例"""
    shared_manager.cleanup()


def reload_shared_cosyvoice():
    """重新加载共享的 CosyVoice 实例"""
    return shared_manager.reload()