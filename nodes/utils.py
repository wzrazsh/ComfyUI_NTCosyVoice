import os
import torch

def nt_load_wav(waveform, sample_rate, target_sample_rate):
    """音频重采样函数"""
    # 如果采样率不同，需要重采样
    if sample_rate != target_sample_rate:
        # 使用线性插值进行重采样
        waveform = torch.nn.functional.interpolate(waveform.unsqueeze(0), 
                                                  size=int(waveform.shape[-1] * target_sample_rate / sample_rate), 
                                                  mode='linear', 
                                                  align_corners=False).squeeze(0)
    return waveform


def get_available_speakers():
    """
    获取可用说话人列表的封装函数
    
    Args:
        use_cosyvoice (bool): 是否使用CosyVoice实例获取列表，如果为False则直接从文件读取
    
    Returns:
        list: 可用说话人名称列表
    """
    # 获取基础路径 - 与delete_speaker.py保持一致
    nor_dir = os.path.dirname(os.path.dirname(__file__))    
     
    # 直接从spk2info.pt文件读取说话人列表 - 与delete_speaker.py保持一致
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
    
    return available_speakers


def get_speakers_with_default():
    """
    获取可用说话人列表，如果没有可用说话人则返回默认选项
    
    Returns:
        list: 可用说话人名称列表，如果没有则返回["no_speakers_available"]
    """
    available_speakers = get_available_speakers()
    
    # 如果没有可用说话人，提供一个默认选项
    if not available_speakers:
        available_speakers = ["no_speakers_available"]
    
    return available_speakers