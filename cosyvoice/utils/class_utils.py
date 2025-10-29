# Copyright [2023-11-28] <sxc19@mails.tsinghua.edu.cn, Xingchen Song>
#            2024 Alibaba Inc (authors: Xiang Lyu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch

from cosyvoice.transformer.activation import Swish
from cosyvoice.transformer.subsampling import (
    LinearNoSubsampling,
    EmbedinigNoSubsampling,
    Conv1dSubsampling2,
    Conv2dSubsampling4,
    Conv2dSubsampling6,
    Conv2dSubsampling8,
)
from cosyvoice.transformer.embedding import (PositionalEncoding,
                                             RelPositionalEncoding,
                                             WhisperPositionalEncoding,
                                             LearnablePositionalEncoding,
                                             NoPositionalEncoding)
from cosyvoice.transformer.attention import (MultiHeadedAttention,
                                             RelPositionMultiHeadedAttention)
from cosyvoice.transformer.embedding import EspnetRelPositionalEncoding
from cosyvoice.transformer.subsampling import LegacyLinearNoSubsampling
from cosyvoice.llm.llm import TransformerLM, Qwen2LM
from cosyvoice.flow.flow import MaskedDiffWithXvec, CausalMaskedDiffWithXvec
from cosyvoice.hifigan.generator import HiFTGenerator
from cosyvoice.cli.model import CosyVoiceModel, CosyVoice2Model


COSYVOICE_ACTIVATION_CLASSES = {
    "hardtanh": torch.nn.Hardtanh,
    "tanh": torch.nn.Tanh,
    "relu": torch.nn.ReLU,
    "selu": torch.nn.SELU,
    "swish": getattr(torch.nn, "SiLU", Swish),
    "gelu": torch.nn.GELU,
}

COSYVOICE_SUBSAMPLE_CLASSES = {
    "linear": LinearNoSubsampling,
    "linear_legacy": LegacyLinearNoSubsampling,
    "embed": EmbedinigNoSubsampling,
    "conv1d2": Conv1dSubsampling2,
    "conv2d": Conv2dSubsampling4,
    "conv2d6": Conv2dSubsampling6,
    "conv2d8": Conv2dSubsampling8,
    'paraformer_dummy': torch.nn.Identity
}

COSYVOICE_EMB_CLASSES = {
    "embed": PositionalEncoding,
    "abs_pos": PositionalEncoding,
    "rel_pos": RelPositionalEncoding,
    "rel_pos_espnet": EspnetRelPositionalEncoding,
    "no_pos": NoPositionalEncoding,
    "abs_pos_whisper": WhisperPositionalEncoding,
    "embed_learnable_pe": LearnablePositionalEncoding,
}

COSYVOICE_ATTENTION_CLASSES = {
    "selfattn": MultiHeadedAttention,
    "rel_selfattn": RelPositionMultiHeadedAttention,
}


def get_model_type(configs):
    # NOTE CosyVoice2Model inherits CosyVoiceModel
    # Check the actual type of the instantiated objects
    llm_type = type(configs['llm'])
    flow_type = type(configs['flow'])
    hift_type = type(configs['hift'])
    
    if llm_type == TransformerLM and flow_type == MaskedDiffWithXvec and hift_type == HiFTGenerator:
        return CosyVoiceModel
    if llm_type == Qwen2LM and flow_type == CausalMaskedDiffWithXvec and hift_type == HiFTGenerator:
        return CosyVoice2Model
    raise TypeError('No valid model type found! Got llm: {}, flow: {}, hift: {}'.format(llm_type, flow_type, hift_type))
