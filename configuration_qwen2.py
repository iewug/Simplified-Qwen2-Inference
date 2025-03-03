'''
model config
'''
from dataclasses import dataclass

@dataclass
class Qwen2Config:
    vocab_size: int = 152064
    max_position_embeddings: int = 131072
    hidden_size: int = 5120
    intermediate_size: int = 13824
    num_hidden_layers: int = 48
    num_attention_heads: int = 40
    max_window_layers: int = 48
    num_key_value_heads: int = 8
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-05
    use_cache: bool = True
    rope_theta: float = 1000000.0
    attention_dropout: float = 0.0