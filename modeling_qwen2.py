from typing import List, Optional, Tuple, Union
import torch
from torch import nn
from cache import DynamicCache
from configuration_qwen2 import Qwen2Config

class Qwen2MLP(nn.Module):
    def __init__(self, config: Qwen2Config):
        '''
        (...,hidden_size) -> (...,hidden_size)
        '''
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size # 5120
        self.intermediate_size = config.intermediate_size # 13824
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False, dtype=torch.bfloat16)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False, dtype=torch.bfloat16)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False, dtype=torch.bfloat16)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


def rotate_half(x):
    """
    Rotates half the hidden dims of the input.
    [[1,2,3,4],  ->  [[-3,-4,1,2],
     [5,6,7,8]]       [-7,-8,5,6]]
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """
    Applies Rotary Position Embedding to the query and key tensors.
    [               [                   [               [
    q_0             cosmθ_0             -q_{d/2}        sinmθ_0
    q_1             cosmθ_1             -q_{d/2+1}      sinmθ_1
    q_2             cosmθ_2             -q_{d/2+2}      sinmθ_2
    ...             ...                 ...             ...
    q_{d/2-1}   *   cosmθ_{d/2-1}   +   -q_{d-1}   *    sinmθ_{d/2-1}
    q_{d/2}         cosmθ_0             q_0             sinmθ_0
    q_{d/2+1}       cosmθ_1             q_1             sinmθ_1
    ...             ...                 ...             ...
    q_{d-1}         cosmθ_{d/2-1}       q_{d/2-1}       cosmθ_{d/2-1}
    ]               ]                   ]               ]
    这里是element-wise相乘, d为head_dim, q_i都是标量, m是这个q所在原句子的位置
    
    Args:
        q (bs,num_attention_heads,qryLen,head_dim)
        k (bs,num_key_value_heads,qryLen,head_dim)
        cos (1,qryLen,head_dim)
        sin (1,qryLen,head_dim)
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim) # (1,1,qryLen,head_dim)
    sin = sin.unsqueeze(unsqueeze_dim) # (1,1,qryLen,head_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin) # (bs,num_attention_heads,qryLen,head_dim)
    k_embed = (k * cos) + (rotate_half(k) * sin) # (bs,num_key_value_heads,qryLen,head_dim)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor, # (bs,num_attention_heads,qryLen,head_dim)
    key: torch.Tensor, # (bs,num_key_value_heads,cacheLen+qryLen,head_dim)
    value: torch.Tensor, # (bs,num_key_value_heads,cacheLen+qryLen,head_dim)
    attention_mask: Optional[torch.Tensor], # causal mask (bs,1,qryLen,cacheLen+1+qryLen)
    scaling: float,
    dropout: float = 0.0
):
    key_states = repeat_kv(key, module.num_key_value_groups) # (bs,num_key_value_heads,cacheLen+qryLen,head_dim)
    value_states = repeat_kv(value, module.num_key_value_groups) # (bs,num_key_value_heads,cacheLen+qryLen,head_dim)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling # QK^T/sqrt(head_dim) (bs,num_attention_heads,qryLen,cacheLen+qryLen)
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]] # (bs,1,qryLen,cacheLen+qryLen) 最后一列的负无穷到现在才删掉, 莫非另有隐情?
        attn_weights = attn_weights + causal_mask # (bs,num_attention_heads,qryLen,cacheLen+qryLen)

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states) # (bs,num_attention_heads,qryLen,head_dim)
    attn_output = attn_output.transpose(1, 2).contiguous() # (bs,qryLen,num_attention_heads,head_dim)

    return attn_output


class Qwen2Attention(nn.Module):
    def __init__(self, config: Qwen2Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads # 每个KV负责多少个Q
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=True, dtype=torch.bfloat16)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True, dtype=torch.bfloat16)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True, dtype=torch.bfloat16)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False, dtype=torch.bfloat16)
        # W^Q (hidden_size,hidden_size); W^{KV} (hidden_size,1024) hidden_size=5120 Group Query Attention (GQA) 每个KV负责五个Q
        # W^O (hidden_size,hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor, # (bs,qryLen,hidden_size)
        position_embeddings: Tuple[torch.Tensor, torch.Tensor], # Tuple((1,qryLen,head_dim),(1,qryLen,head_dim))
        attention_mask: Optional[torch.Tensor], # causal mask (bs,1,qryLen,cacheLen+1+qryLen)
        past_key_value: Optional[DynamicCache] = None # DynamicCache
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim) # (bs,qryLen,-1,head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2) # (bs,num_attention_heads,qryLen,head_dim)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2) # (bs,num_key_value_heads,qryLen,head_dim)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2) # (bs,num_key_value_heads,qryLen,head_dim)

        cos, sin = position_embeddings # (1,qryLen,head_dim)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # 和cache里的KV拼接
        # key_states (bs,num_key_value_heads,qryLen,head_dim) -> (bs,num_key_value_heads,cacheLen+qryLen,head_dim)
        # value_states (bs,num_key_value_heads,qryLen,head_dim) -> (bs,num_key_value_heads,cacheLen+qryLen,head_dim)
        if past_key_value is not None:
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx)

        attn_output = eager_attention_forward(
            self,
            query_states, # (bs,num_attention_heads,qryLen,head_dim)
            key_states, # (bs,num_key_value_heads,cacheLen+qryLen,head_dim)
            value_states, # (bs,num_key_value_heads,cacheLen+qryLen,head_dim)
            attention_mask, # causal mask (bs,1,qryLen,cacheLen+1+qryLen)
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling
        ) # (bs,qryLen,num_attention_heads,head_dim)

        attn_output = attn_output.reshape(*input_shape, -1).contiguous() # (bs,qryLen,hidden_size)
        attn_output = self.o_proj(attn_output) # (bs,qryLen,hidden_size)
        return attn_output


class Qwen2RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Qwen2RMSNorm is equivalent to T5LayerNorm
        对于token x=(x1,x2,..,xn), 计算均方根 RMS(x) = sqrt(sum(x^2)/n)
        除以均方根 x'=(x1/RMS,x2/RMS,...,xn/RMS)
        学习一个n大小的向量 g=(g1,g2,...,gn)
        相乘的到最后结果 (x1/RMS*g1,x2/RMS*g2,...,xn/RMS*gn)
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=torch.bfloat16))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        '''
        Input: hidden_states (bs,qryLen,hidden_size)
        Return: normed_hidden_states (bs,qryLen,hidden_size)
        '''
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True) # (bs,qryLen,1) 每个token计算一下自己的均方根
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon) # (bs,qryLen,hidden_size) 乘以倒数平方根
        return self.weight * hidden_states.to(input_dtype)


class Qwen2DecoderLayer(nn.Module):
    def __init__(self, config: Qwen2Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen2Attention(config=config, layer_idx=layer_idx)
        self.mlp = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[DynamicCache] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None  # necessary, but kept here for BC
    ):
        residual = hidden_states # (bs,qryLen,hidden_size)
        hidden_states = self.input_layernorm(hidden_states) # (bs,qryLen,hidden_size)

        # Self Attention
        hidden_states = self.self_attn(
            hidden_states=hidden_states, # (bs,qryLen,hidden_size)
            attention_mask=attention_mask, # causal mask (bs,1,qryLen,cacheLen+1+qryLen)
            past_key_value=past_key_value, # DynamicCache
            position_embeddings=position_embeddings # Tuple((1,qryLen,head_dim),(1,qryLen,head_dim))
        )
        hidden_states = residual + hidden_states # (bs,qryLen,hidden_size)

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


def compute_default_rope_parameters(
    config: Qwen2Config,
    device: Optional["torch.device"] = None
):
    """
    θ_i = 1 / rope_theta^(2i/d), i=0,1,...,d/2-1  d为head_dim

    Computes the inverse frequencies according to the original RoPE implementation
    Returns:
        Tuple of (`torch.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
        post-processing scaling factor applied to the computed cos/sin (unused in this type of RoPE).
    """
    base = config.rope_theta # 1000000
    partial_rotary_factor = config.partial_rotary_factor if hasattr(config, "partial_rotary_factor") else 1.0
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads) # 128
    dim = int(head_dim * partial_rotary_factor) # 128
    attention_factor = 1.0  # Unused in this type of RoPE
    # Compute the inverse frequencies
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).float().to(device) / dim))
    return inv_freq, attention_factor

class Qwen2RotaryEmbedding(nn.Module):
    def __init__(self, config: Qwen2Config, device=None):
        super().__init__()
        self.config = config
        self.inv_freq, self.attention_scaling = compute_default_rope_parameters(self.config, device)

    @torch.no_grad()
    def forward(self, x, position_ids):
        '''
        Args
        ----
        x (bs,qryLen,hidden_size)
        position_ids [[cacheLen,...,cacheLen+qryLen-1]]

        Return
        ------
        sin矩阵和cos矩阵, 大小均为(1,qryLen,head_dim), 第0维和position_ids的第0维有关, 但是如果我们最初没有传入该参数的话就默认是1,
        一二维(qryLen,d)为 (head_dim简写为d)
        sin/cos[[mθ_0,mθ_1,...,mθ_{d/2-1},mθ_0,mθ_1,...,mθ_{d/2-1}],
                [(m+1)θ_0,(m+1)θ_1,...,(m+1)θ_{d/2-1},(m+1)θ_0,(m+1)θ_1,...,(m+1)θ_{d/2-1}],
                ...] 这里m是cacheLen; θ_i=1/rope_theta^(2i/d), i=0,1,...,d/2-1
        '''
        # Core RoPE block
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1) # (d/2,) -> (1,d/2,1)
        position_ids_expanded = position_ids[:, None, :].float() # (1,qryLen) -> (1,1,qryLen)
        # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2) # (1,qryLen,d/2)
            emb = torch.cat((freqs, freqs), dim=-1) # (1,qryLen,d)
            cos = emb.cos()
            sin = emb.sin()
        # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
        cos = cos * self.attention_scaling # (1,qryLen,d)
        sin = sin * self.attention_scaling # (1,qryLen,d)
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class Qwen2Model(nn.Module):
    def __init__(self, config: Qwen2Config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, dtype=torch.bfloat16)
        self.layers = nn.ModuleList(
            [Qwen2DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen2RotaryEmbedding(config=config)
        self.config = config

    def forward(
        self,
        input_ids: torch.LongTensor, # (bs,qryLen)
        attention_mask: Optional[torch.Tensor] = None, # (bs,T) 区分有效输入和填充部分, 有效部分用1标注, 反之用0
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[DynamicCache] = None,
        use_cache: Optional[bool] = None, # 是否使用KVcache, None则查询config文件
        cache_position: Optional[torch.LongTensor] = None
    ):
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        inputs_embeds = self.embed_tokens(input_ids) # (bs,qryLen) -> (bs,qryLen,hidden_size)

        if use_cache and past_key_values is None: # 如果使用kvcache但是没有传入已有的kvcache, 则创建一个新的
            past_key_values = DynamicCache()

        if cache_position is None: # [cacheLen,...,cacheLen+qryLen-1] input_ids中每个元素位于原输入的哪个位置
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None: # [[cacheLen,...,cacheLen+qryLen-1]]
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask( # 因果掩码 预测当前token时不能看到未来token
            attention_mask, inputs_embeds, cache_position, past_key_values
        ) # (bs,1,qryLen,cacheLen+1+qryLen)

        hidden_states = inputs_embeds # (bs,qryLen,hidden_size)

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids) # Tuple((1,qryLen,head_dim),(1,qryLen,head_dim))

        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states, # (bs,qryLen,hidden_size)
                attention_mask=causal_mask, # (bs,1,qryLen,cacheLen+1+qryLen)
                past_key_value=past_key_values, # DynamicCache
                position_embeddings=position_embeddings # Tuple((1,qryLen,head_dim),(1,qryLen,head_dim))
            )
        hidden_states = self.norm(hidden_states)
        return hidden_states, past_key_values

    def _update_causal_mask(
        self,
        attention_mask: Optional[torch.Tensor], # (bs,attLen) or None
        input_tensor: torch.Tensor, # (bs,qryLen,hidden_size)
        cache_position: torch.Tensor,
        past_key_values: Optional[DynamicCache] # KV cache
    ): 
        '''
        Return: causal_mask (bs,1,qryLen,cacheLen+1+qryLen)
        '''
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        dtype, device = input_tensor.dtype, input_tensor.device
        sequence_length = input_tensor.shape[1] # qryLen
        target_length = past_seen_tokens + sequence_length + 1 # cacheLen + qryLen + 1

        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask, # (bs,attLen) or None
            sequence_length=sequence_length, # qryLen
            target_length=target_length, # cacheLen + qryLen + 1
            dtype=dtype,
            device=device,
            cache_position=cache_position, # [cacheLen,...,cacheLen+qryLen-1]
            batch_size=input_tensor.shape[0]
        )
        return causal_mask

    @staticmethod
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: Optional[torch.Tensor], # (bs,attLen) or None
        sequence_length: int, # qryLen
        target_length: int, # cacheLen + qryLen + 1
        dtype: torch.dtype,
        device: torch.device,
        cache_position: torch.Tensor, # [cacheLen,...,cacheLen+qryLen-1]
        batch_size: int
    ):
        '''
        Return: causal_mask (bs,1,qryLen,cacheLen+1+qryLen)
        '''
        min_dtype = torch.finfo(dtype).min # 负无穷
        causal_mask = torch.full( # (qryLen, cacheLen+1+qryLen)
            (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device
        )
        # diagonal_attend_mask (qryLen,cacheLen+1+qryLen)
        # [[False*cacheLen个, False,  True,  True,  True] 这里qryLen=3
        #  [False*cacheLen个, False, False,  True,  True] True看不到 False看得到
        #  [False*cacheLen个, False, False, False,  True]]
        diagonal_attend_mask = torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
        causal_mask *= diagonal_attend_mask # False部分置0 True为负无穷
        causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1) # (bs,1,qryLen,cacheLen+1+qryLen)
        if attention_mask is not None:
            causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
            if attention_mask.shape[-1] > target_length:
                attention_mask = attention_mask[:, :target_length]
            mask_length = attention_mask.shape[-1] # 支持传入attention_mask比target_length短
            padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :].to(
                causal_mask.device
            )
            padding_mask = padding_mask == 0
            # causal_mask[0][0] (qryLen,cacheLen+1+qryLen)这里qryLen=3 且该样本的padding长度n小于kvcache缓存数目 否则左边负无穷要继续侵占右边的0
            # [[-∞*n个,0*(cacheLen-n)个, 0, -∞, -∞, -∞] # -∞表示看不到
            #  [-∞*n个,0*(cacheLen-n)个, 0,  0, -∞, -∞]
            #  [-∞*n个,0*(cacheLen-n)个, 0,  0,  0, -∞]]
            causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                padding_mask, min_dtype
            )
        return causal_mask


class Qwen2ForCausalLM(nn.Module):
    def __init__(self, config: Qwen2Config):
        super().__init__()
        self.model = Qwen2Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False, dtype=torch.bfloat16)

    def forward(
        self,
        input_ids: torch.LongTensor, # (bs,qryLen) 数值都是0~vocab_size-1的正整数
        attention_mask: Optional[torch.Tensor] = None, # (bs,T) 区分有效输入和填充部分, 有效部分用1标注, 反之用0 注意paddingside为左边
        position_ids: Optional[torch.LongTensor] = None, # 不知道传入这个有啥用, 可以通过kvcache来算啊
        past_key_values: Optional[Union[DynamicCache, List[torch.FloatTensor]]] = None, # kvcache
        use_cache: Optional[bool] = None, # 是否使用KVcache, None则查询config文件
        cache_position: Optional[torch.LongTensor] = None, # 不知道传入这个有啥用, 可以通过kvcache来算啊
        logits_to_keep:int = 1 # 保留多少个logits
    ):
        '''
        Returns
        -------
        logits: torch.Tensor
            最后logits_to_keep个token的logits (bs,logits_to_keep,vocab_size)
        past_key_values: Optional[DynamicCache]
            若use_cache则KV cache, 不然为None
        '''
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position
        ) # (hidden_states, past_key_values)
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states[:, -logits_to_keep:, :]) # (bs,qryLen,hidden_size) -> (bs,logits_to_keep,hidden_size) -> (bs,logits_to_keep,vocab_size)
        return logits, outputs[1]