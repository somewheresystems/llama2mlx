import math
import mlx.core as mx
import mlx.nn as nn

class Config:
    def __init__(self, dim, multiple_of, n_heads, n_layers, norm_eps, vocab_size, model_type, quantization):
        self.dim = dim
        self.multiple_of = multiple_of
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.norm_eps = norm_eps
        self.vocab_size = vocab_size
        self.model_type = model_type
        self.quantization = quantization

class LlamaAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int):
        super().__init__()

        self.n_heads = n_heads

        self.rope = nn.RoPE(dim // n_heads, traditional=True)
        self.query_proj = nn.Linear(dim, dim, bias=False)
        self.key_proj = nn.Linear(dim, dim, bias=False)
        self.value_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def __call__(self, queries, keys, values, mask=None, cache=None):
        queries = self.query_proj(queries)
        keys = self.key_proj(keys)
        values = self.value_proj(values)

        n_heads = self.n_heads
        B, L, D = queries.shape

        queries = queries.reshape(B, L, n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, n_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, n_heads, -1).transpose(0, 2, 1, 3)

        if cache is not None:
            key_cache, value_cache = cache
            queries = self.rope(queries, offset=key_cache.shape[2])
            keys = self.rope(keys, offset=key_cache.shape[2])
            keys = mx.concatenate([key_cache, keys], axis=2)
            values = mx.concatenate([value_cache, values], axis=2)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        scale = math.sqrt(1 / queries.shape[-1])
        scores = (queries * scale) @ keys.transpose(0, 1, 3, 2)
        if mask is not None:
            scores = scores + mask
        scores = mx.softmax(scores, axis=-1)
        values_hat = (scores @ values).transpose(0, 2, 1, 3).reshape(B, L, -1)

        return self.out_proj(values_hat), (keys, values)

class LlamaEncoderLayer(nn.Module):
    def __init__(self, dim: int, multiple_of: int, n_heads: int):
        super().__init__()

        self.attention = LlamaAttention(dim, n_heads)

        self.norm1 = nn.RMSNorm(dim)
        self.norm2 = nn.RMSNorm(dim)

        self.linear1 = nn.Linear(dim, multiple_of, bias=False)
        self.linear2 = nn.Linear(dim, multiple_of, bias=False)
        self.linear3 = nn.Linear(multiple_of, dim, bias=False)

    def __call__(self, x, mask=None, cache=None):
        y = self.norm1(x)
        y, cache = self.attention(y, y, y, mask, cache)
        x = x + y

        y = self.norm2(x)
        a = self.linear1(y)
        b = self.linear2(y)
        y = a * mx.sigmoid(a) * b
        y = self.linear3(y)
        x = x + y

        return x, cache

class Llama(nn.Module):
    def __init__(self, n_layers: int, vocab_size: int, dim: int, multiple_of: int, n_heads: int):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, dim)
        self.layers = [LlamaEncoderLayer(dim, multiple_of, n_heads) for _ in range(n_layers)]
        self.norm = nn.RMSNorm(dim)
        self.out_proj = nn.Linear(dim, vocab_size, bias=False)

    def __call__(self, x):
        mask = nn.MultiHeadAttention.create_additive_causal_mask(x.shape[1])
        mask = mask.astype(self.embedding.weight.dtype)

        x = self.embedding(x)
        for l in self.layers:
            x, _ = l(x, mask)
        x = self.norm(x)
        return self.out_proj(x)

    def generate(self, x, temp=1.0):
        cache = []

        mask = nn.MultiHeadAttention.create_additive_causal_mask(x.shape[1])
        mask = mask.astype(self.embedding.weight.dtype)

        x = self.embedding(x)
        for l in self.layers:
            x, c = l(x, mask=mask)
            cache.append(c)
        x = self.norm

