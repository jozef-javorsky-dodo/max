# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

from collections.abc import Sequence

from max.dtype import DType
from max.graph import TensorValue, TensorValueLike, ops
from max.pipelines.kv_cache import (
    ContinuousBatchingKVCacheCollection,
    FetchContinuousBatchingKVCacheCollection,
    FetchPagedKVCacheCollection,
    KVCacheParams,
    PagedKVCacheCollection,
)

from ..attention.interfaces import (
    AttentionImpl,
    AttentionImplQKV,
    AttentionImplV2,
)
from ..embedding import Embedding, EmbeddingV2
from ..layer import Layer, LayerList, LayerV2
from ..linear import Linear, LinearV2


class TransformerBlock(LayerV2):
    """Stack of Attention, FeedForward, and RMSNorm layers."""

    def __init__(
        self,
        attention: AttentionImpl | AttentionImplV2 | AttentionImplQKV,
        mlp: Layer,
        attention_norm: Layer,
        mlp_norm: Layer,
    ):
        super().__init__()
        self.self_attn = attention
        self.mlp = mlp
        self.input_layernorm = attention_norm
        self.post_attention_layernorm = mlp_norm

    def __call__(
        self,
        x: TensorValue,
        kv_collection: ContinuousBatchingKVCacheCollection
        | PagedKVCacheCollection,
        **kwargs,
    ) -> TensorValue:
        attn_out = self.self_attn(
            self.input_layernorm(x),
            kv_collection,
            **kwargs,
        )

        h = x + attn_out
        h = h + self.mlp(self.post_attention_layernorm(h))

        return h


class Transformer(LayerV2):
    """Transformer model consisting for TransformerBlock layers."""

    def __init__(
        self,
        dim: int,
        n_heads: int,
        layers: list[TransformerBlock],
        norm: Layer,
        output: Linear | LinearV2,
        embedding: Embedding | EmbeddingV2,
        kv_params: KVCacheParams,
        kv_collection_constructor: (
            FetchContinuousBatchingKVCacheCollection
            | FetchPagedKVCacheCollection
        ),
        all_logits: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.layers = LayerList(layers)
        self.norm = norm
        self.lm_head = output
        self.embed_tokens = embedding
        self.kv_params = kv_params
        self.kv_collection_constructor = kv_collection_constructor
        self.all_logits = all_logits

    def __call__(
        self,
        tokens: TensorValueLike,
        kv_cache_inputs: Sequence[TensorValue],
        **kwargs,
    ) -> tuple[TensorValue, ...]:
        # TODO: Split into a ragged and non-ragged version.
        h = self.embed_tokens(tokens)

        kv_collection = self.kv_collection_constructor(*kv_cache_inputs)

        for _, layer in enumerate(self.layers):
            h = layer(h, kv_collection, **kwargs)

        normalized = self.norm(h)

        if "input_row_offsets" in kwargs:
            # Ragged inputs/activations
            last_indices = kwargs["input_row_offsets"][1:] - 1
            last_tokens = ops.gather(normalized, last_indices, axis=0)
        else:
            # Dense padded inputs/activations
            valid_lengths = kwargs["valid_lengths"]
            # TODO: Remove once `gather_nd` works with nonstatic last dims.
            indices = ops.unsqueeze(valid_lengths - 1, -1)
            last_tokens = ops.gather_nd(normalized, indices, batch_dims=1)

        # Always return float32 logits, no matter the activation type.
        last_token_logits = ops.cast(self.lm_head(last_tokens), DType.float32)

        if self.all_logits:
            all_logits = ops.cast(self.lm_head(normalized), DType.float32)
            return (last_token_logits, all_logits)

        return (last_token_logits,)
