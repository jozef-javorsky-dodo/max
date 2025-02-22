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

from max.dtype import DType
from max.graph import TensorValue, TensorValueLike, ops

from ..attention import NaiveAttentionWithRope
from ..embedding import Embedding, EmbeddingV2
from ..layer import Layer, LayerList, LayerV2
from ..linear import Linear, LinearV2


class NaiveTransformerBlock(LayerV2):
    """Max-Graph Only Stack of Attention, FeedForward, and RMSNorm layers."""

    def __init__(
        self,
        attention: NaiveAttentionWithRope,
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
        attention_mask: TensorValueLike,
        k_cache: TensorValueLike,
        v_cache: TensorValueLike,
        start_pos: TensorValue,
        layer_index: int,
    ) -> tuple[TensorValue, TensorValue, TensorValue]:
        attention_out = self.self_attn(
            self.input_layernorm(x),
            attention_mask,
            k_cache,  # type: ignore
            v_cache,  # type: ignore
            start_pos,
            layer_index,
        )

        h = x + attention_out
        h = h + self.mlp(self.post_attention_layernorm(h))

        return h  # type: ignore


class NaiveTransformer(LayerV2):
    """Max-Graph only model consisting of NaiveTransformerBlock layers."""

    def __init__(
        self,
        dim: int,
        n_heads: int,
        layers: list[NaiveTransformerBlock],
        norm: Layer,
        output: Linear | LinearV2,
        theta: float,
        embedding: Embedding | EmbeddingV2,
        output_type: DType | None = None,
    ):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.layers = LayerList(layers)
        self.norm = norm
        self.lm_head = output
        self.theta = theta
        self.embed_tokens = embedding
        self.output_type = output_type

    def __call__(
        self,
        tokens: TensorValueLike,
        attention_mask: TensorValueLike,
        k_cache: TensorValueLike,
        v_cache: TensorValueLike,
        start_pos: TensorValueLike,
    ) -> tuple[TensorValue]:
        h = self.embed_tokens(tokens)

        for i in range(len(self.layers)):
            h = self.layers[i](
                h,
                attention_mask,
                k_cache,
                v_cache,
                start_pos,
                i,
            )

        output = self.lm_head(self.norm(h))
        if self.output_type is not None:
            casted_output = ops.cast(output, self.output_type)
            return (casted_output,)
        else:
            return (output,)
