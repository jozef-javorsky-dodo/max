# ===----------------------------------------------------------------------=== #
# Copyright (c) 2024, Modular Inc. All rights reserved.
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
"""General interface for Attention."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Union

from max.graph import TensorValue, TensorValueLike
from max.pipelines.kv_cache import (
    ContinuousBatchingKVCacheCollection,
    KVCacheParams,
    PagedKVCacheCollection,
)

from ..layer import Layer
from ..linear import Linear


@dataclass
class AttentionImpl(ABC, Layer):
    """
    A generalized attention interface, that will be used upstream by a general Transformer.
    We would expect a seperate subclass, articulating each variation of Attention:
        - AttentionWithRope
        - AttentionWithAlibi
        - VanillaAttentionWithCausalMask
        - ...

    There are a series of shared attributes, however, more may be needed for each individual variant.
    For example, we may introduce an OptimizedRotaryEmbedding class for the AttentionWithRope class:

    @dataclass
    class AttentionWithRope(AttentionImpl):
        rope: OptimizedRotaryEmbedding
        ...

    We expect the `__call__` abstractmethod to remain relatively consistent, however the **kwargs
    argument is exposed, allowing you to leverage additional arguments for each particular variant.
    For example, we may introduce an VanillaAttentionWithCausalMask class, which includes an attention
    mask:

    @dataclass
    class VanillaAttentionWithCausalMask(AttentionImpl):
        ...

        def __call__(
            self,
            x: TensorValueLike,
            kv_collection: ContinuousBatchingKVCacheCollection,
            valid_lengths: TensorValueLike,
            **kwargs,
        ) -> tuple[TensorValue, ContinuousBatchingKVCacheCollection]: ...

            if "attn_mask" not in kwargs:
                raise ValueError("attn_mask not provided to VanillaAttentionWithCausalMask")

            # Which we can then use the attention mask downstream like so:
            op(
                attn_mask = kwargs["attn_mask"]
            )
    """

    n_heads: int
    """The number of attention heads."""

    kv_params: KVCacheParams
    """KV Cache Params, including the number of kv heads, the head dim, and data type."""

    layer_idx: TensorValue
    """The layer number associated with this Attention block."""

    wqkv: TensorValue
    """The concatenation of q, k, and v weight vectors."""

    wo: Linear
    """A linear layer for the output projection."""

    def __post_init__(self) -> None:
        if not self.kv_params.cache_strategy.uses_opaque():
            raise ValueError(
                f"{self.kv_params.cache_strategy} cache strategy, not supported"
                " in Attention layer."
            )

    @abstractmethod
    def __call__(
        self,
        x: TensorValue,
        kv_collection: Union[
            ContinuousBatchingKVCacheCollection, PagedKVCacheCollection
        ],
        **kwargs,
    ) -> TensorValue: ...


@dataclass
class DistributedAttentionImpl(ABC, Layer):
    """
    A generalized Distributed attention interface.
    """

    @abstractmethod
    def __call__(
        self,
        x: list[TensorValue],
        kv_collections: list[ContinuousBatchingKVCacheCollection],
        **kwargs,
    ) -> list[TensorValue]: ...


@dataclass
class AttentionImplQKV(ABC, Layer):
    """
    A generalized attention interface, that will be used upstream by a general Transformer.
    We would expect a seperate subclass, articulating each variation of Attention:
        - AttentionWithRope
        - AttentionWithAlibi
        - VanillaAttentionWithCausalMask
        - ...

    There are a series of shared attributes, however, more may be needed for each individual variant.
    For example, we may introduce an OptimizedRotaryEmbedding class for the AttentionWithRope class:

    @dataclass
    class AttentionWithRope(AttentionImpl):
        rope: OptimizedRotaryEmbedding
        ...

    We expect the `__call__` abstractmethod to remain relatively consistent, however the **kwargs
    argument is exposed, allowing you to leverage additional arguments for each particular variant.
    For example, we may introduce an VanillaAttentionWithCausalMask class, which includes an attention
    mask:

    @dataclass
    class VanillaAttentionWithCausalMask(AttentionImpl):
        ...

        def __call__(
            self,
            x: TensorValueLike,
            kv_collection: ContinuousBatchingKVCacheCollection,
            valid_lengths: TensorValueLike,
            **kwargs,
        ) -> tuple[TensorValue, ContinuousBatchingKVCacheCollection]: ...

            if "attn_mask" not in kwargs:
                raise ValueError("attn_mask not provided to VanillaAttentionWithCausalMask")

            # Which we can then use the attention mask downstream like so:
            op(
                attn_mask = kwargs["attn_mask"]
            )
    """

    n_heads: int
    """The number of attention heads."""

    kv_params: KVCacheParams
    """KV Cache Params, including the number of kv heads, the head dim, and data type."""

    layer_idx: int
    """The layer number associated with this Attention block."""

    wq: TensorValueLike
    """The q weight vector."""

    wk: TensorValueLike
    """The k weight vector."""

    wv: TensorValueLike
    """The v weight vector."""

    wo: Linear
    """A linear layer for the output projection."""

    def __post_init__(self) -> None:
        if not self.kv_params.cache_strategy.uses_opaque():
            raise ValueError(
                f"{self.kv_params.cache_strategy} cache strategy, not supported"
                " in Attention layer."
            )

    @abstractmethod
    def __call__(
        self,
        x: TensorValue,
        kv_collection: Union[
            ContinuousBatchingKVCacheCollection, PagedKVCacheCollection
        ],
        **kwargs,
    ) -> TensorValue: ...
