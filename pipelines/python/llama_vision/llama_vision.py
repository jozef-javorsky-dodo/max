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

from __future__ import annotations

import logging
import time
from collections.abc import Sequence

import numpy as np
from max.driver import Tensor
from max.dtype import DType
from max.engine import InferenceSession, MultimodalModel
from max.graph import Dim, Graph, TensorType, TensorValue, ops
from max.graph.weights import Weights
from max.pipelines import (
    ModelOutputs,
    PipelineConfig,
    PipelineModel,
    TextAndVisionContext,
)
from max.pipelines.kv_cache import (
    KVCacheManager,
    KVCacheParams,
    KVCacheStrategy,
    estimate_kv_cache_size,
    load_kv_manager,
)
from nn import Linear
from nn.layer import Layer

from .language_model import instantiate_language_model
from .vision_model import instantiate_vision_model


def max_seq_len(config: PipelineConfig) -> int:
    return min(
        config.max_length,
        config.huggingface_config.text_config.max_position_embeddings,
    )


class LlamaVisionModel(Layer):
    """
    The Llama 3.2 vision model.
    """

    def __init__(
        self, pipeline_config: PipelineConfig, weights: Weights
    ) -> None:
        # Set convenience attributes for the text and vision configs.
        self.vision_config = pipeline_config.huggingface_config.vision_config
        self.text_config = pipeline_config.huggingface_config.text_config

        self.vision_model = instantiate_vision_model(
            dtype=pipeline_config.dtype,
            image_size=self.vision_config.image_size,
            patch_size=self.vision_config.patch_size,
            supported_aspect_ratios=self.vision_config.supported_aspect_ratios,
            hidden_size=self.vision_config.hidden_size,
            max_num_tiles=self.vision_config.max_num_tiles,
            num_channels=self.vision_config.num_channels,
            norm_eps=self.vision_config.norm_eps,
            attention_heads=self.vision_config.attention_heads,
            num_hidden_layers=self.vision_config.num_hidden_layers,
            intermediate_size=self.vision_config.intermediate_size,
            num_global_layers=self.vision_config.num_global_layers,
            intermediate_layers_indices=self.vision_config.intermediate_layers_indices,
            weights=weights,
        )

        self.multi_modal_projector = Linear(
            weights.multi_modal_projector.weight.allocate(
                pipeline_config.dtype,
                [
                    self.text_config.hidden_size,
                    self.vision_config.vision_output_dim,
                ],
            ),
            weights.multi_modal_projector.bias.allocate(
                pipeline_config.dtype,
                [self.text_config.hidden_size],
            ),
        )

    def __call__(
        self,
        pixel_values: TensorValue,
        aspect_ratio_ids: TensorValue,
        aspect_ratio_mask: TensorValue,
    ) -> TensorValue:
        if aspect_ratio_ids is None:
            msg = (
                "`aspect_ratio_ids` must be provided if `pixel_values` is "
                "provided"
            )
            raise ValueError(msg)

        # Get vision tokens from vision model.
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            aspect_ratio_ids=aspect_ratio_ids,
            aspect_ratio_mask=aspect_ratio_mask,
        )
        cross_attention_states = vision_outputs[0]

        num_patches = cross_attention_states.shape[-2]

        cross_attention_states = self.multi_modal_projector(
            cross_attention_states
        ).reshape(
            [
                Dim("batch_size")
                * Dim("num_concurrent_media")
                * self.vision_config.max_num_tiles
                * num_patches,
                self.text_config.hidden_size,
            ]
        )

        return cross_attention_states


class LlamaVisionLanguageModel(Layer):
    """
    The Llama 3.2 vision language model.
    """

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        weights: Weights,
        kv_params: KVCacheParams,
    ) -> None:
        text_config = pipeline_config.huggingface_config.text_config

        self.language_model = instantiate_language_model(
            dtype=pipeline_config.dtype,
            hidden_size=text_config.hidden_size,
            n_heads=text_config.num_attention_heads,
            rope_theta=text_config.rope_theta,
            max_seq_len=max_seq_len(pipeline_config),
            num_hidden_layers=text_config.num_hidden_layers,
            cross_attention_layers=text_config.cross_attention_layers,
            vocab_size=text_config.vocab_size,
            rms_norm_eps=text_config.rms_norm_eps,
            num_key_value_heads=text_config.num_key_value_heads,
            intermediate_size=text_config.intermediate_size,
            kv_params=kv_params,
            weights=weights,
        )

    def __call__(
        self,
        cross_attention_states: TensorValue,
        input_ids: TensorValue,
        hidden_input_row_offsets: TensorValue,
        cross_input_row_offsets: TensorValue,
        *kv_cache_inputs: TensorValue,
    ) -> TensorValue:
        logits = self.language_model(
            kv_cache_inputs=kv_cache_inputs,
            input_ids=input_ids,
            hidden_input_row_offsets=hidden_input_row_offsets,
            cross_attention_states=cross_attention_states,
            cross_input_row_offsets=cross_input_row_offsets,
        )
        # Always return float32 logits, no matter the activation type
        return ops.cast(logits, DType.float32)


class LlamaVision(PipelineModel):
    """The entire (multimodal) Llama3.2 vision model."""

    def __init__(
        self, pipeline_config: PipelineConfig, session: InferenceSession
    ) -> None:
        # Set convenience attributes for the text and vision configs.
        self.vision_config = pipeline_config.huggingface_config.vision_config
        self.text_config = pipeline_config.huggingface_config.text_config

        # These need to be set at graph instantiation time.
        self.vision_graph_input_size = -1
        self.language_graph_input_size = -1

        super().__init__(pipeline_config, session)

    def _llama3_vision_vision_graph(self) -> Graph:
        # Inserted a manual CHW -> HWC transpose here.
        pixel_values_type = TensorType(
            self.pipeline_config.dtype,
            shape=[
                "batch_size",
                "num_concurrent_media",
                self.vision_config.max_num_tiles,
                self.vision_config.image_size,  # height
                self.vision_config.image_size,  # width
                self.vision_config.num_channels,
            ],
        )
        aspect_ratio_ids_type = TensorType(
            DType.int64,
            shape=["batch_size", "num_concurrent_media"],
        )
        aspect_ratio_mask_type = TensorType(
            DType.int64,
            shape=[
                "batch_size",
                "num_concurrent_media",
                self.vision_config.max_num_tiles,
            ],
        )

        input_types = [
            pixel_values_type,
            aspect_ratio_ids_type,
            aspect_ratio_mask_type,
        ]
        self.vision_graph_input_size = len(input_types)
        return Graph(
            "llama3-vision-vision-model-graph",
            forward=LlamaVisionModel(
                pipeline_config=self.pipeline_config, weights=self.weights
            ),
            input_types=input_types,
        )

    def _llama3_vision_language_graph(self) -> Graph:
        # Pre-allocate a buffer for input_row_offsets in multistep execution.
        # We do this to avoid materializing and copying a buffer with each multistep step
        self._input_row_offsets_prealloc = Tensor.from_numpy(
            np.arange(
                self.pipeline_config.max_cache_batch_size + 1, dtype=np.uint32
            )
        ).to(self.pipeline_config.devices[0])

        input_ids_type = TensorType(DType.int64, shape=["total_seq_len"])
        # image_size = self.vision_config.image_size
        # patch_size = self.vision_config.patch_size
        cross_attention_states_type = TensorType(
            self.pipeline_config.dtype,
            shape=[
                # TODO(bduke): fix algebraic dim creation outside of graph
                # contexts.
                # Dim("batch_size")
                # * "num_concurrent_media"
                # * self.vision_config.max_num_tiles
                # * ((image_size // patch_size) ** 2 + 1),
                "num_vision_embeddings",
                self.text_config.hidden_size,
            ],
        )
        input_row_offsets_type = TensorType(
            DType.uint32, shape=["input_row_offsets_len"]
        )

        input_types = [
            cross_attention_states_type,
            input_ids_type,
            # Pass 2 sets of offsets for hidden and cross attn states.
            input_row_offsets_type,
            input_row_offsets_type,
            *self.kv_manager.input_symbols()[0],
        ]
        self.language_graph_input_size = len(input_types)

        return Graph(
            "llama3-vision-language-model-graph",
            forward=LlamaVisionLanguageModel(
                pipeline_config=self.pipeline_config,
                weights=self.weights,
                kv_params=self._get_kv_params(),
            ),
            input_types=input_types,
        )

    def prepare_initial_token_inputs(
        self,
        context_batch: Sequence[TextAndVisionContext],  # type: ignore
    ) -> tuple[Tensor, ...]:
        """Creates tensors of token and image inputs, if applicable."""
        if self.pipeline_config.cache_strategy != KVCacheStrategy.CONTINUOUS:
            msg = "Llama Vision only supports continuous batching"
            raise ValueError(msg)

        # Marshal out hyperparameters.
        batch_size = len(context_batch)
        height = self.vision_config.image_size
        width = self.vision_config.image_size
        num_channels = self.vision_config.num_channels
        max_num_tiles = self.vision_config.max_num_tiles
        patch_size = self.vision_config.patch_size
        # TODO(bduke): account for the actual instead of max number of tiles.
        image_seq_len = max_num_tiles * (height * width) // patch_size**2

        pixel_values = Tensor.zeros(
            shape=[batch_size, 1, max_num_tiles, height, width, num_channels],
            dtype=self.pipeline_config.dtype,
        ).to(self.pipeline_config.device)
        aspect_ratio_ids = Tensor.zeros(
            shape=[batch_size, 1], dtype=DType.int64
        ).to(self.pipeline_config.device)
        aspect_ratio_mask = Tensor.zeros(
            shape=[batch_size, 1, max_num_tiles], dtype=DType.int64
        ).to(self.pipeline_config.device)

        # Input row offset type: ["input_row_offsets_len"], UInt32
        input_id_row_offsets = Tensor.from_numpy(
            np.cumsum(
                [0] + [ctx.seq_len for ctx in context_batch],
                dtype=np.uint32,
            )
        ).to(self.pipeline_config.device)

        pixel_row_offsets = Tensor.from_numpy(
            np.cumsum(
                [0]
                + [
                    # Use an input row offset of 0 to mean no image.
                    image_seq_len if ctx.pixel_values is not None else 0
                    for ctx in context_batch
                ],
                dtype=np.uint32,
            )
        ).to(self.pipeline_config.device)

        # Input Ids: ["total_seq_len"], Int64
        # Create a ragged token vector of length: sum(len(t) for t in tokens).
        tokens = np.concatenate([ctx.next_tokens for ctx in context_batch])
        input_id_values = Tensor.from_numpy(tokens).to(
            self.pipeline_config.device
        )

        return (
            pixel_values,
            aspect_ratio_ids,
            aspect_ratio_mask,
            input_id_values,
            pixel_row_offsets,
            input_id_row_offsets,
        )

    def prepare_next_token_inputs(
        self,
        next_tokens: Tensor,
        prev_model_inputs: tuple[Tensor, ...],
    ) -> tuple[Tensor, ...]:
        # Next token inputs always go to the language model.
        # - input ids
        # - input max seq lengths
        # - hidden input row offsets
        input_id_max_seq_len: Tensor
        old_row_offsets: Tensor
        if len(prev_model_inputs) == 7:
            # If the previous inputs include the pixel values
            input_id_max_seq_len = prev_model_inputs[4]
            old_row_offsets = prev_model_inputs[6]
        else:
            # If no pixel values were included
            assert len(prev_model_inputs) == 3
            input_id_max_seq_len = prev_model_inputs[1]
            old_row_offsets = prev_model_inputs[2]
        row_offsets_size = old_row_offsets.shape[0]
        next_row_offsets = self._input_row_offsets_prealloc[:row_offsets_size]
        next_token_inputs = (
            next_tokens,
            input_id_max_seq_len,
            next_row_offsets,
        )
        return next_token_inputs

    def execute(self, *model_inputs: Tensor) -> ModelOutputs:
        assert isinstance(
            self.model.modalities,  # type: ignore
            tuple,
        ), "Modalities must be a tuple"
        assert (
            len(self.model.modalities) == 2  # type: ignore
        ), "Must have only vision and language modalities"
        vision_model, language_model = self.model.modalities  # type: ignore

        model_input_list = list(model_inputs)

        # Vision model has 3 more inputs.
        # pixel_values(1), aspect_ratio_ids(1), aspect_ratio_mask(1)
        if len(model_input_list) >= self.vision_graph_input_size:
            cross_attention_states = vision_model.execute(
                *model_input_list[: self.vision_graph_input_size],
                copy_inputs_to_device=False,
            )[0]
            model_input_list = model_input_list[self.vision_graph_input_size :]

        # Insert vision model output to be fed as input to the subsequent
        # language model. This assumes cross_attention_states is the first input
        # since the list needs to be ordered.
        model_input_list.insert(0, cross_attention_states)  # type: ignore

        # Language model has 8 inputs.
        # kv_cache_inputs (4), input_ids(1), hidden_input_row_offsets(1),
        # cross_attention_states(1), cross_input_row_offsets(1)
        if len(model_input_list) != self.language_graph_input_size:
            raise ValueError(
                "Expecting language_model inputs to have {}, got {} instead".format(
                    self.language_graph_input_size, len(model_input_list)
                )
            )

        model_outputs = language_model.execute(
            *model_input_list, copy_inputs_to_device=False
        )
        assert not self.pipeline_config.enable_echo
        assert isinstance(model_outputs[0], Tensor)
        return ModelOutputs(next_token_logits=model_outputs[0])

    def _get_kv_params(self) -> KVCacheParams:
        return KVCacheParams(
            dtype=self.pipeline_config.dtype,
            n_kv_heads=self.text_config.num_key_value_heads,
            head_dim=(
                self.text_config.hidden_size
                // self.text_config.num_attention_heads
            ),
            cache_strategy=self.pipeline_config.cache_strategy,
        )

    def load_kv_manager(
        self,
        session: InferenceSession,
        available_cache_memory: int,
    ) -> KVCacheManager:
        return load_kv_manager(
            params=self._get_kv_params(),
            max_cache_batch_size=self.pipeline_config.max_cache_batch_size,
            max_seq_len=max_seq_len(self.pipeline_config),
            num_layers=self.text_config.num_hidden_layers,
            devices=self.pipeline_config.devices,
            available_cache_memory=available_cache_memory,
            page_size=self.pipeline_config.kv_cache_page_size,
            session=session,
        )

    def estimate_kv_cache_size(self, available_cache_memory: int) -> int:
        return estimate_kv_cache_size(
            params=self._get_kv_params(),
            max_cache_batch_size=self.pipeline_config.max_cache_batch_size,
            max_seq_len=max_seq_len(self.pipeline_config),
            num_layers=self.text_config.num_hidden_layers,
            available_cache_memory=available_cache_memory,
            devices=self.pipeline_config.devices,
        )

    def load_model(
        self,
        session: InferenceSession,
    ) -> MultimodalModel:
        """
        Load the Llama vision multimodal model. Since this is a multimodal model,
        we have vision and language models (graph) loaded.
        """
        self.weights = self.pipeline_config.load_weights()

        logging.info("Building vision model...")
        vision_model_graph = self._llama3_vision_vision_graph()
        logging.info("Compiling...")
        before = time.perf_counter()
        vision_model = session.load(
            vision_model_graph,
            weights_registry=self.weights.allocated_weights,
        )
        after = time.perf_counter()
        logging.info(
            f"Compiling vision model took {after - before:.6f} seconds"
        )

        logging.info("Building language model...")
        language_model_graph = self._llama3_vision_language_graph()
        logging.info("Compiling...")
        before = time.perf_counter()
        language_model = session.load(
            language_model_graph,
            weights_registry=self.weights.allocated_weights,
        )
        after = time.perf_counter()
        logging.info(
            f"Compiling language model took {after - before:.6f} seconds"
        )
        return MultimodalModel((vision_model, language_model))
