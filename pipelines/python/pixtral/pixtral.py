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

import logging
import time
from typing import Sequence, cast

import numpy as np
from max.driver import Tensor
from max.engine import InferenceSession, Model
from max.graph.weights import SafetensorWeights
from max.pipelines import (
    ModelInputs,
    ModelOutputs,
    PipelineConfig,
    PipelineModel,
    TextAndVisionContext,
)
from max.pipelines.kv_cache import (
    KVCacheManager,
    KVCacheParams,
    estimate_kv_cache_size,
    load_kv_manager,
)

from .model.graph import _build_text_graph, _build_vision_graph
from .vision_encoder.attention_utils import causal_attention_mask_2d_from_imgs


class PixtralInputs(ModelInputs):
    """Holds inputs for the Pixtral model."""

    input_ids: Tensor
    input_row_offsets: Tensor

    # Image inputs
    _pixel_values: Tensor | None = None
    _attention_mask: Tensor | None = None

    def __init__(
        self,
        input_ids: Tensor,
        input_row_offsets: Tensor,
        pixel_values: Tensor | None = None,
        attention_mask: Tensor | None = None,
    ):
        self.input_ids = input_ids
        self.input_row_offsets = input_row_offsets
        self._pixel_values = pixel_values
        self._attention_mask = attention_mask

    @property
    def has_vision_inputs(self) -> bool:
        """Returns true iff this includes vision model inputs."""
        return self._pixel_values is not None

    @property
    def pixel_values(self) -> Tensor:
        assert self._pixel_values is not None
        return self._pixel_values

    @property
    def attention_mask(self) -> Tensor:
        assert self._attention_mask is not None
        return self._attention_mask


class PixtralModel(PipelineModel):
    """The overall interface to the Pixtral model."""

    def __init__(
        self, pipeline_config: PipelineConfig, session: InferenceSession
    ) -> None:
        super().__init__(pipeline_config, session)
        self.vision_model, self.language_model = self.load_model(session)
        # Note that in a multimodal model, the language model is the last model in the
        # pipeline. Unfortunately, self.model is still being used (and exposed)
        # in the token generation code, so we still need to set it here.
        self.model = self.language_model

    def execute(
        self,
        model_inputs: ModelInputs,
        # TODO(zheng): This should be folded as KVCacheInputs into ModelInputs.
        kv_cache_inputs: Sequence[Tensor] | None = None,
    ) -> ModelOutputs:
        model_inputs = cast(PixtralInputs, model_inputs)
        if model_inputs.has_vision_inputs:
            image_embeds = self.vision_model.execute(
                model_inputs.pixel_values,
                model_inputs.attention_mask,
                copy_inputs_to_device=False,
            )[0]
        else:
            # batch_size * num_concurrent_media * num_patches are set to 0 here to imitate a dummy tensor (used in text-only mode).
            image_embeds = Tensor.zeros(
                shape=[
                    0,
                    0,
                    self.pipeline_config.huggingface_config.text_config.hidden_size,
                ],
                dtype=self.pipeline_config.dtype,
            ).to(self.pipeline_config.device)

        assert (
            kv_cache_inputs is not None
        ), "Pixtral has KV cache inputs, but none were provided"
        model_outputs = self.language_model.execute(
            model_inputs.input_ids,
            image_embeds,
            model_inputs.input_row_offsets,
            *kv_cache_inputs,
            copy_inputs_to_device=False,
        )
        assert not self.pipeline_config.enable_echo
        assert isinstance(model_outputs[0], Tensor)
        return ModelOutputs(next_token_logits=model_outputs[0])

    def prepare_initial_token_inputs(
        self,
        context_batch: list[TextAndVisionContext],  # type: ignore
    ) -> PixtralInputs:
        # Input row offset type: ["input_row_offsets_len"], UInt32
        input_row_offsets = Tensor.from_numpy(
            np.cumsum(
                [0] + [ctx.seq_len for ctx in context_batch],
                dtype=np.uint32,
            )
        ).to(self.pipeline_config.device)

        # Input Ids: ["total_seq_len"], Int64
        # Create a ragged token vector of length: sum(len(t) for t in tokens).
        tokens = np.ascontiguousarray(
            np.concatenate([ctx.next_tokens for ctx in context_batch])
        )
        input_ids = Tensor.from_numpy(tokens).to(self.pipeline_config.device)

        # TODO: change this to work with all contexts in the batch.
        if context_batch[
            0
        ].pixel_values:  # check if the request has pixel_values
            # Get first image in first batch and permute the order to (HWC).
            # Pixtral processor returns CHW images.
            image = np.ascontiguousarray(
                np.transpose(context_batch[0].pixel_values[0], (1, 2, 0))
            )
            pixel_values = Tensor.from_numpy(image).to(
                self.pipeline_config.device
            )
            # TODO(KERN-782): This should be -inf but softmax saturates with NaNs.
            fill_val = -10000.0
            attention_mask = causal_attention_mask_2d_from_imgs(
                [image],
                self.pipeline_config.huggingface_config.vision_config.patch_size,
                1,
                fill_val,
            )
            attention_mask = Tensor.from_numpy(attention_mask).to(
                self.pipeline_config.device
            )
            return PixtralInputs(
                input_ids=input_ids,
                input_row_offsets=input_row_offsets,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
            )

        return PixtralInputs(
            input_ids=input_ids,
            input_row_offsets=input_row_offsets,
        )

    def prepare_next_token_inputs(
        self,
        next_tokens: Tensor,
        prev_model_inputs: ModelInputs,
    ) -> PixtralInputs:
        prev_model_inputs = cast(PixtralInputs, prev_model_inputs)
        # input_ids, old_row_offsets, Optional: [pixel_values, attention_mask]
        old_row_offsets = prev_model_inputs.input_row_offsets

        row_offsets_size = old_row_offsets.shape[0]
        next_row_offsets = self._input_row_offsets_prealloc[:row_offsets_size]
        # In multi-step execution, don't re-pass the pixel_values and attention_mask.
        return PixtralInputs(
            input_ids=next_tokens,
            input_row_offsets=next_row_offsets,
        )

    def _get_kv_params(self) -> KVCacheParams:
        return KVCacheParams(
            dtype=self.pipeline_config.dtype,
            n_kv_heads=self.pipeline_config.huggingface_config.text_config.num_key_value_heads,
            head_dim=self.pipeline_config.huggingface_config.text_config.head_dim,
            cache_strategy=self.pipeline_config.cache_strategy,
            page_size=self.pipeline_config.kv_cache_page_size,
            enable_prefix_caching=self.pipeline_config.enable_prefix_caching,
        )

    def load_kv_manager(
        self,
        session: InferenceSession,
        available_cache_memory: int,
    ) -> KVCacheManager:
        return load_kv_manager(
            params=self._get_kv_params(),
            max_cache_batch_size=self.pipeline_config.max_cache_batch_size,
            max_seq_len=self.pipeline_config.huggingface_config.max_seq_len,
            num_layers=self.pipeline_config.huggingface_config.text_config.num_hidden_layers,
            devices=self.pipeline_config.devices,
            available_cache_memory=available_cache_memory,
            page_size=self.pipeline_config.kv_cache_page_size,
            session=session,
        )

    def estimate_kv_cache_size(self, available_cache_memory: int) -> int:
        return estimate_kv_cache_size(
            params=self._get_kv_params(),
            max_cache_batch_size=self.pipeline_config.max_cache_batch_size,
            max_seq_len=self.pipeline_config.huggingface_config.max_seq_len,
            num_layers=self.pipeline_config.huggingface_config.text_config.num_hidden_layers,
            available_cache_memory=available_cache_memory,
            devices=self.pipeline_config.devices,
        )

    def load_model(self, session: InferenceSession) -> tuple[Model, Model]:
        if self.pipeline_config.enable_echo:
            msg = "Pixtral model does not currently implement enable echo."
            raise ValueError(msg)

        # Pre-allocate a buffer for input_row_offsets in multistep execution.
        # We do this to avoid materializing and copying a buffer with each multistep step
        self._input_row_offsets_prealloc = Tensor.from_numpy(
            np.arange(
                self.pipeline_config.max_cache_batch_size + 1, dtype=np.uint32
            )
        ).to(self.pipeline_config.device)

        self._weights = self.pipeline_config.load_weights()

        if not isinstance(self._weights, SafetensorWeights):
            msg = (
                "only safetensors weights are currently supported in Pixtral"
                " models."
            )
            raise ValueError(msg)

        if serialized_path := self.pipeline_config.serialized_model_path:
            # Hydrate all weights to be referenced by the serialized path.
            weights_registry = {}
            for name, weight in self._weights.items():
                weights_registry[name] = weight.raw_tensor()

            def serialized_load(serialized_path):
                logging.info(
                    "Loading serialized model from %s", serialized_path
                )
                before = time.perf_counter()
                model = session.load(
                    f"{serialized_path}", weights_registry=weights_registry
                )
                after = time.perf_counter()
                logging.info(
                    f"Loading serialized model took {after - before:.6f} seconds"
                )
                return model

            vision_model = serialized_load(f"{serialized_path}.vision")
            text_model = serialized_load(f"{serialized_path}.text")

        else:

            def compile_model(graph, label, export_path=None):
                logging.info(f"Compiling {label} model...")
                before = time.perf_counter()
                model = session.load(
                    graph,
                    weights_registry=self._weights.allocated_weights,
                )
                after = time.perf_counter()
                logging.info(
                    f"Compiling {label} model took {after - before:.6f} seconds"
                )
                if export_path:
                    mef_path = f"{export_path}.{label}"
                    logging.info(
                        f"Exporting serialized {label} model to {mef_path}"
                    )
                    model._export_mef(mef_path)
                return model

            export_path = self.pipeline_config.save_to_serialized_model_path
            logging.info("Building vision model...")
            vision_graph = _build_vision_graph(
                self.pipeline_config,
                self._weights,
            )
            vision_model = compile_model(vision_graph, "vision", export_path)

            logging.info("Building text model...")
            text_graph = _build_text_graph(
                self.pipeline_config,
                self._weights,
                self._get_kv_params(),
                self.kv_manager,
            )
            text_model = compile_model(text_graph, "text", export_path)

        return vision_model, text_model
