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
"""Utilities for generating text in the cli."""

import asyncio
import logging
import uuid
from typing import Optional

import requests
from max.pipelines import (
    PIPELINE_REGISTRY,
    PipelineConfig,
)
from max.pipelines.interfaces import (
    PipelineTokenizer,
    TokenGenerator,
    TokenGeneratorRequest,
)

from .metrics import TextGenerationMetrics

logger = logging.getLogger(__name__)

MODEL_NAME = "model"


async def stream_text_to_console(
    pipeline: TokenGenerator,
    tokenizer: PipelineTokenizer,
    prompt: str,
    images: Optional[list[bytes]],
    metrics: Optional[TextGenerationMetrics] = None,
    print_tokens: bool = True,
) -> None:
    req_id = str(uuid.uuid4())
    context = await tokenizer.new_context(
        TokenGeneratorRequest(
            id=req_id,
            index=0,
            prompt=prompt,
            images=images,
            model_name=MODEL_NAME,
        )
    )
    pipeline_request = {req_id: context}
    if print_tokens:
        print(prompt, end="", flush=True)

    prompt_size = context.current_length
    if metrics:
        metrics.prompt_size = prompt_size
        metrics.signpost("begin_generation")

    first_token = True
    while True:
        (response,) = pipeline.next_token(pipeline_request)
        if req_id not in response:
            # next_token is expected to omit the return if
            # it encounters eos.
            break

        encoded_text = response[req_id].next_token
        response_text = await tokenizer.decode(context, encoded_text)
        if metrics:
            if first_token:
                first_token = False
                metrics.signpost("first_token")
            metrics.new_token()
        if print_tokens:
            print(response_text, end="", flush=True)

    if metrics:
        metrics.signpost("end_generation")

    pipeline.release(context)
    if print_tokens:
        print()


def generate_text_for_pipeline(
    pipeline_config: PipelineConfig,
    prompt: str,
    image_urls: list[str] = [],
    num_warmups: int = 0,
) -> None:
    # Run timed run & print results.
    with TextGenerationMetrics(print_report=True) as metrics:
        tokenizer, pipeline = PIPELINE_REGISTRY.retrieve(pipeline_config)

        if image_urls:
            logger.info("Downloading images")
            images = [requests.get(url).content for url in image_urls]
        else:
            images = None

        if num_warmups > 0:
            logger.info("Running warmup")
            for _ in range(num_warmups):
                asyncio.run(
                    stream_text_to_console(
                        pipeline,
                        tokenizer,
                        prompt,
                        images,
                        metrics=None,
                        print_tokens=False,
                    )
                )

        # Run and print results.
        logger.info("Beginning text generation")
        asyncio.run(
            stream_text_to_console(
                pipeline,
                tokenizer,
                prompt,
                images,
                metrics=metrics,
                print_tokens=True,
            )
        )
