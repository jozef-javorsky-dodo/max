# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
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
    num_steps: int,
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

    try:
        first_token = True
        generate_again = True
        while generate_again:
            responses = pipeline.next_token(
                pipeline_request,
                num_steps=num_steps,
            )

            for response in responses:
                if req_id not in response:
                    # next_token is expected to omit the return if
                    # it encounters eos.
                    generate_again = False
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

            # Yield to the event loop.  If at no other point (e.g.
            # tokenizer.decode which we await earlier does not yield to the
            # event loop), it will be at this point that we'll receive a
            # CancelledError if our future was canceled (e.g., we received a
            # SIGINT).
            await asyncio.sleep(0)

    finally:
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
        assert isinstance(pipeline, TokenGenerator)
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
                        num_steps=pipeline_config.max_num_steps,
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
                num_steps=pipeline_config.max_num_steps,
                metrics=metrics,
                print_tokens=True,
            )
        )
