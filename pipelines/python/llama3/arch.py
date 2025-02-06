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

from max.pipelines import (
    PipelineTask,
    RopeType,
    SupportedArchitecture,
    SupportedEncoding,
    TextTokenizer,
    WeightsFormat,
)
from max.pipelines.kv_cache import KVCacheStrategy

from .model import Llama3Model
from .safetensor_converter import (
    ExaoneSafetensorAdapter,
    LlamaSafetensorAdapter,
)

llama_arch = SupportedArchitecture(
    name="LlamaForCausalLM",
    example_repo_ids=[
        "meta-llama/Llama-3.1-8B-Instruct",
        "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "meta-llama/Llama-Guard-3-8B",
        "meta-llama/Llama-3.2-1B-Instruct",
        "meta-llama/Llama-3.2-3B-Instruct",
        "deepseek-ai/deepseek-coder-6.7b-instruct",
        "modularai/llama-3.1",
    ],
    default_encoding=SupportedEncoding.q4_k,
    supported_encodings={
        SupportedEncoding.gptq: [KVCacheStrategy.NAIVE],
        SupportedEncoding.q4_k: [KVCacheStrategy.NAIVE],
        SupportedEncoding.q4_0: [KVCacheStrategy.NAIVE],
        SupportedEncoding.q6_k: [KVCacheStrategy.NAIVE],
        SupportedEncoding.float32: [
            KVCacheStrategy.PAGED,
            KVCacheStrategy.CONTINUOUS,
            KVCacheStrategy.NAIVE,
        ],
        SupportedEncoding.bfloat16: [
            KVCacheStrategy.PAGED,
            KVCacheStrategy.CONTINUOUS,
            KVCacheStrategy.NAIVE,
        ],
    },
    pipeline_model=Llama3Model,
    tokenizer=TextTokenizer,
    rope_type=RopeType.normal,
    default_weights_format=WeightsFormat.gguf,
    weight_converters={WeightsFormat.safetensors: LlamaSafetensorAdapter},
    task=PipelineTask.TEXT_GENERATION,
)

exaone_arch = SupportedArchitecture(
    name="ExaoneForCausalLM",
    default_encoding=SupportedEncoding.float32,
    task=PipelineTask.TEXT_GENERATION,
    supported_encodings={
        SupportedEncoding.q4_k: [KVCacheStrategy.NAIVE],
        SupportedEncoding.q6_k: [KVCacheStrategy.NAIVE],
        SupportedEncoding.float32: [
            KVCacheStrategy.PAGED,
            KVCacheStrategy.CONTINUOUS,
            KVCacheStrategy.NAIVE,
        ],
        SupportedEncoding.bfloat16: [
            KVCacheStrategy.PAGED,
            KVCacheStrategy.CONTINUOUS,
            KVCacheStrategy.NAIVE,
        ],
    },
    example_repo_ids=[
        "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct",
        "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct",
        "LGAI-EXAONE/EXAONE-3.5-32B-Instruct",
    ],
    pipeline_model=Llama3Model,
    tokenizer=TextTokenizer,
    rope_type=RopeType.neox,
    default_weights_format=WeightsFormat.gguf,
    weight_converters={WeightsFormat.safetensors: ExaoneSafetensorAdapter},
)
