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

from max.pipelines import PIPELINE_REGISTRY


def register_all_models():
    """Imports model architectures, thus registering the architecture in the shared PIPELINE_REGISTRY."""
    import max.pipelines.llama3 as llama3

    from .llama_vision import llama_vision_arch
    from .mistral import mistral_arch
    from .mpnet import mpnet_arch
    from .pixtral import pixtral_arch
    from .qwen2 import qwen2_arch
    from .replit import replit_arch

    architectures = [
        llama_vision_arch,
        mistral_arch,
        mpnet_arch,
        pixtral_arch,
        qwen2_arch,
        replit_arch,
    ]

    for arch in architectures:
        PIPELINE_REGISTRY.register(arch)


__all__ = ["register_all_models"]
