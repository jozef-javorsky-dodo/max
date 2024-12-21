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


from max.pipelines import PIPELINE_REGISTRY

from .arch import llama_arch
from .config import get_llama_huggingface_file
from .model import Llama3Model

__all__ = [
    "Llama3Model",
    "get_llama_huggingface_file",
]

PIPELINE_REGISTRY.register(llama_arch)
