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
"""Pipeline cli utilities."""

from .config import config_to_flag, pipeline_config_options
from .device_options import DevicesOptionType
from .encode import pipeline_encode
from .generate import generate_text_for_pipeline, stream_text_to_console
from .list import list_pipelines_to_console
from .metrics import TextGenerationMetrics
from .serve import batch_config_from_pipeline_config, serve_pipeline

__all__ = [
    "DevicesOptionType",
    "TextGenerationMetrics",
    "config_to_flag",
    "pipeline_config_options",
    "batch_config_from_pipeline_config",
    "serve_pipeline",
    "generate_text_for_pipeline",
    "stream_text_to_console",
    "list_pipelines_to_console",
    "pipeline_encode",
]
