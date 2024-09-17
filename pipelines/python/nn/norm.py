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

"""Normalization layer."""

from dataclasses import dataclass

from max.graph import TensorType, TensorValue, ValueLike, ops


@dataclass
class RMSNorm:
    weight: ValueLike
    eps: float = 1e-6

    def __call__(self, x: ValueLike) -> TensorValue:
        return ops.custom(
            "rms_norm",
            [x, ops.cast(self.weight, x.dtype), ops.cast(self.eps, x.dtype)],
            [TensorType(dtype=x.dtype, shape=x.shape)],
        )[0]
