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

import compiler
from complex import ComplexSIMD
from math import iota
from utils.index import IndexList
from tensor_utils import ManagedTensorSlice, foreach
from runtime.asyncrt import MojoCallContextPtr


@always_inline
fn mandelbrot_inner_simd[
    float_type: DType, int_type: DType, simd_width: Int
](
    c: ComplexSIMD[float_type, simd_width], max_iterations: SIMD[int_type, 1]
) -> SIMD[int_type, simd_width]:
    """A vectorized implementation of the inner Mandelbrot computation."""
    var z = ComplexSIMD[float_type, simd_width](0, 0)
    var iters = SIMD[int_type, simd_width](0)

    var in_set_mask: SIMD[DType.bool, simd_width] = True
    for _ in range(max_iterations):
        if not any(in_set_mask):
            break
        in_set_mask = z.squared_norm() <= 4
        iters = in_set_mask.select(iters + 1, iters)
        z = z.squared_add(c)

    return iters


alias float_dtype = DType.float32


@compiler.register("mandelbrot", num_dps_outputs=1)
struct Mandelbrot:
    @staticmethod
    fn execute[
        # Parameter that if true, runs kernel synchronously in runtime
        synchronous: Bool,
        # e.g. "CUDA" or "CPU"
        target: StringLiteral,
    ](
        # as num_dps_outputs=1, the first argument is the "output"
        out: ManagedTensorSlice,
        # starting here are the list of inputs
        min_x: Float32,
        min_y: Float32,
        scale_x: Float32,
        scale_y: Float32,
        max_iterations: Int32,
        # the context is needed for some GPU calls
        ctx: MojoCallContextPtr,
    ):
        @parameter
        @always_inline
        fn func[width: Int](idx: IndexList[out.rank]) -> SIMD[out.type, width]:
            var row = idx[0]
            var col = idx[1]
            var cx = min_x.cast[float_dtype]() + (
                col + iota[float_dtype, width]()
            ) * scale_x.cast[float_dtype]()
            var cy = min_y.cast[float_dtype]() + row * SIMD[float_dtype, width](
                scale_y.cast[float_dtype]()
            )
            var c = ComplexSIMD[float_dtype, width](cx, cy)
            return mandelbrot_inner_simd[cx.type, out.type, width](
                c, max_iterations.cast[out.type]()
            )

        foreach[func, synchronous, target](out, ctx)

    # You only need to implement this if you do not manually annotate
    # output shapes in the graph.
    @staticmethod
    fn shape(
        x: ManagedTensorSlice,
    ) raises -> IndexList[x.rank]:
        raise "NotImplemented"
