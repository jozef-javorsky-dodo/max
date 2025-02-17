# Writing custom CPU or GPU graph operations using Mojo

> [!NOTE]
> This is a preview of an interface for writing custom operations in Mojo,
> and may be subject to change before the next stable release.

Graphs in MAX can be extended to use custom operations written in Mojo. Four
examples of this are shown here:

- Adding 1 to every element of an input tensor.
- Calculating the Mandelbrot set.
- Performing vector addition using a manual GPU function.
- A top-K token sampler, a complex operation that shows a real-world use case
  for a custom operation used today within a large language model processing
  pipeline.

Custom kernels have been written in Mojo to carry out these calculations. For
each example, a simple graph containing a single operation is constructed
in Python. This graph is compiled and dispatched onto a supported GPU if one is
available, or the CPU if not. Input tensors, if there are any, are moved from
the host to the device on which the graph is running. The graph then runs and
the results are copied back to the host for display.

One thing to note is that this same Mojo code runs on CPU as well as GPU. In
the construction of the graph, it runs on a supported accelerator if one is
available or falls back to the CPU if not. No code changes for either path.
The `vector_addition` example shows how this works under the hood for common
MAX abstractions, where compile-time specialization lets MAX choose the optimal
code path for a given hardware architecture.

The `kernels/` directory contains the custom kernel implementations, and the
graph construction occurs in `addition.py`, `mandelbrot.py`, or
`vector_addition.py`. These examples are designed to stand on their own, so
that they can be used as templates for experimentation.

A single Magic command runs each of the examples:

```sh
magic run addition
magic run mandelbrot
magic run vector_addition
magic run top_k
magic run fused_attention
```

The execution has two phases: first a `kernels.mojopkg` is compiled from the
custom Mojo kernel, and then the graph is constructed and run in Python. The
inference session is pointed to the `kernels.mojopkg` in order to load the
custom operations.
