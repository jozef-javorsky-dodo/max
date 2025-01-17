![Modular Logo](https://modular-assets.s3.amazonaws.com/images/modular_github_logo_bg.png)

# Welcome to MAX

The Modular Accelerated Xecution ([MAX](https://www.modular.com/max)) platform
is an integrated suite of AI libraries, tools, and technologies that unifies
commonly fragmented AI deployment workflows. MAX accelerates time to market
for the latest innovations by giving AI developers a single toolchain that
unlocks full programmability, unparalleled performance, and seamless hardware portability.

![Modular Architecture Diagram](https://modular-assets.s3.amazonaws.com/images/modular_architecture_diagram_bg.png)

[See here to get started with MAX](https://docs.modular.com/max/get-started)
and when you want to report issues or request features,
[please create a GitHub issue here](https://github.com/modularml/max/issues/new/choose).

The [Discord](https://discord.gg/modular) community is the best place to share
your experiences and chat with the team and other community members.

In the [examples directory](https://github.com/modularml/max/tree/main/examples),
you will find code examples and Jupyter notebooks that show how to run inference
with MAX.

## Getting Started

MAX is available in both stable and nightly builds. To install either version,
follow the guide to [create a project with
Magic](https://docs.modular.com/max/create-project).

Then clone this repository:

```bash
git clone https://github.com/modularml/max.git
```

If you installed the nightly build, be sure you switch to the `nightly` branch,
because the `main` branch is for stable releases and might not be compatible
with nightly builds:

```bash
git checkout nightly
```

## Running

### MAX Pipelines

To show off the full power of MAX, a
[series of end-to-end pipelines for common AI workloads](./pipelines/python/)
(and more) are ready to run. As one example, this includes everything needed to
self-host
[the Llama 3.1 text-generation model](./pipelines/python/llama3/).
All code is provided so that these pipelines can be customized, built upon, or
learned from.

### Examples

In addition to the end-to-end pipelines, there are many [examples](./examples/)
that exercise various aspects of MAX. You can follow the instructions in the
README for each example or notebook you want to run.

### Notebooks

Check out the [notebooks examples](./examples/notebooks/) for using MAX Engine
🏎️ for models such as

- [Mistral-7B](./examples/notebooks/mistral7b-python-onnx.ipynb)
- [Roberta-pytorch](./examples/notebooks/roberta-python-pytorch.ipynb)

### Tutorials

The [tutorials](./tutorials/) directory contains the "finished" code for
tutorials you can read at
[docs.modular.com/max/tutorials](https://docs.modular.com/max/tutorials).

### Docker Container

The MAX container is our official Docker container for convenient MAX deployment.
It includes the latest MAX version with GPU support, several AI libraries, and
integrates with orchestration tools like Kubernetes.

The MAX container image is available in the
[Modular Docker Hub repository](https://hub.docker.com/r/modular/max-openai-api/).

## Contributing

Thanks for your interest in contributing to this repository!
We are not accepting pull requests yet.

However, we welcome your bug reports.  If you have a bug, please file an issue
[here](https://github.com/modularml/max/issues/new/choose).

If you need support, the [Discord](https://discord.gg/modular)
community is the best place to share your experiences and chat with
the team and other community members.

## License

This repository and its contributions are licensed under the Apache License
v2.0 with LLVM Exceptions (see the LLVM [License](https://llvm.org/LICENSE.txt)).
MAX and Mojo usage and distribution are licensed under the
[MAX & Mojo Community License](https://www.modular.com/legal/max-mojo-license).

### Third Party Licenses

You are entirely responsible for checking and validating the licenses of
third parties (i.e. Huggingface) for related software and libraries that are downloaded.
