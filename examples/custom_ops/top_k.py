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

import os
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict

import numpy as np
from max.driver import CPU, Accelerator, Tensor, accelerator_count
from max.dtype import DType
from max.engine.api import InferenceSession
from max.graph import Graph, TensorType, ops
from numpy.typing import NDArray

INPUT_TEXT = """
The quick rabbit runs past the brown fox
The quick rabbit jumps over the brown dog
The quick dog chases past the lazy fox
The quick dog runs through the tall trees
The quick brown fox jumps over the lazy dog
The brown dog sleeps under the shady tree
The brown rabbit hops under the tall tree
The brown fox runs through the forest trees
The brown fox watches the sleeping rabbit
The lazy fox watches over the sleeping dog
The lazy dog watches the quick rabbit
The shady tree shelters the brown rabbit
The shady fox sleeps under the old tree
The sleeping fox rests beside the shady tree
The lazy rabbit rests beside the brown fox
"""


class NextWordFrequency:
    def __init__(self, text):
        # nested `DefaultDict` to create the keys when first indexed
        # Structure looks like: {"word": {"next_word": count}}
        self.word_frequencies: DefaultDict[str, DefaultDict[str, int]] = (
            defaultdict(lambda: defaultdict(int))
        )

        # Track the largest amount of next words to pad the tensor
        self.max_next_words = 0

        # Build word frequencies
        words = text.lower().split()
        for i in range(len(words) - 1):
            current_word = words[i]
            next_word = words[i + 1]
            self.word_frequencies[current_word][next_word] += 1
            self.max_next_words = max(
                self.max_next_words, len(self.word_frequencies[current_word])
            )

    def next_word_probabilities(self, words) -> NDArray[np.float32]:
        if not words:
            return np.empty(0, dtype=np.float32)

        # List to store the probability distributions for each word
        prob_distributions = []

        for word in words:
            if word not in self.word_frequencies:
                raise ValueError(
                    f"Error: cannot predict word after '{word}', not found in input text"
                )

        for word in words:
            frequencies = self.word_frequencies[word]
            freq_list = np.array(list(frequencies.values()), dtype=np.float32)

            # Avoid division by zero
            total = freq_list.sum()
            if total > 0:
                freq_list /= total

            # Pad to largest length of next words
            padded_dist = np.pad(
                freq_list,
                (0, self.max_next_words - len(freq_list)),
                mode="constant",
                constant_values=0,
            )
            prob_distributions.append(padded_dist)

        return np.stack(prob_distributions, axis=0)

    def __getitem__(self, idx):
        return self.word_frequencies[idx]


# Example usage
if __name__ == "__main__":
    # This is necessary only for Modular internal CI.
    if directory := os.getenv("BUILD_WORKSPACE_DIRECTORY"):
        os.chdir(directory)

    path = Path(__file__).parent / "kernels.mojopkg"

    # Initialize the next word frequency for each unique word
    frequencies = NextWordFrequency(INPUT_TEXT)
    word_predictions = ["the", "quick", "brown"]

    # Get probabilities of each next word after `first_word`
    logit_values = frequencies.next_word_probabilities(word_predictions)
    batch_size = len(logit_values)
    K = frequencies.max_next_words

    # Configure our simple one-operation graph.
    with Graph(
        "top_k_sampler",
        input_types=[TensorType(DType.float32, shape=[batch_size, K])],
    ) as graph:
        # Take in the single input to the graph.
        x, *_ = graph.inputs

        # The top_k_custom op is referenced by its string name.
        results = ops.custom(
            name="top_k_custom",
            parameters={"K": K},
            values=[x],
            out_types=[
                TensorType(x.tensor.dtype, x.tensor.shape),
                TensorType(DType.int32, x.tensor.shape),
            ],
        )
        graph.output(*results)

    # Place the graph on a GPU, if available. Fall back to CPU if not.
    device = CPU() if accelerator_count() == 0 else Accelerator()

    # Set up an inference session for running the graph.
    session = InferenceSession(devices=[device], custom_extensions=path)

    # Compile the graph.
    model = session.load(graph)

    # Create a driver tensor from the next word probabilities
    logits = Tensor.from_numpy(logit_values).to(device)

    print(f"Sampling top k: {K} for batch size: {batch_size}")
    # Run the model on the target device.
    values, indices = model.execute(logits)

    # Copy values and indices back to the CPU to be read.
    assert isinstance(values, Tensor)
    values = values.to(CPU())
    np_values = values.to_numpy()

    assert isinstance(indices, Tensor)
    indices = indices.to(CPU())
    np_indices = indices.to_numpy()

    for i in range(batch_size):
        print(f"\nPredicted word after `{word_predictions[i]}`")
        print("-------------------------------")
        print("| word         | confidence   |")
        print("-------------------------------")
        keys = list(frequencies.word_frequencies[word_predictions[i]].keys())

        for j in range(len(np_indices[i])):
            if j > len(keys) - 1:
                break
            print(f"| {keys[np_indices[i][j]]:<13}| {np_values[i][j]:<13.8}|")
        print("-------------------------------")
