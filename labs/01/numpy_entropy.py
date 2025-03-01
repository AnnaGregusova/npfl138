#!/usr/bin/env python3
import argparse

import numpy as np

from collections import Counter

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--data_path", default="numpy_entropy_data.txt", type=str, help="Data distribution path.")
parser.add_argument("--model_path", default="numpy_entropy_model.txt", type=str, help="Model distribution path.")
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
# If you add more arguments, ReCodEx will keep them with your default values.


def main(args: argparse.Namespace) -> tuple[float, float, float]:
    # TODO: Load data distribution, each line containing a datapoint -- a string.

    string_array = []
    with open(args.data_path, "r") as data:
        for line in data:
            line = line.rstrip("\n")

            string_array.append(line)

            # TODO: Process the line, aggregating data with built-in Python
            # data structures (not NumPy, which is not suitable for incremental
            # addition and string mapping).

    # TODO: Create a NumPy array containing the data distribution. The
    # NumPy array should contain only data, not any mapping. Alternatively,
    # the NumPy array might be created after loading the model distribution.

    data_counts = Counter(string_array)
    total_count = sum(data_counts.values())

    data_distribution = {k: v / total_count for k, v in data_counts.items()}

    # TODO: Load model distribution, each line `string \t probability`.
    distribution_array = {}
    with open(args.model_path, "r") as model:
        for line in model:
            line = line.rstrip("\n")        
            # TODO: Process the line, aggregating using Python data structures.
            key, prob = line.split("\t")  # Split by tab
            distribution_array[key] = float(prob)  # Store as key-value pair

    # TODO: Create a NumPy array containing the model distribution.
    data_probs = np.array([data_distribution[k] for k in data_distribution])
    model_probs = np.array([distribution_array.get(k, 0) for k in data_distribution])

    # TODO: Compute the entropy H(data distribution). You should not use
    # manual for/while cycles, but instead use the fact that most NumPy methods
    # operate on all elements (for example `*` is vector element-wise multiplication).

    logp = np.log(data_probs)
    entropy = np.sum(-data_probs*logp)

    # TODO: Compute cross-entropy H(data distribution, model distribution).
    # When some data distribution elements are missing in the model distribution,
    # the resulting crossentropy should be `np.inf`.

    logq = np.log(model_probs)
    crossentropy = np.sum(-data_probs*logq)

    # TODO: Compute KL-divergence D_KL(data distribution, model_distribution),
    # again using `np.inf` when needed.
    logp = np.log(data_probs)
    logq = np.log(model_probs)
    kl_divergence = np.sum(data_probs*(logp - logq))

    # Return the computed values for ReCodEx to validate.
    return entropy, crossentropy, kl_divergence


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    entropy, crossentropy, kl_divergence = main(main_args)
    print("Entropy: {:.2f} nats".format(entropy))
    print("Crossentropy: {:.2f} nats".format(crossentropy))
    print("KL divergence: {:.2f} nats".format(kl_divergence))
