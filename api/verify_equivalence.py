"""
verify_equivalence.py  —  RUN in the TF environment AFTER extract_weights.py.

Scores the same random (user, item) pairs with the original TF model and the
NumPy forward pass, then asserts they agree. If this passes, the NumPy rewrite
is provably correct and TF can be dropped from serving. If it fails, the diff
localizes the bug (usually a layer-name mismap or a transpose).

Usage:
    python verify_equivalence.py --model ../model/ncf_model.h5 --weights weights
"""


import argparse
import numpy as np
import tensorflow as tf

import ncf_numpy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",default="../model/ncf_model.h5")
    ap.add_argument("--weights", default="weights")
    ap.add_argument("--n", type=int, default=200)


    args =ap.parse_args()

    model = tf.keras.models.load_model(args.model, compile=False)
    ncf_numpy.load_weights(args.weights)

    #valid ID ranges come straight from the embedding table sizes
    n_users = ncf_numpy._W["gmf_user_emb"].shape[0] #16040
    n_items = ncf_numpy._W["gmf_item_emb"].shape[0] #13706


    rng = np.random.default_rng(0)
    users = rng.integers(0, n_users, size=args.n)
    items = rng.integers(0, n_items, size=args.n)

    tf_scores = model.predict([users, items], verbose=0). ravel()
    np_scores = ncf_numpy.score(users, items)

    max_diff = np.max(np.abs(tf_scores - np_scores))
    mean_diff = np.mean(np.abs(tf_scores - np_scores))

    print(f"pairs compared: {args.n}")
    print(f"max abs diff : {max_diff:.3e}")
    print(f"mean abs diff: {mean_diff:.3e}")
    print("Sample (tf vs np:")

    for k in range(5):
        print(f" {tf_scores[k]:.6f} {np_scores[k]:.6f}")

    assert max_diff < 1e-5, f"MISMATCH - max diff {max_diff:.3e} exceeds 1e-5"

    print("\nPASS- Numpy forward pass matches TF withing tolerance.")


if __name__ == "__main__":
    main()