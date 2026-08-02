"""
extract_weights.py  —  RUN ONCE, LOCALLY, in the TF (recsys) environment.

Loads the trained .h5, pulls the 12 weight arrays out by layer name,
and saves them as .npy files. This is the ONLY time TensorFlow touches
the serving artifacts. After this runs, the deployed service needs only
NumPy + the weights/ folder.

Usage:
    python extract_weights.py --model ../model/ncf_model.h5 --out weights
"""

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf


#Maps a friendly filename ->  (keras layer name, which array from get_Weights()).
#embeddings return [table]; Dense returns [kernel,bias]

LAYER_MAP = {
    #embedding (one array each: index 0)
"gmf_user_emb": ("embedding",   0),
    "gmf_item_emb": ("embedding_1", 0),
    "mlp_user_emb": ("embedding_2", 0),
    "mlp_item_emb": ("embedding_3", 0),
    # MLP dense stack (kernel = 0, bias = 1)
    "dense_kernel":   ("dense",   0), "dense_bias":   ("dense",   1),
    "dense_1_kernel": ("dense_1", 0), "dense_1_bias": ("dense_1", 1),
    "dense_2_kernel": ("dense_2", 0), "dense_2_bias": ("dense_2", 1),
    # output
    "dense_3_kernel": ("dense_3", 0), "dense_3_bias": ("dense_3", 1),
}

# What each array's shape MUST be — a guard against a wrong model or renamed layers.
EXPECTED = {
    "gmf_user_emb": (16040, 32), "gmf_item_emb": (13706, 32),
    "mlp_user_emb": (16040, 32), "mlp_item_emb": (13706, 32),
    "dense_kernel": (64, 64),   "dense_bias": (64,),
    "dense_1_kernel": (64, 32), "dense_1_bias": (32,),
    "dense_2_kernel": (32, 16), "dense_2_bias": (16,),
    "dense_3_kernel": (48, 1),  "dense_3_bias": (1,),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="../model/ncf_model.h5")
    ap.add_argument("--out", default="weights")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.model} ...")
    model = tf.keras.models.load_model(args.model, compile=False)

    for fname, (layer_name, idx) in LAYER_MAP.items():
        arr = model.get_layer(layer_name).get_weights()[idx].astype(np.float32)
        exp = EXPECTED[fname]
        if arr.shape != exp:
            raise ValueError(
                f"{fname}: expected shape {exp}, got {arr.shape}. "
                f"Layer names or architecture differ from the confirmed spec."
            )
        np.save(out / f"{fname}.npy", arr)
        print(f"  saved {fname:16s} {arr.shape}")

    print(f"\nDone. 12 arrays written to {out.resolve()}/")


if __name__ == "__main__":
    main()