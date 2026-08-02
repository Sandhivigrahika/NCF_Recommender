"""
ncf_numpy.py  —  TensorFlow-free NCF inference.

Reproduces the exact forward pass of the trained NeuMF model using only NumPy.
Loads the 12 weight arrays produced by extract_weights.py once at import/init,
then scores (user, item) pairs. No tensorflow import anywhere.

Architecture (confirmed from the training notebook):
    GMF: gmf_user[u] * gmf_item[i]                      -> 32
    MLP: relu stack over concat(mlp_user[u], mlp_item[i]) -> 16
    fuse: concat[GMF(32), MLP(16)] = 48 -> dense_3 -> sigmoid
"""


from pathlib import Path
import numpy as np


# ---Weight singletons (loaded once) -------------------------------
_W: dict[str, np.ndarray] = {}


_NAMES = [
    "gmf_user_emb", "gmf_item_emb", "mlp_user_emb", "mlp_item_emb",
    "dense_kernel", "dense_bias",
    "dense_1_kernel", "dense_1_bias",
    "dense_2_kernel", "dense_2_bias",
    "dense_3_kernel", "dense_3_bias",
]

def load_weights(weights_dir: str | Path) -> None:
    """Load all 12 arrays into the module-level cache. Call once at startup."""

    d = Path(weights_dir)
    for name in _NAMES:
        _W[name] = np.load(d/f"{name}.npy")

def is_loaded() -> bool:
    return len(_W) == len(_NAMES)

def _relu(x: np.ndarray) -> np.array:
    return np.maximum(0.0, x)

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return np.where( x>=0, 1.0/(1.0 + np.exp(-x)),
                     np.exp(x)/(1.0 + np.exp(x)))


def score(user_ids: np.ndarray, item_ids: np.ndarray) -> np.ndarray:
    '''
    Score aligned(user, item) pairs.
    user_ids, item_ids: 1-d int arrays of equal length N (internal/compact IDs).
    Returns: 1-D float array of length N, each in [0,1].
    Mirrors _model.predict([user_ids, item_ids]).flatten()
    '''

    if not is_loaded():
        raise RuntimeError("wieghts not loaded - call load_weights() first")

    u = np.asarray(user_ids).ravel() #.ravel() a function that flattens a multi-dimensional array into a single 1D array
    i = np.asarray(item_ids).ravel()

    # 1. Lookups -> (N,32) each
    gmf_u = _W["gmf_user_emb"][u]
    gmf_i = _W["gmf_item_emb"][i]
    mlp_u = _W["mlp_user_emb"][u]
    mlp_i = _W["mlp_item_emb"][i]


    # 2. GMF branch: element-wise product -> (N,32)

    gmf = gmf_u * gmf_i

    # 3. MLP branch: concat -=> (N, 64), then relu dense stack -> (N, 16)
    x = np.concatenate([mlp_u, mlp_i], axis=1)
    x = _relu(x @ _W["dense_kernel"] + _W["dense_bias"]) # -> (N, 64)
    x = _relu(x @ _W["dense_1_kernel"] + _W["dense_1_bias"]) # -> (N, 32)
    x = _relu(x @ _W["dense_2_kernel"] + _W["dense_2_bias"])  # -> (N, 16)


    # 4. fuse GMF(32) then MLP(16) -> (N, 48)
    neu = np.concatenate([gmf, x], axis=1)

    # 5. Output -> (N,1) -> sigmoid -> (N,)
    logit = neu @ _W["dense_3_kernel"] + _W["dense_3_bias"]
    return _sigmoid(logit).ravel()


def score_user_against_items(user_id: int, item_ids: np.ndarray) -> np.ndarray:
    """Score one user against many itmes. Tiles the user id across the batch."""
    items = np.asarray(item_ids).ravel()
    users = np.full(len(items), user_id, dtype=np.int64)
    return score(users, items)

