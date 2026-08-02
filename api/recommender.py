import pickle
import numpy as np
import pandas as pd
#import tensorflow as tf
from pathlib import Path as path
import api.ncf_numpy as ncf_numpy

import json
import time


import logging

from api.config import (
 MODEL_PATH, USER2ID_PATH, MOVIE2ID_PATH,
    ID2MOVIE_PATH, MOVIES_PKL_PATH, MOVIES_DAT_PATH,
    RETRAIN_EPOCHS, RETRAIN_LEARNING_RATE,
    REGISTRY_PATH, ID2USER_PATH, RATINGS_UPDATED_PATH, RATINGS_DAT_PATH, MODEL_SAVE_PATH, USER2ID_SAVE_PATH,
ID2USER_SAVE_PATH, WEIGHTS_DIR
)


logger = logging.getLogger(__name__)

#Singleton state
# loaded once on startup, reused across all requests

#_model:  tf.keras.Model | None = None
_user2id: dict = {}
_id2user: dict = {}
_movie2id: dict = {}
_id2movie: dict = {}
_movies_with_ratings: pd.DataFrame | None = None
_movie_dict: dict = {} #movieId => title
_user_last_movie: dict = {} #userId -> last-rated movieId by timestamp for metric calculation
_ratings_df: pd.DataFrame | None = None #declare at module level




def load_artifacts():
    """ Load model + all pickle files into module-level singletons"""


    global _user2id, _movie2id, _id2movie, _movies_with_ratings, _movie_dict, _id2user, _user_last_movie, _ratings_df

    model_src = MODEL_SAVE_PATH if MODEL_SAVE_PATH.exists()  else MODEL_PATH
    user2id_src = USER2ID_SAVE_PATH if USER2ID_SAVE_PATH.exists() else USER2ID_PATH
    id2user_src = ID2USER_SAVE_PATH if ID2USER_SAVE_PATH.exists() else ID2USER_PATH


    logger.info("Loading model from %%s", model_src)
    #_model = tf.keras.models.load_model(str(model_src), compile=False)
    ncf_numpy.load_weights(WEIGHTS_DIR) #loading the weights from api/weights for the numpy forward pass
    with open(user2id_src, "rb") as f: _user2id = pickle.load(f) #from raw user ids to compact ids
    with open(id2user_src, "rb") as f: _id2user = pickle.load(f)
    with open(MOVIE2ID_PATH, "rb") as f: _movie2id = pickle.load(f) #from raw movies ids to compact ids
    with open(ID2MOVIE_PATH, "rb") as f: _id2movie = pickle.load(f) #from compact ids to movie, in order to find the movies back
    with open(MOVIES_PKL_PATH, "rb") as f: _movies_with_ratings = pickle.load(f) #movies and their reatings


    _ratings_df = pd.read_csv(
        str(RATINGS_DAT_PATH), sep="::", engine="python",
        header=None, names=["userId", "movieId", "rating", "timestamp"], encoding="latin-1"
    )


    movies_df = pd.read_csv(
        str(MOVIES_DAT_PATH), sep="::", engine="python",
        header=None, names=["movieId", "title", "genres"], encoding="latin-1"
    )

    _movie_dict =  dict(zip(movies_df["movieId"], movies_df["title"]))



    _user_last_movie = (
        _ratings_df.sort_values("timestamp")
                  .groupby("userId")["movieId"]
                  .last()
                  .to_dict()
    )

    logger.info("Last-rated movies loaded for %d users.", len(_user_last_movie))



    logger.info("All artifacts loaded. known users: %d, movies: %d", len(_user2id), len(_movie2id))



def is_model_loaded() -> bool:
    return ncf_numpy.is_loaded()



# ── Core recommendation ───────────────────────────────────────────
def recommend_movies(user_id: int, top_n: int = 10) -> list[dict]:
 """

 Returns top_n recommendations for an existing user.
Each item: {"title": str, "score": float, "movie_id": int}
Returns [] if user_id not in training data.
 """

 if user_id not in _user2id:
     return []

 uid = _user2id[user_id]
 all_movie_ids = list(_movie2id.values()) #internal models ids
 all_raw_ids = list(_movie2id.keys()) # original movielens ids

 user_array = np.full(len(all_movie_ids), uid)
 movie_array = np.array(all_movie_ids)


 if not ncf_numpy.is_loaded():
     raise RuntimeError("Model not loaded.")

 #preds = _model.predict([user_array, movie_array], verbose=0).flatten()
 preds = ncf_numpy.score(user_array, movie_array)


 top_indices = preds.argsort()[-top_n:][::-1]

 results = []

 for idx in top_indices:
     raw_id = all_raw_ids[idx]
     results.append({
         "title": _movie_dict.get(raw_id),
         "score": round(float(preds[idx]), 4),
         "movie_id": raw_id,
     })
 return results
# ── Cold start ────────────────────────────────────────────────────


def cold_start_recommendations(genre: str | None = None, top_n: int = 10) -> list[dict]:
    """
    Returns top popular movies, optionally filtered by genre.
    Each item: {"title": str, "score": None, "movie_id": int}
    """
    df = _movies_with_ratings.copy()

    if genre:
        df = df[df["genres"].str.contains(genre, na=False)]
        '''Using a boolean mask, na=False, this newly created df contains only the 
        movies from the requested genre'''

    df = df.sort_values("avg_rating", ascending=False).head(top_n)
    ''' From the newly created genre dataframe, this sorts the values, taking the top-n rated movies'''

    return [
        {"title": row["title"], "score": round(float(row["avg_rating"]), 4), "movie_id": int(row["movieId"])}
        for _, row in df.iterrows()
    ] #interrows iterates row by row, each iteration gives (index, row)

# ── Hit@K / NDCG@K evaluation ────────────────────────────────────

def _dcg_at_k(hits: list[int], k: int) -> float:
    """Discounted Cumulative Gain at K."""
    hits = hits[:k]
    result = 0.0

    for i, h in enumerate(hits):
        result += (h/np.log2(i+2))
    return result



def compute_metrics(k: int = 10, sample_users: int = 200) -> dict:
    """
    Leave-one-out evaluation on a random sample of known users.
    Holds out each user's 'last' movie (by index as proxy) and checks
    whether the model ranks it in the top-K.

    Returns hit@k, ndcg@k, k, test_users.
    """
    all_users = list(_user2id.keys()) #{ user_id: 0, user_id: 1 },
    all_movie_ids = list(_movie2id.values())
    all_raw_ids = list(_movie2id.keys())

    all_users = [u for u in _user2id.keys() if isinstance(u, (int, np.integer))]
    '''user2id.pkl was built in the notebook where pandas/numpy operations produce int64 keys, 
    not plain Python int. So every key failed the isinstance(u, int) check and all_users came back empty — zero users sampled, 
    zero evaluated, np.mean([]) returned nan, FastAPI serialised it as null.'''
    sampled = np.random.choice(all_users, size=min(sample_users, len(all_users)), replace=False)

    hits, ndcgs = [], []

    for user_id in sampled:
        uid = _user2id[user_id]

        # Use the last movie in _id2movie as the held-out item (proxy)
        # In a real setup you'd use the actual last-rated movie from ratings.dat
        #held_out_raw_id = all_raw_ids[-1]  # deterministic stand-in

        #--Real held-out: last movie rated by this user by timestamp
        held_out_raw_id = _user_last_movie.get(user_id)

        if held_out_raw_id is None:
            continue #skip users with no rating history
        if held_out_raw_id not in _movie2id:
            continue

        ##The 100-candidate protocol(standard NeuMF evaluation)
        # sample 99 random movies the user has NOT rated (this is not implemented yet)
        # + the held-out movie = 100 candidates total

        #exclude the already rated movies, used a set for faster lookup
        #

        user_rated = set(_ratings_df[_ratings_df["userId"] == user_id]["movieId"].tolist())

        user_rated.add(held_out_raw_id)

        negative_pool = [m for m in all_raw_ids if m not in user_rated]
        negatives_99 = np.random.choice(negative_pool, size=99, replace=False).tolist()

        candidates = negatives_99 + [held_out_raw_id]

        #get compact ids for these 100 candidates only
        cand_compact = [_movie2id[m] for m in candidates]
        cand_raw = candidates

        #score only the 100 candidates
        user_array = np.full(len(cand_compact), uid)
        movie_array = np.array(cand_compact)

        predictions = ncf_numpy.score(user_array, movie_array)

        ''' _model.predict() will return a 2D array 
        [[0.3 ],
         [0.7 ],
         [0.5 ],
         [0.2 ],
         [0.9 ],
         ...]
         .flatten() -> squeezes it into a 1D Array - shape(10,)
         Flattening is needed so that preds.argsort works cleanly
        '''

        #Build top_k_raw
        sorted_indices  = predictions.argsort()

        '''
        Index based sorting using argsort() -:
        ,argsort() does not directly sort the array - it returns the indices that that will sort the array 
        for eg.
        
        predictions = np.array([0.23, 0.87, 0.45, 0.91, 0.12]).argsort()
        will return -          [4      0      2    1      3]
        
        that's why argsort does not change the indices and can be used to find the movie ids 
        


        '''
        top_k_indices = sorted_indices[-k:]
        top_k_indices = top_k_indices[::-1]

        #map indices back to raw movie IDs

        top_k_raw = []
        for i in top_k_indices:
            top_k_raw.append(cand_raw[i])

        '''.argsort() -> sorts the preds array in descending order
        [-k:] -> grabs the last 3 elements -> the highest rating ones
        [::-1] -> flip the order
        all_raw_ids[i] , maps the indices back to raw ids [101, 102 etc ]'''

        hit_vec = [1 if mid == held_out_raw_id else 0 for mid in top_k_raw]
        '''hit_vec would be a list of hits and misses, if the held out movies
        appears in the top_k_raw, the list will be [0,0,0,1,0,0,0] -> one representing
         the hit.'''
        if held_out_raw_id in top_k_raw:
            hits.append(1)
        else:
            hits.append(0)
        '''hits is independent of hit_vec and used to calculate hit@k,
        while hit_vec is fed directly into the _dcg_at_k
        
        #while the external loop is running, this block will check the hits, and create an array of hits [1,0,1] -> 
        the hits array will contain a binary representation of whether the held_out_movie was found in the the top_k_raw
        the length of the hits will be equal to the length of users taken in the sample
        
        '''
        ndcgs.append(_dcg_at_k(hit_vec, k))

        '''
        for each user:
            - hit_vec is [0, 1, 0] (where the held-out movie was found in top-k)
            - _dcg_at_k takes hit_vec and computes a rank-weighted score
            - that score is appended to ndcgs list

        after all 200 users:
            ndcgs = [0.63, 0.0, 1.0, 0.5, ...]  ← one score per user
            np.mean(ndcgs) gives the final ndcg@k
        '''
    logger.info("Metrics computed: %d users evaluated out of %d sampled", len(hits), len(sampled))

    return {
        "hit_at_k": round(float(np.mean(hits)), 4), #np.mean(hits) (1+0+1+1+0+0+1...) / 200 = 0.42
        "ndcg_at_k": round(float(np.mean(ndcgs)), 4),
        "k": k,
        "test_users": len(hits),
    }


# ── Retraining ────────────────────────────────────────────────────


'''
This is a partial retrain — just adjusting the weights of the already trained model.

What this function actually does

python
_model.fit([X_user, X_movie], Y, epochs=RETRAIN_EPOCHS, batch_size=32, verbose=0)
Takes the existing model with all its learned weights
Runs a few epochs on just the 20 new ratings
Makes small nudges to the weights via RETRAIN_LEARNING_RATE = 1e-4
Saves the updated weights back to disk
The model architecture never changes. The vast majority of weights (all the other users' embeddings, the dense layers) barely move because the learning rate is tiny.
'''



'''The one big issue with retraining something worth looking out for 
When the model was created the the embedding table had a defined shape for eg.

User ID  |  Embedding (32 numbers)
---------|------------------------
0        |  [0.23, -0.11, ...]
1        |  [0.87,  0.45, ...]
2        |  [-0.12, 0.33, ...]
...      |  ...
670      |  [0.56, -0.22, ...]   ← hard stop

when keras builds the model, it allocates a matrix of shape  (671, 32) in memory. 
That matrix is fixed. It doesn't grow like aPythin list would -it's a tensor with a defined shape
that lives in the computation graph.

Why it can't just grow automatically
When you do model.save() and model.load(), Keras serialises the weight matrix at its exact shape. 
If the shape changed, the saved file and the model definition would be inconsistent and it would fail to load.
The entire model architecture — inputs, layers, outputs — 
is compiled into a static graph. Changing a dimension means rebuilding and recompiling the graph from scratch.

SOLUTION -

The buffer solution
When you train the model for the first time, you intentionally lie about how many users there
are: 
num_users = len(_user2id)        # actual users, e.g. 671
BUFFER    = 10000                 # extra empty slots

user_embedding_gmf = Embedding(num_users + BUFFER, embedding_dim)(user_input)
user_embedding_mlp = Embedding(num_users + BUFFER, embedding_dim)(user_input)

So your embedding table now looks like:
User ID  |  Embedding (32 numbers)
---------|------------------------
0        |  [0.23, -0.11, ...]   ← real user, trained
1        |  [0.87,  0.45, ...]   ← real user, trained
...      |  ...
670      |  [0.56, -0.22, ...]   ← last real user, trained
671      |  [0.00,  0.00, ...]   ← empty slot, waiting
672      |  [0.00,  0.00, ...]   ← empty slot, waiting
...      |  ...
10670    |  [0.00,  0.00, ...]   ← last empty slot


The empty slots are randomly initialized but never used during the original training - they're
just reserved seats

When a new user arrives

new_user_id = max(_user2id.values()) +1 #671 falls inside the buffer

The model looks up index 671 — the slot exists, 
it gets the random embedding, retraining nudges it toward that user's actual taste. No crash, no clipping.


What to observe in project:

loss_before will be high — the new user's embedding is random noise
loss_after will drop — 20 ratings were enough to nudge the embedding toward their taste
This is exactly the cold start problem being solved in real time, which is what you wanted to study

Track new users:
Keep a counter of how many new users you've added:

num_new_users = max(_user2id.values()) + 1 - num_original_users


'''





def get_title_for_movie_id(raw_movie_id: int) -> str | None:
    """Look up a movie title from it's raw MovieLens ID. Instant, no network call"""
    return _movie_dict.get(raw_movie_id)

def register_user(name: str) -> tuple[int , int, bool] :
    """
    Register a user by name. Returns (internal_user_id, is_new_user).
    Same name always gets the same internal ID.


    """
    global _user2id, _id2user #needed since both are mutated

    #load existing registry
    registry = {} #start with an empty dictionary
    if path(REGISTRY_PATH).exists(): #check if the registry file exists
        with open(REGISTRY_PATH) as f:
            registry = json.load(f)

    name_key = name.strip().lower()

    #returning the user
    if name_key in registry:
        raw_id = int(registry[name_key]) #integer raw ID from registry
        internal_id = _user2id[raw_id] # internal embedding index
        return raw_id, internal_id , False

    #assing a new integer raw ID beyond the original range
    #registry values are ints, so max works clearly

    existing_raw_ids = list(registry.values())
    max_original_raw = max(u for u in _user2id.keys() if isinstance(u,(int, np.integer)))
    new_raw_id = max(existing_raw_ids, default=int(max_original_raw)) + 1

    #new user - assign next buffer slot
    new_internal_id = max(_user2id.values()) + 1 # _user2id is the dictionary from the pickle file contains

    EMBEDDING_TABLE_SIZE =  16040 #matched the retrained notebook

    if new_internal_id >= EMBEDDING_TABLE_SIZE:
        raise RuntimeError(
            f"User Buffer exhausted (id {new_internal_id} >= {EMBEDDING_TABLE_SIZE} ). "
            "Model needs to be rebuilt with a larger embedding table."
        )

    #update registry - name ->raw integer ID
    registry[name_key] = int(new_raw_id)
    with open(REGISTRY_PATH, "w") as f:
        json.dump(registry,f, indent=2)

    #update in memory and on-disk maps
    _user2id[new_raw_id] = new_internal_id
    _id2user[new_internal_id] = new_raw_id

    with open(USER2ID_SAVE_PATH, "wb") as f:
        pickle.dump(_user2id, f)

    '''This is fine for now but it's a race condition waiting to happen — if two users register simultaneously, 
    one write will overwrite the other. 
    For a portfolio project it's acceptable. 
    Just be aware of it if an interviewer asks.'''

    with open(ID2USER_SAVE_PATH, "wb") as f:
        pickle.dump(_id2user, f)

    logger.info("New user registered: %s → internal_id %d", name_key, new_internal_id)
    return new_raw_id, new_internal_id, True



def append_to_ratings(internal_user_id: int, session_ratings: list[dict]) -> None:

    """Append a session's ratings to ratings_updated.csv
    session_ratings:  [{"item_id": int, "score": int}]
    """

    rows = []
    for r in session_ratings:
        if r["item_id"] not in _movie2id:
            continue

        rows.append({
            "user_id": internal_user_id,
            "item_id": r["item_id"],
            "rating": r["score"],
            "timestamp": int(time.time()),
            "interactions": r["score"],
            "user_id_mapped": internal_user_id,
            "item_id_mapped": _movie2id[r["item_id"]],
            "source": "live"
        })

    if not rows:
        return

    new_df = pd.DataFrame(rows)

    if path(RATINGS_UPDATED_PATH).exists():
        existing = pd.read_csv(RATINGS_UPDATED_PATH)
        updated = pd.concat([existing, new_df], ignore_index=True)

    else:
        updated = new_df

    updated.to_csv(RATINGS_UPDATED_PATH, index=False)
    logger.info("Appended %d ratings for user %d", len(rows), internal_user_id)




def _name_to_internal_id(name: str) -> int | None:
    """Convert a registered name to its internal embedding index as present in user2id.pkl"""

    if not path(REGISTRY_PATH).exists():
        return None

    with open(REGISTRY_PATH) as f:
        registry = json.load(f) #json.load return "6041" (string)
    name_key = name.strip().lower()
    if name_key not in registry:
        return None
    raw_id = int(registry[name_key]) # int raw ID
    return _user2id.get(raw_id) # internal embedding index


def _name_to_raw_id(name: str) -> int | None:
    if not path(REGISTRY_PATH).exists():
        return None

    with open(REGISTRY_PATH) as f:
        registry = json.load(f)
    name_key = name.strip().lower()
    raw_id = registry.get(name_key)

    return int(raw_id) if raw_id is not None else None




