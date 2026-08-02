import pickle
import numpy as np
import pandas as pd
#import tensorflow as tf
from pathlib import Path as path
import ncf_numpy

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







def retrain_model(new_ratings: list[dict]) -> dict:
    """
        Retrain the model on binary ratings from a single registered user session.
        new_ratings: list of {"internal_id": int, "movie_id": int, "score": float}
        score is 0 or 1 — no normalisation needed.
        """
    global _model

    user_indices, movie_indices, scores = [],[],[]

    for r in new_ratings:
        raw_movie_id = r["movie_id"]
        internal_uid = r["internal_id"]

        if raw_movie_id not in _movie2id:
            continue


        #lookup the internal id assigned during registration
        # don't create a new one here = register_user already did that

        '''internal_uid = _name_to_internal_id(user_name) ##from the function

        if internal_uid is None:
            logger.warning("User %S not registered. Skipping. ", user_name)
            continue'''

        user_indices.append(internal_uid)
        movie_indices.append(_movie2id[raw_movie_id])
        scores.append(float(r["score"]))

    if not user_indices:
        logger.warning("No valid ratings to retrain on.")
        return {"loss_before": None, "loss_after": None, "epochs": 0 }


    X_user = np.array(user_indices)
    X_movie = np.array(movie_indices)

    Y = np.array(scores, dtype=np.float32)

    if _model is None:
        raise RuntimeError("Model not loaded.")

    _model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=RETRAIN_LEARNING_RATE),
        loss="binary_crossentropy" #correct loss for 0/1 labels
    )

    loss_before= float(_model.evaluate([X_user, X_movie], Y, verbose=0))
    _model.fit([X_user, X_movie], Y, epochs=RETRAIN_EPOCHS, batch_size=32, verbose=0)
    loss_after = float(_model.evaluate([X_user, X_movie],Y, verbose=0))


    _model.save(str(MODEL_SAVE_PATH))
    logger.info("Model Retrained. Loss %5.4f -> %.4f", loss_before, loss_after)

    return {
        "loss_before": round(loss_before, 6),
        "loss_after": round(loss_after, 6),
        "epochs": RETRAIN_EPOCHS,
    }