from pathlib import Path
import os
from dotenv import load_dotenv


#base directory = project root (one level up from api/)
BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = BASE_DIR/"model"

# Model + pickle paths
MODEL_PATH      = MODEL_DIR / "ncf_model.h5"
USER2ID_PATH    = MODEL_DIR / "user2id.pkl"
ID2USER_PATH = MODEL_DIR / "id2user.pkl"
MOVIE2ID_PATH   = MODEL_DIR / "movie2id.pkl"
ID2MOVIE_PATH   = MODEL_DIR / "id2movie.pkl"
MOVIES_PKL_PATH = MODEL_DIR / "movies_with_ratings.pkl"

#movies metadata
MOVIES_DAT_PATH = MODEL_DIR / "movies.dat"
RATINGS_DAT_PATH  = MODEL_DIR / "ratings.dat"

#registry + live ratings
REGISTRY_PATH = BASE_DIR/ "user_registry.json"
RATINGS_UPDATED_PATH = BASE_DIR / "ratings_updated.csv"
RATINGS_ORIGINAL_PATH = MODEL_DIR / "ratings.dat"

#database
DATABASE_URL = f"sqlite:///{BASE_DIR}/ratings.db"


#TMDB
TMDB_API_KEY = os.getenv("TMDB_API_KEY", "")

# Retraining
MIN_NEW_RATINGS_TO_RETRAIN = 20   # trigger retrain after this many new ratings
RETRAIN_EPOCHS             = 3    # fine-tune epochs (keep small for speed)
RETRAIN_LEARNING_RATE      = 1e-4