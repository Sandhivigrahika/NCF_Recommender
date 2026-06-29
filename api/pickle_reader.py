import pickle
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent


USER2ID_PATH    = BASE_DIR / "user2id.pkl"
ID2USER_PATH = BASE_DIR /"model"/ "id2user.pkl"
MOVIE2ID_PATH   = BASE_DIR / "movie2id.pkl"
ID2MOVIE_PATH   = BASE_DIR / "id2movie.pkl"

with open(USER2ID_PATH,"rb") as f:
    obj = pickle.load(f)

with open(ID2USER_PATH, "rb") as f:
    ob2 = pickle.load(f)

#print(type(obj))
#print(obj)


print(ob2.keys())

