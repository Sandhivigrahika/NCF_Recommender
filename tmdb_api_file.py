


######################SSL ERROR FIX #########################
import requests
import ssl
from requests.adapters import HTTPAdapter
from urllib3.util import Retry
import streamlit as st

import os
from dotenv import load_dotenv

load_dotenv()  # loads all variables from .env

def get_secrets(name, default=None): #to handle api and tokens
    """Get secret from streamlit cloud or locally"""
    try:
        return st.secrets[name] #for st cloud
    except Exception:
        return os.getenv(name,default) #works locally


# Use it for your keys
hf_token = get_secrets("HF_TOKEN")
tmdb_key = get_secrets("TMDB_API_KEY")

class TLS12Adapter(HTTPAdapter):
    def init_poolmanager(self, *args, **kwargs):
        ctx = ssl.create_default_context()
        ctx.minimum_version = ssl.TLSVersion.TLSv1_2
        kwargs['ssl_context'] = ctx
        return super().init_poolmanager(*args, **kwargs)



url = "https://api.themoviedb.org/3/search/movie"
params = {"api_key": tmdb_key, "query": "Inception"}


#######################################################

#Create a session with TLS + retries + timeout
session = requests.Session()
retries = Retry(
    total=5,  # retry up to 5 times
    backoff_factor=0.5,  # wait 0.5s, 1s, 2s...
    status_forcelist=[500, 502, 503, 504]
)
adapter = TLS12Adapter(max_retries=retries)
session.mount("https://", adapter)
session.mount("http://", adapter)

#####Generate AI Summary, in case TMDB summary not available

API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
headers = {"Authorization": f"Bearer {hf_token}"}  # replace with your token


def hf_summary(text: str, max_len=80, min_len=30) -> str:
    payload = {
        "inputs": text,
        "parameters": {"max_length": max_len, "min_length": min_len, "do_sample": False},
    }
    response = requests.post(API_URL, headers=headers, json=payload)

    if response.status_code != 200:
        return f"(AI summary unavailable, error {response.status_code})"

    try:
        return response.json()[0]["summary_text"]
    except Exception:
        return "(AI summary unavailable)"



######Placeholder image
PLACEHOLDER_IMAGE = "https://via.placeholder.com/500x750.png?text=No+Image"



def get_movie_details(movie_title):
    url = "https://api.themoviedb.org/3/search/movie"
    params = {"api_key": tmdb_key , "query": movie_title}

    try:
        raw_response = session.get(url, params=params, timeout=10)  #  timeout added
        raw_response.raise_for_status()
    except requests.exceptions.RequestException as e:
        return None, f"API request failed: {e}"

    data = raw_response.json()
    if data.get("results"):
        first = data["results"][0]
        poster_path = f"https://image.tmdb.org/t/p/w500{first['poster_path']}" if first.get("poster_path") else PLACEHOLDER_IMAGE

        summary = first.get("overview")
        if not summary:
            return hf_summary(movie_title,max_len=80, min_len=30)
        return poster_path, summary
    return PLACEHOLDER_IMAGE, hf_summary(movie_title,max_len=80,min_len=30)


'''def get_movie_details(title):
    url = "https://api.themoviedb.org/3/search/movie"
    params = {"api_key": API_KEY, "query": title}

    raw_response = session.get(url, params=params)  # ✅ Use session instead of requests.get

    if raw_response.status_code != 200:
        return None, f"Error: {raw_response.status_code}"

    response = raw_response.json()

    if response.get("results"):
        movie = response["results"][0]
        poster_path = movie.get("poster_path")
        poster_url = f"https://image.tmdb.org/t/p/w500{poster_path}" if poster_path else None
        overview = movie.get("overview", "No summary available")
        return poster_url, overview

    return None, "Movie Not Found"
'''