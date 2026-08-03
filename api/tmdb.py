import os
import logging
import httpx
from dotenv import load_dotenv
from groq import Groq
from fastapi import HTTPException
import re



load_dotenv(".env.local")
logger = logging.getLogger(__name__)

#----------------config-----------------------------------------------
TMDB_API_KEY = os.getenv("TMDB_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
PLACEHOLDER  = "https://placehold.co/500x750/1a1a2e/ffffff?text=No+Poster"

groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None



def _clean_title(raw: str) -> tuple[str, str | None]:
    """MovieLens 'Matrix, The (1999)' -> ('The Matrix', '1999') for TMDB search."""
    year = None
    m = re.search(r"\((\d{4})\)\s*$", raw)
    if m:
        year = m.group(1)
        raw = raw[:m.start()].strip()
    m = re.match(r"^(.*),\s+(The|A|An)$", raw)   # 'Matrix, The' -> 'The Matrix'
    if m:
        raw = f"{m.group(2)} {m.group(1)}"
    return raw.strip(), year

#----- Groq summary -------------------------------------------------
"""
Always generates a fresh summary via Groq.
    Uses TMDB overview as grounding context if available.
    
    Movie title + TMDB overview
            ↓
      Prompt engineering
            ↓
        Llama model
            ↓
Better recommendation-style summary
"""

def _groq_summary(movie_title=str, tmdb_overview: str ="") -> str:

    if not groq_client:
        logger.warning("GROQ_API_KEY not set.")
        return tmdb_overview or "Summary unavailable"

    context = f"TMDB description: {tmdb_overview}\n\n" if tmdb_overview else ""


    prompt = (
        f"{context}"
        f"Write a concise, engaging 2-sentence summary of the movie '{movie_title}' "
        f"for a movie recommendation app. "
        f"Make it sound exciting and help the user decide if they'd enjoy it. "
        f"Do not start with 'This movie' or 'The movie'. "
        f"Return only the summary, nothing else."
    )

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=120,
            temperature=0.7,
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        logger.error("Groq failed for  '%s': %s", movie_title, e)
        return tmdb_overview or "Summary Unavailable."


#---- step 2: Public async function ---------------------------------------------------

async def get_movie_details(movie_title: str) -> dict:
    """

            Movie title
                ↓
            TMDB API call
                ↓
            Get poster + overview
                ↓
            Send overview to LLM
                ↓
            Generate better summary
                ↓
            Return structured movie data

        Fetch poster from TMDB and generate summary via Groq.
        Uses httpx async client — FastAPI native, no SSL hacks needed.

        Returns:
            {
                "poster_url": str,
                "summary":    str,
                "tmdb_id":    int | None,
            }


             http request -> https://api.themoviedb.org/3/search/movie
                            ?api_key=XYZ
                            &query=Interstellar

            tmdb returns json -> {

                                        "results": [
                                        {
                                        "id":123243,
                                        "overview": " str ....",
                                        "poster_path": "/xxyxyx.jpg"
                                          }
                                         ]
                                        }
        """

    #default values
    poster_url = PLACEHOLDER
    tmdb_overview = ""
    tmdb_id = None


    #step 1: fetch poster from TMDB (async) ------------------
    if TMDB_API_KEY:
        try:
            clean, year = _clean_title(movie_title)
            params = {"api_key": TMDB_API_KEY, "query": clean}
            if year:
                params["year"] = year
            async with httpx.AsyncClient(timeout=10.0) as client: #this creates an http client with httpx
                resp = await client.get(
                    "https://api.themoviedb.org/3/search/movie",
                    params=params,
                )

                resp.raise_for_status() #this checks for errors -> did the request succeed

                results = resp.json().get("results", []) #converts API response into python dict, safely gets response["results"]

                if results:
                    movie = results[0] #take the first result (the best match)
                    tmdb_id = movie.get("id")
                    tmdb_overview = movie.get("overview", "")
                    poster = movie.get("poster_path")

                    if poster:
                        poster_url = f"https://image.tmdb.org/t/p/w500{poster}" #build the poster url
        except httpx.HTTPError as e:
            logger.error("TMDB request failed for '%s': %s", movie_title, e)


            raise HTTPException(
                status_code=503,
                detail="External Service Unavailable"
            )

    else:
        logger.warning("TMDB_API_KEY not set — using placeholder poster.")


    #step 2 -> generate summary
    summary = _groq_summary(movie_title, tmdb_overview)


    return {
        "title": movie_title,
        "poster_url": poster_url,
        "summary": summary,
        "tmdb_id" : tmdb_id
    }


""" Why did i use httpx.HTTPError instead of HTTPException

HTTPException is something you raise intentionally in FastAPI. It is raised by your fastapi application

httpx.HTTPError comes from external API request Failure

"""