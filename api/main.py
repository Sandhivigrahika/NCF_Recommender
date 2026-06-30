import logging
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from api import recommender
from api.config import MIN_NEW_RATINGS_TO_RETRAIN
from api.database import init_db, get_db, Rating, RetrainHistory
from api.tmdb import get_movie_details
from api.models import (
    RecommendRequest, RecommendResponse, MovieResult, MovieDetails,
    RateRequest, RateResponse,
    PopularRequest, PopularResponse,
    MetricsResponse,
    RetrainResponse, RetrainHistoryItem,
    HealthResponse, RegisterResponse, RegisterRequest, RecommendByNameRequest
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ── Startup / shutdown ────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Runs once on startup: create DB tables, load model."""
    init_db()
    recommender.load_artifacts()
    yield
    # (teardown goes here if needed)


# ── App ───────────────────────────────────────────────────────────

app = FastAPI(
    title="NCF Movie Recommender API",
    description=(
        "Neural Collaborative Filtering recommender with cold-start support, "
        "online retraining pipeline, and Hit@K / NDCG@K evaluation."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten this after deployment
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Health ────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health(db: Session = Depends(get_db)):
    """Quick liveness check — use this as Render's health-check URL."""
    db_ok = True
    try:
        db.execute(__import__("sqlalchemy").text("SELECT 1"))
    except Exception:
        db_ok = False

    return HealthResponse(
        status="ok" if recommender.is_model_loaded() and db_ok else "degraded",
        model_loaded=recommender.is_model_loaded(),
        db_ok=db_ok,
    )


# ── Recommendations ───────────────────────────────────────────────

@app.post("/recommend", response_model=RecommendResponse, tags=["Recommendations"])
def recommend(req: RecommendRequest):
    """
    Get top-N personalised recommendations for an existing user.
    Returns 404 if the user_id was not seen during training.
    """
    movies = recommender.recommend_movies(req.raw_id, req.top_n)
    if not movies:
        raise HTTPException(
            status_code=404,
            detail=f"User {req.raw_id} not found in training data. Use /popular for cold-start."
        )
    return RecommendResponse(
        raw_id=req.raw_id,
        movies=[MovieResult(**m) for m in movies],
    )

@app.post("/recommend/by-name", response_model=RecommendResponse, tags=["Recommendations"])
def recommend_by_name(req: RecommendByNameRequest): #name str
    raw_id = recommender._name_to_raw_id(req.name)

    if raw_id is None:
        raise HTTPException(404, f" '{req.name}' is not registered. call /regiser first.")

    movies = recommender.recommend_movies(raw_id, req.top_n)
    if not movies:
        raise HTTPException(404, f"raw_id {raw_id} not found in training data. Use /popular for cold-start.")

    return RecommendResponse(raw_id=req.name, movies=[MovieResult(**m) for m in movies])




@app.post("/popular", response_model=PopularResponse, tags=["Recommendations"])
def popular(req: PopularRequest):
    """
    Cold-start endpoint — returns top popular movies, optionally by genre.
    Use this for new users before they have enough ratings.
    """
    movies = recommender.cold_start_recommendations(req.genre, req.top_n)
    return PopularResponse(
        genre=req.genre,
        movies=[MovieResult(**m) for m in movies],
    )

# --------- Get details -------------------------------------------------

@app.get("/movie/{movie_id}/details", response_model=MovieDetails, tags=["Movies"])
async def movie_details(movie_id: int):
    """
        Fetch poster and AI-generated summary for a single movie.

        Frontend calls this per card AFTER /recommend returns the list —
        keeping /recommend fast with no TMDB or Groq calls blocking it.

        Flow:
            1. Title lookup from local movie_dict  (instant, no network)
            2. TMDB search for poster              (async httpx)
            3. Groq generates 2-sentence summary   (uses TMDB overview as context)
        """

    #Step-1 - Local lookup, no network call
    title = recommender.get_title_for_movie_id(movie_id)

    if not title:
        raise HTTPException(
            status_code=404,
            detail= f"movie_id {movie_id} not found in dataset"
        )

    #step 2+3 --TMDB + GROQ
    details = await get_movie_details(title)

    return MovieDetails(
        movie_id=movie_id,
        title=title,
        poster_url=details["poster_url"],
        summary=details["summary"],
        tmdb_id=details["tmdb_id"],
    )

# ── Ratings ───────────────────────────────────────────────────────

@app.post("/rate", response_model=RateResponse, tags=["Ratings"])
def rate(req: RateRequest, db: Session = Depends(get_db)):
    """
    Submit a rating from a new user.
    Once a user submits 20+ ratings, ready_for_rec = True.
    The retraining pipeline picks these up automatically.


    RateRequest ->
    class RateRequest(BaseModel):
    user_id: str
    raw_id: int = Field(..., examples=[6041])
    movie_id: int   = Field(..., examples=[1])
    score:    float = Field(..., ge=1.0, le=5.0, description="Rating between 1.0 and 5.0")



    """

    internal_id = recommender._user2id.get(req.raw_id)
    if internal_id is None:
        raise HTTPException(404, f"raw_id {req.raw_id} not registerd. Call  /register first.")



    rating = Rating(user_id=str(req.raw_id),
                    raw_id=req.raw_id,
                    internal_id=internal_id,
                    movie_id=req.movie_id,
                    score=req.score)
    db.add(rating)
    db.commit()
    count= db.query(Rating).filter(Rating.raw_id == req.raw_id).count()
    return RateResponse(message="Rating Saved.", ratings_count=count, ready_for_rec=count>=MIN_NEW_RATINGS_TO_RETRAIN)

@app.get("/ratings/{raw_id}", tags=["Ratings"])
def get_user_ratings(raw_id: int, db: Session = Depends(get_db)):
    """Fetch all ratings a specific user has submitted."""
    ratings = db.query(Rating).filter(Rating.raw_id == raw_id).all()
    return [
        {"movie_id": r.movie_id, "score": r.score, "created_at": r.created_at}
        for r in ratings
    ]


# ── Retraining ────────────────────────────────────────────────────

def _run_retrain(raw_id: int):
    """Background task: fine-tune the model on all new ratings.
    Starting a new db connection within the background task:
    fast api runs db.close() as soon as the endpoint function returns, because that's when the dependency generator gets closed out,
    it doesn't wait for the background tasks. Beause of this a query or commit inside a background could raise Session is closed error.
    The fix-
    open a fresh connection inside the task itself, scoped to that task's own lifetime and close it within the task itself.
    """

    from api.database import SessionLocal
    db = SessionLocal()
    new_ratings = []
    history = None

    try:
        new_ratings=db.query(Rating).filter(Rating.raw_id == raw_id).all()

        if len(new_ratings) < MIN_NEW_RATINGS_TO_RETRAIN:
            logger.info("Not enough new ratings to retrain (%d < %d).", len(new_ratings), MIN_NEW_RATINGS_TO_RETRAIN)
            return

        internal_id = new_ratings[0].internal_id #already resolved at /rate tim

        if internal_id is None:
            logger.error("Ratings for raw_id %s have no internal_id. Aborting retrain")
            return

        recommender.append_to_ratings(
            raw_id,
            [{"item_id": r.movie_id,
              "score":r.score}
             for r in new_ratings]
        )

        payload = [ {"internal_id": internal_id, "movie_id": r.movie_id, "score": r.score} for r in new_ratings]

        result = recommender.retrain_model(payload)

        history = RetrainHistory(
            triggered_at = datetime.utcnow(),
            new_ratings = len(payload),
            epochs=result["epochs"],
            loss_before=result["loss_before"],
            loss_after=result["loss_after"],
            status="Success",
        )

    except Exception as exc:
        logger.error("Retraining Failed: %s", exc)
        history= RetrainHistory(
        triggered_at=datetime.utcnow(),
        new_ratings=len(new_ratings),
        epochs=0,
        status="failed",
        notes=str(exc),
        )
    finally:
        if history:
            db.add(history)
            db.commit()
        db.close()



@app.post("/retrain", response_model=RetrainResponse, tags=["MLOps"])
def trigger_retrain(background_tasks: BackgroundTasks,
                    raw_id: int,
                    db: Session = Depends(get_db)):
    """
    Manually trigger a retraining run.
    The fine-tuning runs in the background — the response returns immediately.
    Requires at least MIN_NEW_RATINGS_TO_RETRAIN ratings in the DB.
    """
    count = db.query(Rating).filter(Rating.raw_id == raw_id).count()
    if count < MIN_NEW_RATINGS_TO_RETRAIN:
        raise HTTPException(
            status_code=400,
            detail=f"Need at least {MIN_NEW_RATINGS_TO_RETRAIN} ratings to retrain. Currently have {count}."
        )

    background_tasks.add_task(_run_retrain, raw_id)

    return RetrainResponse(
        message="Retraining started in background.",
        triggered_at=datetime.utcnow(),
        new_ratings=count,
        status="started",
    )


@app.get("/retrain/history", response_model=list[RetrainHistoryItem], tags=["MLOps"])
def retrain_history(db: Session = Depends(get_db)):
    """Returns the full log of retraining events — shown in the frontend activity feed."""
    return db.query(RetrainHistory).order_by(RetrainHistory.triggered_at.desc()).all()


# ── Metrics ───────────────────────────────────────────────────────

@app.get("/metrics", response_model=MetricsResponse, tags=["MLOps"])
def metrics(k: int = 10, sample_users: int = 200):
    """
    Compute Hit@K and NDCG@K via leave-one-out evaluation.
    Runs on a random sample of training users (slow for large k/sample — cache in prod).
    """
    result = recommender.compute_metrics(k=k, sample_users=sample_users)
    return MetricsResponse(**result)


@app.post("/register", response_model=RegisterResponse, tags=["Users"])
def register(req: RegisterRequest):
    """
        Register a new user by name or return their existing internal ID.
        Call this when a user lands on the frontend before showing them movies to rate.
        """

    raw_id, internal_id, is_new = recommender.register_user(req.name) #internal_id  = 6043

    return RegisterResponse(
        internal_id=internal_id,
        raw_id=raw_id,
        is_new=is_new,
        message="Welcome!" if is_new else "Welcome back!"
    )
