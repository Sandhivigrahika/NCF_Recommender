from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


# ── Request schemas ──────────────────────────────────────────────

class RecommendRequest(BaseModel):
    raw_id: int = Field(..., examples=[1], description="Existing MovieLens user ID")
    top_n:   int = Field(10, ge=1, le=50, description="Number of recommendations (1–50)")


class RecommendByNameRequest(BaseModel):
    name: str
    top_n: int = Field(10, ge=1, le=50, description="Number of recommendations (1-50)")



class RateRequest(BaseModel):
    raw_id: int = Field(..., examples=[6041])
    movie_id: int   = Field(..., examples=[1])
    score:    float = Field(..., ge=1.0, le=5.0, description="Rating between 1.0 and 5.0")


class PopularRequest(BaseModel):
    genre: Optional[str] = Field(None, examples=["Action"])
    top_n: int           = Field(10, ge=1, le=50)


# ── Response schemas ─────────────────────────────────────────────

class MovieResult(BaseModel):
    title:   str
    score:   Optional[float] = None   # prediction score from the model
    movie_id: Optional[int]  = None
    rating: float | None = None




class MovieDetails(BaseModel):
    '''Returned by GET /movie/{movie_id}/details
    Frontend fetches this per card after the recommendation lisdt renders'''
    title: str
    poster_url: str
    summary: str
    tmdb_id: Optional[int] = None
    rating: float | None = None

class RecommendResponse(BaseModel):
    raw_id: int
    movies:  list[MovieResult]


class PopularResponse(BaseModel):
    genre:  Optional[str]
    movies: list[MovieResult]


class RateResponse(BaseModel):
    message:      str
    ratings_count: int          # how many ratings this user has submitted total
    ready_for_rec: bool         # True once they hit the 5-rating threshold


class MetricsResponse(BaseModel):
    hit_at_k:  float = Field(..., description="Hit@K averaged over test users")
    ndcg_at_k: float = Field(..., description="NDCG@K averaged over test users")
    k:         int
    test_users: int


class RetrainResponse(BaseModel):
    message:      str
    triggered_at: datetime
    new_ratings:  int
    status:       str


class RetrainHistoryItem(BaseModel):
    triggered_at: datetime
    new_ratings:  int
    epochs:       int
    loss_before:  Optional[float]
    loss_after:   Optional[float]
    status:       str

    class Config:
        from_attributes = True


class HealthResponse(BaseModel):
    status:      str
    model_loaded: bool
    db_ok:       bool


class RegisterRequest(BaseModel):
    name: str

class RegisterResponse(BaseModel):
    internal_id: int
    raw_id: int
    is_new: bool
    message: str