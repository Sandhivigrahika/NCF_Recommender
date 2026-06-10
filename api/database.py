from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, Text
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime
from api.config import DATABASE_URL


engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
Base = declarative_base()


class Rating(Base):
    """Stores ratings submitted by new users via the API."""
    __tablename__ = "ratings"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True)  # e.g. "new_user_abc123"
    movie_id = Column(Integer)  # MovieLens movieId
    score = Column(Float)  # 1.0 – 5.0
    created_at = Column(DateTime, default=datetime.utcnow)


class RetrainHistory(Base):
    """Logs every retraining event."""
    __tablename__ = "retraining_history"

    id = Column(Integer, primary_key=True, index=True)
    triggered_at = Column(DateTime, default=datetime.utcnow)
    new_ratings = Column(Integer)  # how many new ratings were used
    epochs = Column(Integer)
    loss_before = Column(Float, nullable=True)
    loss_after = Column(Float, nullable=True)
    status = Column(String, default="success")  # "success" | "failed"
    notes = Column(Text, nullable=True)


def init_db():
    """Create all tables on startup."""
    Base.metadata.create_all(bind=engine)


def get_db():
    """FastAPI dependency — yields a DB session and closes it after the request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()