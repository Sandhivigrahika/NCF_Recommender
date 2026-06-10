"use client";

import { useEffect, useState, useCallback } from "react";
import {
  GetPopular, submitRating, triggerRetrain,
  getRecommendations, getMovieDetails,
  MovieResult, MovieDetails
} from "@/lib/api";

const GENRES      = ["Action", "Comedy", "Drama", "Romance", "Thriller", "Sci-Fi", "Horror", "Animation"];
const MIN_RATINGS = 20;
const SHOW_COUNT  = 10;
const BUFFER_SIZE = 50;

interface Props {
  userName:   string;
  internalId: number;
  onComplete: (movies: MovieResult[]) => void;
}

interface RatingCard {
  movie:     MovieResult;
  details:   MovieDetails | null;
  loading:   boolean;
  rating:    number | null;
  submitted: boolean;
}

export default function RatingFlow({ userName, internalId, onComplete }: Props) {
  const [step, setStep]               = useState<"genre" | "rate" | "retrain">("genre");
  const [selectedGenre, setGenre]     = useState<string | null>(null);
  const [buffer, setBuffer]           = useState<MovieResult[]>([]);     // all loaded, unrated
  const [visible, setVisible]         = useState<RatingCard[]>([]);      // the 10 shown
  const [ratedIds, setRatedIds]       = useState<Set<number>>(new Set()); // already rated
  const [ratingCount, setRatingCount] = useState(0);
  const [loadingGenre, setLoadingGenre] = useState(false);
  const [retraining, setRetraining]   = useState(false);
  const [message, setMessage]         = useState("");

  // ── Fetch details for a movie card ───────────────────────────
  const fetchDetails = useCallback(async (movie: MovieResult): Promise<RatingCard> => {
    if (!movie.movie_id) return { movie, details: null, loading: false, rating: null, submitted: false };
    try {
      const details = await getMovieDetails(movie.movie_id);
      return { movie, details, loading: false, rating: null, submitted: false };
    } catch {
      return { movie, details: null, loading: false, rating: null, submitted: false };
    }
  }, []);

  // ── Pick 10 unrated from buffer and fetch their details ──────
  const loadVisible = useCallback(async (pool: MovieResult[], alreadyRated: Set<number>) => {
    const unrated  = pool.filter(m => m.movie_id && !alreadyRated.has(m.movie_id));
    const shuffled = [...unrated].sort(() => Math.random() - 0.5);
    const picked   = shuffled.slice(0, SHOW_COUNT);

    // Set loading placeholders immediately
    setVisible(picked.map(m => ({ movie: m, details: null, loading: true, rating: null, submitted: false })));

    // Fetch details in parallel
    const withDetails = await Promise.all(picked.map(fetchDetails));
    setVisible(withDetails);
  }, [fetchDetails]);

  // ── Genre selected — load buffer ─────────────────────────────
  const handleGenreSelect = async (genre: string) => {
    setGenre(genre);
    setLoadingGenre(true);
    try {
      const res = await GetPopular(genre, BUFFER_SIZE);
      setBuffer(res.movies);
      setStep("rate");
      await loadVisible(res.movies, new Set());
    } catch {
      setMessage("Failed to load movies. Try again.");
    } finally {
      setLoadingGenre(false);
    }
  };

  // ── Shuffle — pick 10 new unrated from buffer ────────────────
  const handleShuffle = async () => {
    await loadVisible(buffer, ratedIds);
  };

  // ── Rate a movie ─────────────────────────────────────────────
  const handleRate = async (idx: number, score: number) => {
    const card = visible[idx];
    if (!card.movie.movie_id || card.submitted) return;

    // Optimistic update
    setVisible(prev => prev.map((c, i) => i === idx ? { ...c, rating: score } : c));

    try {
      const res = await submitRating(userName, card.movie.movie_id, score);

      // Mark submitted in visible
      setVisible(prev => prev.map((c, i) => i === idx ? { ...c, submitted: true } : c));

      // Track rated IDs
      const newRated = new Set(ratedIds);
      newRated.add(card.movie.movie_id!);
      setRatedIds(newRated);
      setRatingCount(res.ratings_count);

      if (res.ratings_count >= MIN_RATINGS) {
        setStep("retrain");
      }
    } catch {
      // Revert on failure
      setVisible(prev => prev.map((c, i) => i === idx ? { ...c, rating: null } : c));
      setMessage("Failed to save rating. Try again.");
    }
  };

  // ── Retrain + get recommendations ────────────────────────────
  const handleRetrain = async () => {
    setRetraining(true);
    setMessage("Retraining model on your ratings...");
    try {
      await triggerRetrain(userName);
      setMessage("Retrain complete! Fetching your personalised picks...");
      await new Promise(r => setTimeout(r, 3000));
      const res = await getRecommendations(internalId, 10);
      onComplete(res.movies);
    } catch (e: any) {
      setMessage(e.message?.includes("ratings")
        ? `Need ${MIN_RATINGS}+ ratings first.`
        : "Retrain failed — try again.");
    } finally {
      setRetraining(false);
    }
  };

  // ── Unrated remaining in current visible ─────────────────────
  const unratedVisible = visible.filter(c => !c.submitted);

  // ── GENRE SELECTION ──────────────────────────────────────────
  if (step === "genre") {
    return (
      <div style={{ backgroundColor: "#18181b", border: "1px solid #27272a", borderRadius: "12px", padding: "1.5rem" }}>
        <h2 style={{ color: "white", fontWeight: 600, fontSize: "1.1rem", marginBottom: "0.5rem" }}>
          Pick a genre to get started 🎬
        </h2>
        <p style={{ color: "#a1a1aa", fontSize: "0.875rem", marginBottom: "1.25rem" }}>
          We'll show you 10 popular movies at a time to rate. Rate {MIN_RATINGS} to train your personalised model.
        </p>
        <div style={{ display: "flex", flexWrap: "wrap", gap: "0.625rem" }}>
          {GENRES.map(g => (
            <button
              key={g}
              onClick={() => handleGenreSelect(g)}
              disabled={loadingGenre}
              style={{
                padding: "0.625rem 1.25rem", borderRadius: "9999px",
                border: "1px solid #3f3f46", backgroundColor: "transparent",
                color: "#a1a1aa", fontSize: "0.875rem",
                cursor: loadingGenre ? "not-allowed" : "pointer",
              }}
              onMouseEnter={e => { (e.currentTarget as HTMLElement).style.borderColor = "#059669"; (e.currentTarget as HTMLElement).style.color = "white"; }}
              onMouseLeave={e => { (e.currentTarget as HTMLElement).style.borderColor = "#3f3f46"; (e.currentTarget as HTMLElement).style.color = "#a1a1aa"; }}
            >
              {g}
            </button>
          ))}
        </div>
        {loadingGenre && <p style={{ color: "#71717a", fontSize: "0.75rem", marginTop: "1rem" }}>Loading movies...</p>}
      </div>
    );
  }

  // ── RATING STEP ──────────────────────────────────────────────
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>

      {/* Header + progress */}
      <div style={{ backgroundColor: "#18181b", border: "1px solid #27272a", borderRadius: "12px", padding: "1.5rem" }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: "1rem" }}>
          <div>
            <h2 style={{ color: "white", fontWeight: 600, fontSize: "1.1rem", margin: 0 }}>
              Rate {selectedGenre} movies
            </h2>
            <p style={{ color: "#a1a1aa", fontSize: "0.75rem", marginTop: "4px" }}>
              Rate any movie using the stars. Shuffle to see 10 different ones.
            </p>
          </div>
          <div style={{ textAlign: "right" }}>
            <p style={{ color: "#34d399", fontWeight: 700, fontSize: "1.5rem", margin: 0, lineHeight: 1 }}>
              {ratingCount}<span style={{ color: "#52525b", fontSize: "0.875rem", fontWeight: 400 }}>/{MIN_RATINGS}</span>
            </p>
            <p style={{ color: "#71717a", fontSize: "0.7rem", margin: 0 }}>ratings</p>
          </div>
        </div>

        {/* Progress bar */}
        <div style={{ height: "4px", backgroundColor: "#27272a", borderRadius: "9999px", overflow: "hidden", marginBottom: "1rem" }}>
          <div style={{
            height: "100%", backgroundColor: "#10b981", borderRadius: "9999px",
            width: `${Math.min((ratingCount / MIN_RATINGS) * 100, 100)}%`,
            transition: "width 0.4s ease",
          }} />
        </div>

        {/* Shuffle button */}
        <div style={{ display: "flex", gap: "0.75rem", alignItems: "center" }}>
          <button
            onClick={handleShuffle}
            style={{
              padding: "0.5rem 1.25rem", borderRadius: "8px",
              border: "1px solid #3f3f46", backgroundColor: "transparent",
              color: "#a1a1aa", fontSize: "0.8rem", cursor: "pointer",
            }}
            onMouseEnter={e => { (e.currentTarget as HTMLElement).style.borderColor = "#71717a"; (e.currentTarget as HTMLElement).style.color = "white"; }}
            onMouseLeave={e => { (e.currentTarget as HTMLElement).style.borderColor = "#3f3f46"; (e.currentTarget as HTMLElement).style.color = "#a1a1aa"; }}
          >
            🔀 Shuffle movies
          </button>
          <p style={{ color: "#52525b", fontSize: "0.7rem" }}>
            {buffer.filter(m => m.movie_id && !ratedIds.has(m.movie_id!)).length} unrated movies left in pool
          </p>
        </div>
      </div>

      {/* Message */}
      {message && (
        <div style={{ backgroundColor: "rgba(16,185,129,0.1)", border: "1px solid rgba(16,185,129,0.2)", borderRadius: "8px", padding: "0.625rem 0.875rem", color: "#34d399", fontSize: "0.75rem" }}>
          {message}
        </div>
      )}

      {/* Movie cards grid */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(160px, 1fr))", gap: "1rem" }}>
        {visible.map((card, idx) => (
          <div
            key={card.movie.movie_id ?? idx}
            style={{
              backgroundColor: card.submitted ? "#0f2d1f" : "#18181b",
              border: `1px solid ${card.submitted ? "#10b981" : "#27272a"}`,
              borderRadius: "12px", overflow: "hidden",
              opacity: card.submitted ? 0.6 : 1,
              transition: "all 0.2s",
            }}
          >
            {/* Poster */}
            <div style={{ position: "relative", aspectRatio: "2/3", backgroundColor: "#27272a", overflow: "hidden" }}>
              {card.loading ? (
                <div style={{ width: "100%", height: "100%", backgroundColor: "#3f3f46" }} />
              ) : (
                <img
                  src={card.details?.poster_url || "https://placehold.co/500x750/1a1a2e/ffffff?text=No+Poster"}
                  alt={card.movie.title}
                  style={{ width: "100%", height: "100%", objectFit: "cover" }}
                  onError={e => { (e.target as HTMLImageElement).src = "https://placehold.co/500x750/1a1a2e/ffffff?text=No+Poster"; }}
                />
              )}
              {/* Rated overlay */}
              {card.submitted && (
                <div style={{
                  position: "absolute", inset: 0,
                  backgroundColor: "rgba(16,185,129,0.2)",
                  display: "flex", alignItems: "center", justifyContent: "center",
                  fontSize: "2rem",
                }}>
                  ✓
                </div>
              )}
            </div>

            {/* Info */}
            <div style={{ padding: "0.75rem" }}>
              {/* Title */}
              <p style={{
                color: card.submitted ? "#34d399" : "white",
                fontWeight: 600, fontSize: "0.75rem", lineHeight: 1.4,
                marginBottom: "0.375rem",
                display: "-webkit-box", WebkitLineClamp: 2,
                WebkitBoxOrient: "vertical", overflow: "hidden",
              }}>
                {card.movie.title}
              </p>

              {/* Summary */}
              {!card.submitted && (
                <p style={{
                  color: "#71717a", fontSize: "0.65rem", lineHeight: 1.5,
                  marginBottom: "0.625rem",
                  display: "-webkit-box", WebkitLineClamp: 3,
                  WebkitBoxOrient: "vertical", overflow: "hidden",
                }}>
                  {card.loading ? "Loading..." : card.details?.summary || ""}
                </p>
              )}

              {/* Stars */}
              {!card.submitted ? (
                <div style={{ display: "flex", gap: "2px", justifyContent: "center" }}>
                  {[1, 2, 3, 4, 5].map(star => (
                    <button
                      key={star}
                      onClick={() => handleRate(idx, star)}
                      style={{
                        background: "none", border: "none", cursor: "pointer",
                        fontSize: "1.1rem", padding: "2px",
                        color: card.rating !== null && star <= card.rating ? "#fbbf24" : "#3f3f46",
                        transition: "color 0.1s",
                      }}
                      onMouseEnter={e => {
                        const parent = (e.currentTarget as HTMLElement).parentElement;
                        if (parent) Array.from(parent.children).forEach((c, i) => {
                          (c as HTMLElement).style.color = i < star ? "#fbbf24" : "#3f3f46";
                        });
                      }}
                      onMouseLeave={e => {
                        const parent = (e.currentTarget as HTMLElement).parentElement;
                        if (parent) Array.from(parent.children).forEach((c, i) => {
                          (c as HTMLElement).style.color =
                            card.rating !== null && i < card.rating ? "#fbbf24" : "#3f3f46";
                        });
                      }}
                    >
                      ★
                    </button>
                  ))}
                </div>
              ) : (
                <p style={{ color: "#34d399", fontSize: "0.7rem", textAlign: "center" }}>
                  {"★".repeat(card.rating ?? 0)}
                </p>
              )}
            </div>
          </div>
        ))}
      </div>

      {/* Retrain button */}
      {step === "retrain" && (
        <button
          onClick={handleRetrain}
          disabled={retraining}
          style={{
            width: "100%", padding: "1rem",
            backgroundColor: retraining ? "#3f3f46" : "#059669",
            color: retraining ? "#71717a" : "white",
            border: "none", borderRadius: "10px",
            fontSize: "0.875rem", fontWeight: 600,
            cursor: retraining ? "not-allowed" : "pointer",
          }}
        >
          {retraining
            ? "⚙️ Retraining model on your ratings..."
            : `⚡ Train model on my ${ratingCount} ratings → Get personalised recommendations`}
        </button>
      )}
    </div>
  );
}