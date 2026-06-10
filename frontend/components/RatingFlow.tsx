"use client";

import { useEffect, useState } from "react";
import { GetPopular, submitRating, triggerRetrain, getRecommendations, MovieResult } from "@/lib/api";

const GENRES = ["Action", "Comedy", "Drama", "Romance", "Thriller", "Sci-Fi", "Horror", "Animation"];
const MIN_RATINGS = 5;

interface Props {
  userName: string;
  internalId: number;
  onComplete: (movies: MovieResult[]) => void;
}

interface RatingCard {
  movie: MovieResult;
  rating: number | null;  // 1-5 or null if not rated
  submitted: boolean;
}

export default function RatingFlow({ userName, internalId, onComplete }: Props) {
  const [step, setStep]               = useState<"genre" | "rate" | "retrain" | "done">("genre");
  const [selectedGenre, setGenre]     = useState<string | null>(null);
  const [cards, setCards]             = useState<RatingCard[]>([]);
  const [loading, setLoading]         = useState(false);
  const [retraining, setRetraining]   = useState(false);
  const [ratingCount, setRatingCount] = useState(0);
  const [message, setMessage]         = useState("");

  // Load popular movies for selected genre
  const handleGenreSelect = async (genre: string) => {
    setGenre(genre);
    setLoading(true);
    try {
      const res = await GetPopular(genre, 10);
      setCards(res.movies.map(m => ({ movie: m, rating: null, submitted: false })));
      setStep("rate");
    } catch {
      setMessage("Failed to load movies.");
    } finally {
      setLoading(false);
    }
  };

  // Submit a single rating
  const handleRate = async (idx: number, score: number) => {
    const card = cards[idx];
    if (!card.movie.movie_id || card.submitted) return;

    // Optimistically update UI
    setCards(prev => prev.map((c, i) =>
      i === idx ? { ...c, rating: score } : c
    ));

    try {
      const res = await submitRating(userName, card.movie.movie_id, score);
      setCards(prev => prev.map((c, i) =>
        i === idx ? { ...c, submitted: true } : c
      ));
      setRatingCount(res.ratings_count);

      if (res.ratings_count >= MIN_RATINGS) {
        setStep("retrain");
      }
    } catch {
      // revert on failure
      setCards(prev => prev.map((c, i) =>
        i === idx ? { ...c, rating: null } : c
      ));
      setMessage("Failed to save rating.");
    }
  };

  // Trigger retrain then get recommendations
  const handleRetrain = async () => {
    setRetraining(true);
    setMessage("");
    try {
      await triggerRetrain(userName);
      setMessage("Retraining complete! Fetching your personalised picks...");

      // Wait a moment for retrain to finish in background
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

  // ── GENRE SELECTION STEP ─────────────────────────────────────
  if (step === "genre") {
    return (
      <div style={{ backgroundColor: "#18181b", border: "1px solid #27272a", borderRadius: "12px", padding: "1.5rem" }}>
        <h2 style={{ color: "white", fontWeight: 600, fontSize: "1.1rem", marginBottom: "0.5rem" }}>
          Pick a genre you love 🎬
        </h2>
        <p style={{ color: "#a1a1aa", fontSize: "0.875rem", marginBottom: "1.25rem" }}>
          We'll show you popular movies from that genre to rate. Your ratings train a personalised model.
        </p>
        <div style={{ display: "flex", flexWrap: "wrap", gap: "0.625rem" }}>
          {GENRES.map(g => (
            <button
              key={g}
              onClick={() => handleGenreSelect(g)}
              disabled={loading}
              style={{
                padding: "0.625rem 1.25rem",
                borderRadius: "9999px",
                border: "1px solid #3f3f46",
                backgroundColor: "transparent",
                color: "#a1a1aa",
                fontSize: "0.875rem",
                cursor: loading ? "not-allowed" : "pointer",
                transition: "all 0.15s",
              }}
              onMouseEnter={e => {
                (e.currentTarget as HTMLButtonElement).style.borderColor = "#059669";
                (e.currentTarget as HTMLButtonElement).style.color = "white";
              }}
              onMouseLeave={e => {
                (e.currentTarget as HTMLButtonElement).style.borderColor = "#3f3f46";
                (e.currentTarget as HTMLButtonElement).style.color = "#a1a1aa";
              }}
            >
              {g}
            </button>
          ))}
        </div>
        {loading && <p style={{ color: "#71717a", fontSize: "0.75rem", marginTop: "1rem" }}>Loading movies...</p>}
      </div>
    );
  }

  // ── RATING STEP ──────────────────────────────────────────────
  if (step === "rate" || step === "retrain") {
    return (
      <div style={{ backgroundColor: "#18181b", border: "1px solid #27272a", borderRadius: "12px", padding: "1.5rem" }}>

        {/* Header */}
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "1rem" }}>
          <div>
            <h2 style={{ color: "white", fontWeight: 600, fontSize: "1.1rem" }}>
              Rate these {selectedGenre} movies
            </h2>
            <p style={{ color: "#a1a1aa", fontSize: "0.75rem", marginTop: "0.25rem" }}>
              Rate at least {MIN_RATINGS} movies to get personalised recommendations
            </p>
          </div>

          {/* Progress */}
          <div style={{ textAlign: "right" }}>
            <p style={{ color: "#34d399", fontWeight: 700, fontSize: "1.25rem" }}>
              {ratingCount}/{MIN_RATINGS}
            </p>
            <p style={{ color: "#71717a", fontSize: "0.7rem" }}>ratings</p>
          </div>
        </div>

        {/* Progress bar */}
        <div style={{ height: "4px", backgroundColor: "#27272a", borderRadius: "9999px", overflow: "hidden", marginBottom: "1.5rem" }}>
          <div style={{
            height: "100%",
            backgroundColor: "#10b981",
            borderRadius: "9999px",
            width: `${Math.min((ratingCount / MIN_RATINGS) * 100, 100)}%`,
            transition: "width 0.4s ease",
          }} />
        </div>

        {/* Rating cards */}
        <div style={{ display: "flex", flexDirection: "column", gap: "0.75rem", marginBottom: "1.5rem" }}>
          {cards.map((card, idx) => (
            <div key={card.movie.movie_id ?? idx} style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
              backgroundColor: card.submitted ? "#0f2d1f" : "#27272a",
              border: `1px solid ${card.submitted ? "#10b981" : "#3f3f46"}`,
              borderRadius: "10px",
              padding: "0.75rem 1rem",
              transition: "all 0.2s",
            }}>
              {/* Title */}
              <p style={{
                color: card.submitted ? "#34d399" : "white",
                fontSize: "0.875rem",
                fontWeight: 500,
                flex: 1,
                marginRight: "1rem",
                overflow: "hidden",
                textOverflow: "ellipsis",
                whiteSpace: "nowrap",
              }}>
                {card.submitted ? "✓ " : ""}{card.movie.title}
              </p>

              {/* Star rating */}
              <div style={{ display: "flex", gap: "4px" }}>
                {[1, 2, 3, 4, 5].map(star => (
                  <button
                    key={star}
                    onClick={() => handleRate(idx, star)}
                    disabled={card.submitted}
                    style={{
                      background: "none",
                      border: "none",
                      cursor: card.submitted ? "default" : "pointer",
                      fontSize: "1.25rem",
                      padding: "0",
                      opacity: card.submitted ? 0.5 : 1,
                      color: card.rating !== null && star <= card.rating ? "#fbbf24" : "#3f3f46",
                      transition: "color 0.1s",
                    }}
                    onMouseEnter={e => {
                      if (!card.submitted) {
                        // highlight up to this star
                        const parent = (e.currentTarget as HTMLButtonElement).parentElement;
                        if (parent) {
                          Array.from(parent.children).forEach((child, i) => {
                            (child as HTMLButtonElement).style.color = i < star ? "#fbbf24" : "#3f3f46";
                          });
                        }
                      }
                    }}
                    onMouseLeave={e => {
                      if (!card.submitted) {
                        const parent = (e.currentTarget as HTMLButtonElement).parentElement;
                        if (parent) {
                          Array.from(parent.children).forEach((child, i) => {
                            (child as HTMLButtonElement).style.color =
                              card.rating !== null && i < card.rating ? "#fbbf24" : "#3f3f46";
                          });
                        }
                      }
                    }}
                  >
                    ★
                  </button>
                ))}
              </div>
            </div>
          ))}
        </div>

        {/* Message */}
        {message && (
          <p style={{ color: "#34d399", fontSize: "0.75rem", marginBottom: "1rem", backgroundColor: "rgba(16,185,129,0.1)", padding: "0.5rem 0.75rem", borderRadius: "6px" }}>
            {message}
          </p>
        )}

        {/* Retrain button — appears after MIN_RATINGS */}
        {step === "retrain" && (
          <button
            onClick={handleRetrain}
            disabled={retraining}
            style={{
              width: "100%",
              padding: "0.875rem",
              backgroundColor: retraining ? "#3f3f46" : "#059669",
              color: retraining ? "#71717a" : "white",
              border: "none",
              borderRadius: "10px",
              fontSize: "0.875rem",
              fontWeight: 600,
              cursor: retraining ? "not-allowed" : "pointer",
            }}
          >
            {retraining ? "⚙️ Retraining model on your ratings..." : "⚡ Train model on my ratings → Get recommendations"}
          </button>
        )}
      </div>
    );
  }

  return null;
}