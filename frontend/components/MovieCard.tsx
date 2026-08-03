"use client";

import { useEffect, useState } from "react";
import { getMovieDetails, MovieDetails, MovieResult } from "@/lib/api";

interface Props {
  movie: MovieResult;
}

export default function MovieCard({ movie }: Props) {
  const [details, setDetails] = useState<MovieDetails | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!movie.movie_id) return;
    getMovieDetails(movie.movie_id)
      .then(setDetails)
      .catch(() => setDetails(null))
      .finally(() => setLoading(false));
  }, [movie.movie_id]);

  // Only treat score as a "match %" when it's a genuine 0–1 model output
  // (personalised /recommend). Popular/genre results carry a 1–5 avg_rating,
  // which must NOT be shown as a percentage.
  const isMatch = movie.score != null && movie.score >= 0 && movie.score <= 1;
  const scorePercent = isMatch ? Math.round(movie.score! * 100) : null;

  // TMDB community rating, 0–10. Arrives with the details fetch.
  const rating = details?.rating ?? null;
  const showRating = rating != null && rating > 0;

  return (
    <div style={{
      backgroundColor: "#18181b",
      borderRadius: "12px",
      overflow: "hidden",
      border: "1px solid #27272a",
      transition: "border-color 0.2s, transform 0.2s",
      cursor: "pointer",
    }}
      onMouseEnter={e => {
        (e.currentTarget as HTMLDivElement).style.borderColor = "#52525b";
        (e.currentTarget as HTMLDivElement).style.transform = "scale(1.02)";
      }}
      onMouseLeave={e => {
        (e.currentTarget as HTMLDivElement).style.borderColor = "#27272a";
        (e.currentTarget as HTMLDivElement).style.transform = "scale(1)";
      }}
    >
      {/* Poster */}
      <div style={{ position: "relative", aspectRatio: "2/3", backgroundColor: "#27272a", overflow: "hidden" }}>
        {loading ? (
          <div style={{ width: "100%", height: "100%", backgroundColor: "#3f3f46" }} />
        ) : (
          <img
            src={details?.poster_url || "https://placehold.co/500x750/1a1a2e/ffffff?text=No+Poster"}
            alt={movie.title}
            style={{ width: "100%", height: "100%", objectFit: "cover" }}
            onError={(e) => {
              (e.target as HTMLImageElement).src = "https://placehold.co/500x750/1a1a2e/ffffff?text=No+Poster";
            }}
          />
        )}

        {/* Rating badge (TMDB) */}
        {showRating && (
          <div style={{
            position: "absolute", top: "8px", right: "8px",
            backgroundColor: "rgba(0,0,0,0.85)",
            color: "#fbbf24",
            fontSize: "11px",
            fontWeight: 700,
            padding: "2px 8px",
            borderRadius: "9999px",
            border: "1px solid rgba(251,191,36,0.3)",
            display: "flex",
            alignItems: "center",
            gap: "3px",
          }}>
            ★ {rating!.toFixed(1)}
          </div>
        )}
      </div>

      {/* Info */}
      <div style={{ padding: "0.875rem" }}>

        {/* Title */}
        <h3 style={{
          color: "white",
          fontWeight: 600,
          fontSize: "0.8rem",
          lineHeight: 1.4,
          marginBottom: "0.5rem",
          display: "-webkit-box",
          WebkitLineClamp: 2,
          WebkitBoxOrient: "vertical",
          overflow: "hidden",
        }}>
          {movie.title}
        </h3>

        {/* Match bar — only for personalised results (genuine 0–1 score) */}
        {scorePercent !== null && (
          <div style={{ marginBottom: "0.625rem" }}>
            <div style={{ display: "flex", justifyContent: "space-between", fontSize: "10px", color: "#71717a", marginBottom: "3px" }}>
              <span>Match</span>
              <span style={{ color: "#34d399" }}>{scorePercent}%</span>
            </div>
            <div style={{ height: "3px", backgroundColor: "#3f3f46", borderRadius: "9999px", overflow: "hidden" }}>
              <div style={{ width: `${scorePercent}%`, height: "100%", backgroundColor: "#10b981", borderRadius: "9999px" }} />
            </div>
          </div>
        )}

        {/* Summary */}
        <p style={{
          color: "#a1a1aa",
          fontSize: "0.7rem",
          lineHeight: 1.5,
          display: "-webkit-box",
          WebkitLineClamp: 3,
          WebkitBoxOrient: "vertical",
          overflow: "hidden",
        }}>
          {loading ? "Loading summary..." : details?.summary || "Summary unavailable."}
        </p>
      </div>
    </div>
  );
}