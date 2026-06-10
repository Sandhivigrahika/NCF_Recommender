"use client";

import { useState } from "react";
import {
  getRecommendations, GetPopular, MovieResult,
  registerUser, submitRating, triggerRetrain,
  RegisterResponse
} from "@/lib/api";
import HeroBanner from "@/components/HeroBanner";
import MovieGrid from "@/components/MovieGrid";
import MetricsPanel from "@/components/MetricsPanel";
import RegisterModal from "@/components/RegisterModal";
import RatingFlow from "@/components/RatingFlow";

const GENRES = ["Action", "Comedy", "Drama", "Romance", "Thriller", "Sci-Fi", "Horror", "Animation"];

type Tab = "discover" | "newuser";

// ── New user state ────────────────────────────────────────────────
interface UserSession {
  name: string;
  internal_id: number;
  is_new: boolean;
}

export default function Home() {
  const [tab, setTab] = useState<Tab>("discover");

  // ── Discover tab state ───────────────────────────────────────
  const [userId, setUserId]               = useState("");
  const [movies, setMovies]               = useState<MovieResult[]>([]);
  const [loading, setLoading]             = useState(false);
  const [error, setError]                 = useState("");
  const [mode, setMode]                   = useState<"idle" | "personalised" | "popular">("idle");
  const [selectedGenre, setSelectedGenre] = useState<string | null>(null);

  // ── New user tab state ───────────────────────────────────────
  const [showRegister, setShowRegister]   = useState(false);
  const [session, setSession]             = useState<UserSession | null>(null);
  const [newUserMovies, setNewUserMovies] = useState<MovieResult[]>([]);

  // ── Discover handlers ────────────────────────────────────────
  const handleRecommend = async () => {
    if (!userId.trim()) return;
    setLoading(true);
    setError("");
    setMovies([]);
    try {
      const res = await getRecommendations(parseInt(userId), 10);
      setMovies(res.movies);
      setMode("personalised");
    } catch {
      setError("User not found. Showing popular movies instead.");
      handlePopular(null);
    } finally {
      setLoading(false);
    }
  };

  const handlePopular = async (genre: string | null) => {
    setLoading(true);
    setError("");
    setMovies([]);
    setSelectedGenre(genre);
    try {
      const res = await GetPopular(genre, 10);
      setMovies(res.movies);
      setMode("popular");
    } catch {
      setError("Failed to load movies. Is the API running?");
    } finally {
      setLoading(false);
    }
  };

  // ── New user handlers ────────────────────────────────────────
  const handleRegistered = (user: RegisterResponse & { name: string }) => {
    setSession({ name: user.name, internal_id: user.internal_id, is_new: user.is_new });
    setShowRegister(false);
  };

  const handleNewUserTab = () => {
    setTab("newuser");
    if (!session) setShowRegister(true);
  };

  const handleRatingComplete = (movies: MovieResult[]) => {
    setNewUserMovies(movies);
  };

  // ── STYLES ───────────────────────────────────────────────────
  const s = {
    page:    { minHeight: "100vh", backgroundColor: "#09090b", color: "white" } as React.CSSProperties,
    body:    { maxWidth: "1280px", margin: "0 auto", padding: "2.5rem 1.5rem", display: "flex", flexDirection: "column" as const, gap: "2rem" },
    card:    { backgroundColor: "#18181b", border: "1px solid #27272a", borderRadius: "12px", padding: "1.5rem" },
    input:   { flex: 1, backgroundColor: "#27272a", border: "1px solid #3f3f46", borderRadius: "8px", padding: "0.625rem 1rem", color: "white", fontSize: "0.875rem", outline: "none" } as React.CSSProperties,
    btnGreen:(disabled: boolean) => ({ padding: "0.625rem 1.5rem", backgroundColor: disabled ? "#3f3f46" : "#059669", color: disabled ? "#71717a" : "white", border: "none", borderRadius: "8px", fontSize: "0.875rem", fontWeight: 500, cursor: disabled ? "not-allowed" : "pointer" }) as React.CSSProperties,
    pill:    (active: boolean) => ({ fontSize: "0.75rem", padding: "0.375rem 0.875rem", borderRadius: "9999px", border: `1px solid ${active ? "#059669" : "#3f3f46"}`, backgroundColor: active ? "#059669" : "transparent", color: active ? "white" : "#a1a1aa", cursor: "pointer" }) as React.CSSProperties,
    tab:     (active: boolean) => ({ padding: "0.5rem 1.25rem", borderRadius: "8px", border: `1px solid ${active ? "#059669" : "#27272a"}`, backgroundColor: active ? "#059669" : "transparent", color: active ? "white" : "#71717a", fontSize: "0.875rem", fontWeight: 500, cursor: "pointer" }) as React.CSSProperties,
  };

  return (
    <main style={s.page}>
      {/* Register modal */}
      {showRegister && <RegisterModal onRegistered={handleRegistered} />}

      {/* Hero banner */}
      <HeroBanner />

      <div style={s.body}>

        {/* Tab switcher */}
        <div style={{ display: "flex", gap: "0.75rem", alignItems: "center" }}>
          <button onClick={() => setTab("discover")} style={s.tab(tab === "discover")}>
            🎬 Discover
          </button>
          <button onClick={handleNewUserTab} style={s.tab(tab === "newuser")}>
            ✨ New User
          </button>

          {/* Session indicator */}
          {session && tab === "newuser" && (
            <div style={{ marginLeft: "auto", display: "flex", alignItems: "center", gap: "8px" }}>
              <div style={{ width: "28px", height: "28px", borderRadius: "50%", backgroundColor: "#059669", display: "flex", alignItems: "center", justifyContent: "center", fontSize: "12px", fontWeight: 600, color: "white" }}>
                {session.name[0].toUpperCase()}
              </div>
              <div>
                <p style={{ color: "white", fontSize: "0.8rem", fontWeight: 500, margin: 0 }}>
                  {session.name}
                </p>
                <p style={{ color: "#52525b", fontSize: "0.7rem", margin: 0 }}>
                  ID #{session.internal_id} · {session.is_new ? "new user" : "returning"}
                </p>
              </div>
              <button
                onClick={() => { setSession(null); setShowRegister(true); setNewUserMovies([]); }}
                style={{ marginLeft: "8px", background: "none", border: "0.5px solid #3f3f46", borderRadius: "6px", color: "#71717a", fontSize: "0.7rem", padding: "3px 8px", cursor: "pointer" }}
              >
                Switch
              </button>
            </div>
          )}
        </div>

        {/* ── DISCOVER TAB ─────────────────────────────────────── */}
        {tab === "discover" && (
          <>
            {/* Search card */}
            <div style={s.card}>
              <h2 style={{ color: "white", fontWeight: 600, fontSize: "1.1rem", marginBottom: "0.25rem" }}>
                Get Recommendations
              </h2>
              <p style={{ color: "#a1a1aa", fontSize: "0.875rem", marginBottom: "1.25rem" }}>
                Enter a User ID (1–6040) for personalised picks, or browse by genre.
              </p>

              <div style={{ display: "flex", gap: "0.75rem", marginBottom: "1.5rem" }}>
                <input
                  type="number" min={1} max={6040}
                  placeholder="User ID (e.g. 42)"
                  value={userId}
                  onChange={e => setUserId(e.target.value)}
                  onKeyDown={e => e.key === "Enter" && handleRecommend()}
                  style={s.input}
                />
                <button onClick={handleRecommend} disabled={loading || !userId.trim()} style={s.btnGreen(loading || !userId.trim())}>
                  {loading && mode === "personalised" ? "Loading..." : "Recommend"}
                </button>
              </div>

              <p style={{ color: "#71717a", fontSize: "0.75rem", marginBottom: "0.75rem" }}>Or browse by genre</p>
              <div style={{ display: "flex", flexWrap: "wrap", gap: "0.5rem" }}>
                {GENRES.map(g => (
                  <button key={g} onClick={() => handlePopular(g)} style={s.pill(selectedGenre === g && mode === "popular")}>
                    {g}
                  </button>
                ))}
              </div>
            </div>

            {/* Error */}
            {error && (
              <div style={{ backgroundColor: "rgba(251,191,36,0.1)", border: "1px solid rgba(251,191,36,0.2)", borderRadius: "8px", padding: "0.75rem 1rem", color: "#fbbf24", fontSize: "0.875rem" }}>
                {error}
              </div>
            )}

            {/* Loading skeletons */}
            {loading && (
              <div style={{ display: "grid", gridTemplateColumns: "repeat(5, 1fr)", gap: "1rem" }}>
                {Array.from({ length: 10 }).map((_, i) => (
                  <div key={i} style={{ backgroundColor: "#18181b", borderRadius: "12px", overflow: "hidden", border: "1px solid #27272a" }}>
                    <div style={{ aspectRatio: "2/3", backgroundColor: "#27272a" }} />
                    <div style={{ padding: "1rem" }}>
                      <div style={{ height: "12px", backgroundColor: "#3f3f46", borderRadius: "4px", marginBottom: "8px", width: "80%" }} />
                      <div style={{ height: "12px", backgroundColor: "#3f3f46", borderRadius: "4px", width: "60%" }} />
                    </div>
                  </div>
                ))}
              </div>
            )}

            {/* Movie results */}
            {!loading && movies.length > 0 && (
              <MovieGrid
                movies={movies}
                title={mode === "personalised" ? `Personalised picks for User ${userId}` : `Top ${selectedGenre || "Popular"} movies`}
              />
            )}

            {/* Metrics */}
            <MetricsPanel />
          </>
        )}

        {/* ── NEW USER TAB ─────────────────────────────────────── */}
        {tab === "newuser" && (
          <>
            {/* No session — prompt to register */}
            {!session && (
              <div style={{ ...s.card, textAlign: "center", padding: "3rem" }}>
                <p style={{ fontSize: "3rem", marginBottom: "1rem" }}>✨</p>
                <h2 style={{ color: "white", fontWeight: 600, fontSize: "1.25rem", marginBottom: "0.5rem" }}>
                  New here?
                </h2>
                <p style={{ color: "#a1a1aa", fontSize: "0.875rem", marginBottom: "1.5rem" }}>
                  Register to rate movies and train a personalised recommendation model just for you.
                </p>
                <button onClick={() => setShowRegister(true)} style={{ ...s.btnGreen(false), padding: "0.75rem 2rem" }}>
                  Get Started →
                </button>
              </div>
            )}

            {/* Has session — show rating flow */}
            {session && newUserMovies.length === 0 && (
              <>
                {/* User card */}
                <div style={{ ...s.card, display: "flex", alignItems: "center", gap: "1rem" }}>
                  <div style={{ width: "44px", height: "44px", borderRadius: "50%", backgroundColor: "#059669", display: "flex", alignItems: "center", justifyContent: "center", fontSize: "16px", fontWeight: 700, color: "white", flexShrink: 0 }}>
                    {session.name[0].toUpperCase()}
                  </div>
                  <div>
                    <h2 style={{ color: "white", fontWeight: 600, fontSize: "1.1rem", margin: 0 }}>
                      Hey, {session.name} 👋
                    </h2>
                    <p style={{ color: "#a1a1aa", fontSize: "0.8rem", margin: 0 }}>
                      {session.is_new ? "Welcome! Rate 5 movies to train your personalised model." : "Welcome back! Rate more movies to improve your recommendations."}
                    </p>
                  </div>
                  <div style={{ marginLeft: "auto", display: "flex", gap: "8px", alignItems: "center" }}>
                <button
                  onClick={() => { setSession(null); setNewUserMovies([]); setShowRegister(true); }}
                  style={{ background: "none", border: "0.5px solid #3f3f46", borderRadius: "6px", color: "#71717a", fontSize: "0.7rem", padding: "4px 10px", cursor: "pointer" }}
                >
                  Clear session
                </button>
                <button
                  onClick={() => setTab("discover")}
                  style={{ background: "none", border: "0.5px solid #3f3f46", borderRadius: "6px", color: "#71717a", fontSize: "0.7rem", padding: "4px 10px", cursor: "pointer" }}
                >
                  ✕ Close
                </button>
                </div>
                </div>

                {/* Rating flow */}
                <RatingFlow
                  userName={session.name}
                  internalId={session.internal_id}
                  onComplete={handleRatingComplete}
                />
              </>
            )}

            {/* Personalised recommendations after retrain */}
            {session && newUserMovies.length > 0 && (
              <>
                <div style={{ ...s.card, backgroundColor: "rgba(16,185,129,0.08)", border: "1px solid rgba(16,185,129,0.2)" }}>
                  <p style={{ color: "#34d399", fontWeight: 600, fontSize: "1rem", margin: 0 }}>
                    ✓ Model retrained on your ratings — here are your personalised picks, {session.name}!
                  </p>
                </div>

                <MovieGrid movies={newUserMovies} title="Your personalised recommendations" />

                <button
                  onClick={() => setNewUserMovies([])}
                  style={{ ...s.btnGreen(false), alignSelf: "flex-start" }}
                >
                  Rate more movies →
                </button>
              </>
            )}
          </>
        )}

      </div>
    </main>
  );
}