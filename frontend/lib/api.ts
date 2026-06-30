/**
 * lib/api.ts — API Abstraction Layer
 *
 * Central module for all backend communication in this app.
 * Keeps fetch logic out of components and in one maintainable place.
 *
 * Contains:
 *  - Base URL config (reads from NEXT_PUBLIC_API_URL env variable)
 *  - TypeScript interfaces: Movie, Metrics (shared across components)
 *  - Fetch functions: getMovies(), getMovieById(), getMetrics()
 *  - Error handling for all API calls
 *
 * Used by:
 *  - MovieGrid.tsx  → getMovies()
 *  - MetricsPanel.tsx → getMetrics()
 *  - app/page.tsx   → orchestrates data fetching
 */


const BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";


//Types-------------------------------------------------------------------------

export interface MovieResult {
    title: string;
    score: number | null;
    movie_id: number | null;

}

export interface MovieDetails {
    movie_id: number;
    title: string;
    poster_url: string;
    summary: string;
    tmdb_id: number | null;
}


export interface RecommendResponse {
    raw_id: number;
    movies: MovieResult[];

}

export interface PopularResponse {
    genre: string | null;
    movies: MovieResult[];
}

export interface MetricsResponse {
  hit_at_k: number;
  ndcg_at_k: number;
  k: number;
  test_users: number;
}


export interface RateResponse {
  message: string;
  ratings_count: number;
  ready_for_rec: boolean;
}

export interface RegisterResponse {
  internal_id:number;
  raw_id: number;
  is_new: boolean;
  message: string;
}

export interface RetrainResponse {
  message: string;
  status: string;
}


// ── API calls ─────────────────────────────────────────────────────


//register a new user

export async function registerUser(name: string): Promise<RegisterResponse> {

    const res = await fetch(`${BASE_URL}/register`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name }),
  });
  if (!res.ok) throw new Error(await res.text());
  const data = await res.json();

  console.log("CHECKPOINT 1 (api.ts):", data, "raw_id type:", typeof data.raw_id);
  
  return data;
}

/** Get personalised recommendations for an existing user */
export async function getRecommendations(
    rawId: number,
    topN: number = 10

): Promise<RecommendResponse> {
    const res = await fetch(`${BASE_URL}/recommend`, {
        method: "POST", //used post instead of GET because /recommend payloads can grow
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({user_id: rawId, top_n: topN}),
    });

    if (!res.ok) throw new Error(await res.text());
    return res.json();
}


/**Get personalised recommendations for an exisitng user, by name (convenience entry point). */
export async function getRecommendationsByName(
  name: string,
  topN: number = 10
): Promise<RecommendResponse> {
  const res = await fetch(`${BASE_URL}/recommend/by-name`, {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({name, top_n: topN}),
  });

  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

/** Get popular movies — cold start for new users */

export async function GetPopular (
genre: string | null=null,
topN: number = 10,
): Promise<PopularResponse> {
    const res = await fetch(`${BASE_URL}/popular`, {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({ genre, top_n: topN}),
    });

    if (!res.ok) throw new Error(await res.text());
    return res.json();
}


/** Fetch poster + AI summary for a single movie (called per card) */
export async function getMovieDetails(movieId: number): Promise<MovieDetails> {
  const res = await fetch(`${BASE_URL}/movie/${movieId}/details`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

/** Submit a rating from a new user */
export async function submitRating(
  rawId: number,
  movieId: number,
  score: number,
  userName: string = "" //optional display name; backend stores in but doesn't use it for lookups
): Promise<RateResponse> {

  console.log("DEBUG rawId:", rawId, typeof rawId)

  const res = await fetch(`${BASE_URL}/rate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({raw_id: rawId, movie_id: movieId, score,userName}),
  });
  if (!res.ok) throw new Error(await res.text());
  const data = await res.json()

  console.log("DEBUG: Submit rating raw response ", data)
  
  return data
}


/** Get Hit@K and NDCG@K metrics */
export async function getMetrics(k: number = 10): Promise<MetricsResponse> {
  const res = await fetch(`${BASE_URL}/metrics?k=${k}&sample_users=200`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}


 
/** Trigger a manual retrain */
export async function triggerRetrain(rawId?: number): Promise<{ message: string; status: string }> {
  const res = await fetch(`${BASE_URL}/retrain`, { method: "POST" });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}