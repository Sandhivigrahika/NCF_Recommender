"use client";
 
import { MovieResult } from "@/lib/api";
import MovieCard from "./MovieCard";
 
interface Props {
  movies: MovieResult[];
  title: string;
}
 
export default function MovieGrid({ movies, title }: Props) {
  if (movies.length === 0) return null;
 
  return (
    <div>
      <h2 style={{ color: "white", fontSize: "1.1rem", fontWeight: 600, marginBottom: "1rem" }}>
        {title}
      </h2>
      <div style={{
        display: "grid",
        gridTemplateColumns: "repeat(5, 1fr)",
        gap: "1rem",
      }}>
        {movies.map((movie, i) => (
          <MovieCard key={movie.movie_id ?? i} movie={movie} />
        ))}
      </div>
    </div>
  );
}
 