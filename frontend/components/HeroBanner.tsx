
"use client";
 
import { useEffect, useState } from "react";
import { QUOTES } from "@/lib/Quotes";
 
export default function HeroBanner() {
  const [current, setCurrent] = useState(0);
  const [fading, setFading]   = useState(false);
 
  useEffect(() => {
    const interval = setInterval(() => {
      setFading(true);
      setTimeout(() => {
        setCurrent((prev) => (prev + 1) % QUOTES.length);
        setFading(false);
      }, 600);
    }, 5000);
    return () => clearInterval(interval);
  }, []);
 
  const quote = QUOTES[current];
 
  return (
    <div style={{ position: "relative", width: "100%", height: "420px", overflow: "hidden" }}>
 
      {/* Blurred poster background */}
      <div style={{
        position: "absolute", inset: 0,
        backgroundImage: `url(${quote.poster})`,
        backgroundSize: "cover",
        backgroundPosition: "center",
        transform: "scale(1.05)",
        filter: "blur(4px) brightness(0.3)",
        opacity: fading ? 0 : 1,
        transition: "opacity 0.6s ease",
      }} />
 
      {/* Bottom gradient to page bg */}
      <div style={{
        position: "absolute", inset: 0,
        background: "linear-gradient(to bottom, rgba(0,0,0,0.1) 0%, transparent 40%, #09090b 100%)",
      }} />
 
      {/* App title — top left */}
      <div style={{ position: "absolute", top: "1.5rem", left: "2rem", zIndex: 10, display: "flex", alignItems: "center", gap: "0.75rem" }}>
        <span style={{ fontSize: "2rem" }}>🎬</span>
        <div>
          <h1 style={{ color: "white", fontWeight: 700, fontSize: "1.25rem", lineHeight: 1.2, textShadow: "0 2px 8px rgba(0,0,0,0.8)" }}>
            NCF Recommender
          </h1>
          <p style={{ color: "#d4d4d8", fontSize: "0.75rem", textShadow: "0 1px 4px rgba(0,0,0,0.8)" }}>
            Neural Collaborative Filtering · MovieLens 1M
          </p>
        </div>
      </div>
 
      {/* Quote — bottom left */}
      <div style={{
        position: "absolute", bottom: "2.5rem", left: "2rem", right: "2rem",
        zIndex: 10, maxWidth: "720px",
        opacity: fading ? 0 : 1,
        transition: "opacity 0.6s ease",
      }}>
        <blockquote style={{
          color: "white",
          fontSize: "clamp(1.1rem, 2.5vw, 1.6rem)",
          fontWeight: 300,
          fontStyle: "italic",
          lineHeight: 1.4,
          marginBottom: "0.625rem",
          textShadow: "0 2px 12px rgba(0,0,0,0.9)",
        }}>
          "{quote.text}"
        </blockquote>
        <p style={{ color: "#d4d4d8", fontSize: "0.875rem", fontWeight: 500, textShadow: "0 1px 6px rgba(0,0,0,0.8)" }}>
          — {quote.movie}, {quote.year}
        </p>
 
        {/* Progress dots */}
        <div style={{ display: "flex", gap: "6px", marginTop: "1rem" }}>
          {QUOTES.map((_, i) => (
            <div key={i} style={{
              height: "3px",
              borderRadius: "9999px",
              transition: "all 0.4s ease",
              width: i === current ? "24px" : "6px",
              backgroundColor: i === current ? "#10b981" : "rgba(255,255,255,0.3)",
            }} />
          ))}
        </div>
      </div>
    </div>
  );
}