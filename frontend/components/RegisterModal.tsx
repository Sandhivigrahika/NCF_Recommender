"use client";

import { useState } from "react";
import { registerUser, RegisterResponse } from "@/lib/api";

interface Props {
  onRegistered: (user: RegisterResponse & { name: string }) => void;
}

export default function RegisterModal({ onRegistered }: Props) {
  const [name, setName]       = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState("");

  const handleSubmit = async () => {
    if (!name.trim()) return;
    setLoading(true);
    setError("");
    try {
      const res = await registerUser(name.trim());
      onRegistered({ ...res, name: name.trim().toLowerCase() });
    } catch (e: any) {
      setError("Registration failed. Is the API running?");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{
      position: "fixed", inset: 0, zIndex: 50,
      backgroundColor: "rgba(0,0,0,0.85)",
      display: "flex", alignItems: "center", justifyContent: "center",
      padding: "1rem",
    }}>
      <div style={{
        backgroundColor: "#18181b",
        border: "1px solid #27272a",
        borderRadius: "16px",
        padding: "2rem",
        width: "100%",
        maxWidth: "400px",
      }}>
        {/* Icon + title */}
        <div style={{ textAlign: "center", marginBottom: "1.5rem" }}>
          <span style={{ fontSize: "3rem" }}>🎬</span>
          <h2 style={{ color: "white", fontWeight: 700, fontSize: "1.25rem", marginTop: "0.5rem" }}>
            Welcome to NCF Recommender
          </h2>
          <p style={{ color: "#a1a1aa", fontSize: "0.875rem", marginTop: "0.375rem" }}>
            Enter your name to get personalised movie recommendations
          </p>
        </div>

        {/* Input */}
        <input
          type="text"
          placeholder="Your name (e.g. Neel)"
          value={name}
          onChange={(e) => setName(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && handleSubmit()}
          autoFocus
          style={{
            width: "100%",
            backgroundColor: "#27272a",
            border: "1px solid #3f3f46",
            borderRadius: "8px",
            padding: "0.75rem 1rem",
            color: "white",
            fontSize: "0.875rem",
            outline: "none",
            marginBottom: "1rem",
            boxSizing: "border-box",
          }}
        />

        {/* Error */}
        {error && (
          <p style={{ color: "#f87171", fontSize: "0.75rem", marginBottom: "0.75rem" }}>
            {error}
          </p>
        )}

        {/* Button */}
        <button
          onClick={handleSubmit}
          disabled={loading || !name.trim()}
          style={{
            width: "100%",
            padding: "0.75rem",
            backgroundColor: loading || !name.trim() ? "#3f3f46" : "#059669",
            color: loading || !name.trim() ? "#71717a" : "white",
            border: "none",
            borderRadius: "8px",
            fontSize: "0.875rem",
            fontWeight: 600,
            cursor: loading || !name.trim() ? "not-allowed" : "pointer",
          }}
        >
          {loading ? "Registering..." : "Get Started →"}
        </button>

        {/* Existing user note */}
        <p style={{ color: "#52525b", fontSize: "0.7rem", textAlign: "center", marginTop: "1rem" }}>
          Same name = same account. Your ratings are saved.
        </p>
      </div>
    </div>
  );
}