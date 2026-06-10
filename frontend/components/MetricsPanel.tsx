"use client";
 
import { useEffect, useState } from "react";
import { getMetrics, MetricsResponse } from "@/lib/api";  // removed triggerRetrain
 
export default function MetricsPanel() {
  const [metrics, setMetrics] = useState<MetricsResponse | null>(null);
  const [loading, setLoading] = useState(true);

  const fetchMetrics = () => {
    setLoading(true);
    getMetrics(10)
      .then(setMetrics)
      .catch(() => setMetrics(null))
      .finally(() => setLoading(false));
  };
 
  useEffect(() => { fetchMetrics(); }, []);
 
  return (
    <div style={{ backgroundColor: "#18181b", border: "1px solid #27272a", borderRadius: "12px", padding: "1.5rem" }}>
 
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "1.25rem" }}>
        <h2 style={{ color: "white", fontWeight: 600, fontSize: "1.1rem" }}>Model Metrics</h2>
        <button
          onClick={fetchMetrics}
          disabled={loading}
          style={{
            fontSize: "0.75rem", padding: "0.5rem 1rem",
            backgroundColor: loading ? "#3f3f46" : "#27272a",
            color: loading ? "#71717a" : "#a1a1aa",
            border: "1px solid #3f3f46", borderRadius: "8px",
            cursor: loading ? "not-allowed" : "pointer", fontWeight: 500,
          }}
        >
          {loading ? "Loading..." : "↻ Refresh"}
        </button>
      </div>
 
      {loading ? (
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "1rem" }}>
          {[1, 2].map(i => (
            <div key={i} style={{ backgroundColor: "#27272a", borderRadius: "8px", height: "100px" }} />
          ))}
        </div>
      ) : metrics ? (
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "1rem" }}>
          <div style={{ backgroundColor: "#27272a", borderRadius: "8px", padding: "1rem" }}>
            <p style={{ color: "#71717a", fontSize: "0.75rem", marginBottom: "0.25rem" }}>Hit@{metrics.k}</p>
            <p style={{ color: "#34d399", fontSize: "2rem", fontWeight: 700, lineHeight: 1 }}>
              {(metrics.hit_at_k * 100).toFixed(1)}%
            </p>
            <p style={{ color: "#52525b", fontSize: "0.7rem", marginTop: "0.25rem" }}>
              relevant movie in top {metrics.k}
            </p>
            <div style={{ marginTop: "0.625rem", height: "3px", backgroundColor: "#3f3f46", borderRadius: "9999px", overflow: "hidden" }}>
              <div style={{ width: `${metrics.hit_at_k * 100}%`, height: "100%", backgroundColor: "#10b981" }} />
            </div>
          </div>
 
          <div style={{ backgroundColor: "#27272a", borderRadius: "8px", padding: "1rem" }}>
            <p style={{ color: "#71717a", fontSize: "0.75rem", marginBottom: "0.25rem" }}>NDCG@{metrics.k}</p>
            <p style={{ color: "#60a5fa", fontSize: "2rem", fontWeight: 700, lineHeight: 1 }}>
              {(metrics.ndcg_at_k * 100).toFixed(1)}%
            </p>
            <p style={{ color: "#52525b", fontSize: "0.7rem", marginTop: "0.25rem" }}>
              ranking quality score
            </p>
            <div style={{ marginTop: "0.625rem", height: "3px", backgroundColor: "#3f3f46", borderRadius: "9999px", overflow: "hidden" }}>
              <div style={{ width: `${metrics.ndcg_at_k * 100}%`, height: "100%", backgroundColor: "#3b82f6" }} />
            </div>
          </div>

          <p style={{ gridColumn: "span 2", color: "#52525b", fontSize: "0.7rem" }}>
            Evaluated on {metrics.test_users} users via leave-one-out · K={metrics.k}
          </p>
        </div>
      ) : (
        <p style={{ color: "#71717a", fontSize: "0.875rem" }}>Failed to load metrics.</p>
      )}
    </div>
  );
}