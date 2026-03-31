import { useRef, useState, useEffect, useCallback } from "react";

//   VITE_API_URL=http://localhost:8000
const API_URL = import.meta.env.VITE_API_URL 

const BAR_COLOURS = [
  "#7F77DD", "#1D9E75", "#D85A30", "#D4537E", "#888780",
  "#378ADD", "#639922", "#BA7517", "#E24B4A", "#534AB7",
];

export default function App() {
  const canvasRef = useRef(null);
  const isDrawing = useRef(false);
  const [result, setResult]   = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.strokeStyle = "#000000";
    ctx.lineWidth   = 18;
    ctx.lineCap     = "round";
    ctx.lineJoin    = "round";
  }, []);

  const getPos = (e) => {
    const rect = canvasRef.current.getBoundingClientRect();
    const clientX = e.touches ? e.touches[0].clientX : e.clientX;
    const clientY = e.touches ? e.touches[0].clientY : e.clientY;
    return { x: clientX - rect.left, y: clientY - rect.top };
  };

  const startDraw = useCallback((e) => {
    isDrawing.current = true;
    const { x, y } = getPos(e);
    const ctx = canvasRef.current.getContext("2d");
    ctx.beginPath();
    ctx.moveTo(x, y);
  }, []);

  const draw = useCallback((e) => {
    if (!isDrawing.current) return;
    e.preventDefault();
    const { x, y } = getPos(e);
    const ctx = canvasRef.current.getContext("2d");
    ctx.lineTo(x, y);
    ctx.stroke();
  }, []);

  const stopDraw = useCallback(() => { isDrawing.current = false; }, []);

  const handleClear = () => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    setResult(null);
    setError(null);
  };

  const handlePredict = async () => {
    setLoading(true);
    setError(null);
    const imageData = canvasRef.current.toDataURL("image/png");
    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: imageData }),
      });
      if (!response.ok) throw new Error(`Server error: ${response.status}`);
      setResult(await response.json());
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={styles.page}>
      <h1 style={styles.title}>MNIST Digit Recogniser</h1>
      <p style={styles.subtitle}>Draw a digit (0–9) and hit Predict</p>
      <div style={styles.layout}>
        <div style={styles.card}>
          <p style={styles.cardLabel}>Draw here</p>
          <canvas
            ref={canvasRef} width={280} height={280} style={styles.canvas}
            onMouseDown={startDraw} onMouseMove={draw}
            onMouseUp={stopDraw}   onMouseLeave={stopDraw}
            onTouchStart={startDraw} onTouchMove={draw} onTouchEnd={stopDraw}
          />
          <div style={styles.buttons}>
            <button style={styles.btnPrimary} onClick={handlePredict} disabled={loading}>
              {loading ? "Predicting…" : "Predict"}
            </button>
            <button style={styles.btnSecondary} onClick={handleClear}>Clear</button>
          </div>
          {error && <p style={styles.error}>{error}</p>}
        </div>

        <div style={styles.card}>
          <p style={styles.cardLabel}>Result</p>
          {result ? (
            <>
              <div style={styles.digitBox}>
                <span style={styles.digit}>{result.digit}</span>
                <span style={styles.confidence}>
                  {(result.confidence * 100).toFixed(1)}% confident
                </span>
              </div>
              <p style={styles.chartTitle}>Scores for each digit</p>
              <div style={styles.barChart}>
                {result.scores.map((score, idx) => (
                  <div key={idx} style={styles.barRow}>
                    <span style={styles.barLabel}>{idx}</span>
                    <div style={styles.barTrack}>
                      <div style={{
                        ...styles.barFill,
                        width: `${(score * 100).toFixed(1)}%`,
                        background: BAR_COLOURS[idx],
                        opacity: idx === result.digit ? 1 : 0.45,
                      }}/>
                    </div>
                    <span style={styles.barValue}>{(score * 100).toFixed(1)}%</span>
                  </div>
                ))}
              </div>
            </>
          ) : (
            <div style={styles.placeholder}>
              <span style={styles.placeholderIcon}>🖊</span>
              <p style={styles.placeholderText}>Draw a digit then press Predict</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

const styles = {
  page: { minHeight: "100vh", background: "#f8f7f4", display: "flex", flexDirection: "column", alignItems: "center", padding: "40px 16px", fontFamily: "system-ui, sans-serif" },
  title: { fontSize: 28, fontWeight: 600, color: "#1a1a1a", margin: "0 0 6px" },
  subtitle: { color: "#666", margin: "0 0 32px", fontSize: 15 },
  layout: { display: "flex", gap: 24, flexWrap: "wrap", justifyContent: "center" },
  card: { background: "#fff", borderRadius: 16, padding: 24, boxShadow: "0 2px 12px rgba(0,0,0,0.08)", width: 320 },
  cardLabel: { fontSize: 12, fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.08em", color: "#999", margin: "0 0 12px" },
  canvas: { border: "2px solid #e0e0e0", borderRadius: 8, cursor: "crosshair", display: "block", touchAction: "none", width: "100%" },
  buttons: { display: "flex", gap: 10, marginTop: 14 },
  btnPrimary: { flex: 1, padding: "10px 0", background: "#534AB7", color: "#fff", border: "none", borderRadius: 8, fontSize: 15, fontWeight: 600, cursor: "pointer" },
  btnSecondary: { flex: 1, padding: "10px 0", background: "#f0eefc", color: "#534AB7", border: "none", borderRadius: 8, fontSize: 15, fontWeight: 600, cursor: "pointer" },
  error: { color: "#c0392b", fontSize: 13, marginTop: 8 },
  digitBox: { display: "flex", flexDirection: "column", alignItems: "center", padding: "16px 0", borderBottom: "1px solid #f0f0f0", marginBottom: 16 },
  digit: { fontSize: 80, fontWeight: 700, color: "#534AB7", lineHeight: 1 },
  confidence: { fontSize: 14, color: "#888", marginTop: 4 },
  chartTitle: { fontSize: 12, fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.08em", color: "#999", margin: "0 0 10px" },
  barChart: { display: "flex", flexDirection: "column", gap: 5 },
  barRow: { display: "flex", alignItems: "center", gap: 8 },
  barLabel: { width: 14, fontSize: 12, color: "#555", textAlign: "right", fontWeight: 600 },
  barTrack: { flex: 1, height: 10, background: "#f0f0f0", borderRadius: 5, overflow: "hidden" },
  barFill: { height: "100%", borderRadius: 5, transition: "width 0.4s ease" },
  barValue: { width: 42, fontSize: 11, color: "#888", textAlign: "right" },
  placeholder: { display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", height: 220, color: "#bbb" },
  placeholderIcon: { fontSize: 40, marginBottom: 12 },
  placeholderText: { fontSize: 14, textAlign: "center", lineHeight: 1.5 },
};