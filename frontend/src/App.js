// frontend/src/App.js
import React, { useState, useEffect } from "react";

function App() {
  const [userId, setUserId] = useState("");
  const [appId, setAppId] = useState("");
  const [modelUserId, setModelUserId] = useState("");
  const [advUserId, setAdvUserId] = useState("");
  const [advItemUserId, setAdvItemUserId] = useState("");
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const [loading, setLoading] = useState(false);
  const [currentLabel, setCurrentLabel] = useState("");
  const [progress, setProgress] = useState(0);

  const BASE_URL = "http://127.0.0.1:8000";

  const handleFetch = async (url, label) => {
    try {
      setError(null);
      setResult(null);
      setCurrentLabel(label);
      setLoading(true);

      const res = await fetch(url);
      if (!res.ok) throw new Error("서버 응답 오류");

      const data = await res.json();
      console.log(`${label} result:`, data);

      if (!data || (Array.isArray(data.result) && data.result.length === 0)) {
        setError("아이디가 존재하지 않습니다.");
        setResult(null);
        return;
      }

      setResult(data);
    } catch (err) {
      console.error(err);
      setError(`Failed to fetch (${label})`);
    } finally {
      setLoading(false);
    }
  };

  // 🔹 label을 Card의 title과 동일하게 맞춰줌
  const handleUserBased = () =>
    handleFetch(`${BASE_URL}/recommend/user/${userId}`, "1. User-based");

  const handleItemBased = () =>
    handleFetch(`${BASE_URL}/recommend/item/${appId}`, "2. Item-based");

  const handleModelBased = () =>
    handleFetch(`${BASE_URL}/recommend/model/${modelUserId}`, "5. Model-based");

  const handleUserBasedAdvanced = () =>
    handleFetch(
      `${BASE_URL}/recommend/user-advanced/${advUserId}`,
      "3. User-based Advanced"
    );

  const handleItemBasedAdvanced = () =>
    handleFetch(
      `${BASE_URL}/recommend/item-advanced/${advItemUserId}`,
      "4. Item-based Advanced"
    );

  const formatScore = (item) => {
    const candidates = [
      item.score,
      item.similarity,
      item.sim,
      item.cosine,
      item.distance,
    ];
    for (const v of candidates) {
      if (v !== null && v !== undefined && !isNaN(Number(v))) {
        return Number(v).toFixed(5);
      }
    }
    for (const [key, value] of Object.entries(item)) {
      if (key === "title" || key === "name") continue;
      if (value !== null && value !== undefined && !isNaN(Number(value))) {
        return Number(value).toFixed(5);
      }
    }
    return "-";
  };

  useEffect(() => {
    let intervalId;
    if (loading) {
      setProgress(0);
      intervalId = setInterval(() => {
        setProgress((prev) => {
          if (prev >= 90) return prev;
          return prev + 2;
        });
      }, 200);
    } else {
      setProgress(0);
    }
    return () => {
      if (intervalId) clearInterval(intervalId);
    };
  }, [loading]);

  return (
    <div
      style={{
        position: "relative",
        minHeight: "100vh",
        margin: 0,
        padding: "40px 20px",
        fontFamily: "'Segoe UI', sans-serif",
        background:
          "radial-gradient(circle at 0% 0%, #c7ddf5 0, #e5e7ff 32%, #f8ecc0 72%, #fed7e2 100%)",
        overflow: "hidden",
        boxSizing: "border-box",
      }}
    >
      <style>{`
        @keyframes floatOrb {
          0% { transform: translate3d(0, 0, 0); }
          50% { transform: translate3d(18px, -22px, 0); }
          100% { transform: translate3d(0, 0, 0); }
        }
        .bg-orb {
          position: absolute;
          border-radius: 9999px;
          filter: blur(40px);
          opacity: 0.3;
          pointer-events: none;
        }
        .bg-orb-1 { width: 260px; height: 260px; top: -60px; left: -40px; background: radial-gradient(circle at 30% 30%, #60a5fa, #a855f7); animation: floatOrb 18s ease-in-out infinite; }
        .bg-orb-2 { width: 300px; height: 300px; right: -60px; top: 120px; background: radial-gradient(circle at 30% 30%, #f97316, #facc15); animation: floatOrb 20s ease-in-out infinite; }
        .bg-orb-3 { width: 320px; height: 320px; left: 10%; bottom: -140px; background: radial-gradient(circle at 30% 30%, #22c55e, #06b6d4); animation: floatOrb 22s ease-in-out infinite; }
      `}</style>

      <div className="bg-orb bg-orb-1" />
      <div className="bg-orb bg-orb-2" />
      <div className="bg-orb bg-orb-3" />

      <div style={{ position: "relative", zIndex: 1 }}>
        {/* HEADER */}
        <div style={{ maxWidth: "1200px", margin: "0 auto 35px" }}>
          <h1
            style={{
              fontSize: "2.4rem",
              margin: 0,
              color: "#0f172a",
              fontWeight: 800,
              letterSpacing: "-0.02em",
            }}
          >
            Steam Recommendation
          </h1>
          <p
            style={{
              margin: "6px 0 0",
              color: "#4b5563",
              fontSize: "0.96rem",
            }}
          >
            사용자 ID를 입력하고 실행 버튼을 눌러 보세요. 서로 다른 알고리즘이
            각기 다른 관점에서 게임을 추천합니다.
          </p>
        </div>

        {/* MAIN CARD */}
        <div
          style={{
            width: "100%",
            maxWidth: "1200px",
            margin: "0 auto",
            background: "rgba(255,255,255,0.92)",
            borderRadius: "18px",
            padding: "30px 28px 28px",
            boxShadow: "0 18px 45px rgba(15,23,42,0.18)",
            backdropFilter: "blur(10px)",
            border: "1px solid rgba(209,213,219,0.8)",
          }}
        >
          {/* 1행: User-based */}
          <div style={{ display: "flex", width: "100%", marginBottom: "20px" }}>
            <Card
              title="1. User-based"
              desc="나와 취향이 비슷한 사용자들의 행동을 분석해 빠르고 직관적으로 게임을 추천하는 방식."
              placeholder="예: 11764552"
              value={userId}
              setValue={setUserId}
              onClick={handleUserBased}
              gradient="#2563eb, #4f46e5"
              fullWidth
              disabled={loading}
              currentLabel={currentLabel}
            />
          </div>

          {/* 2행: Item-based */}
          <div style={{ display: "flex", width: "100%", marginBottom: "20px" }}>
            <Card
              title="2. Item-based"
              desc="내가 플레이한 게임과 유사한 게임 간 관계를 분석해 비슷한 분위기와 스타일의 게임을 추천하는 방식."
              placeholder="예: 11764552"
              value={appId}
              setValue={setAppId}
              onClick={handleItemBased}
              gradient="#059669, #10b981"
              fullWidth
              disabled={loading}
              currentLabel={currentLabel}
            />
          </div>

          {/* 3행: User-based Advanced */}
          <div style={{ display: "flex", width: "100%", marginBottom: "20px" }}>
            <Card
              title="3. User-based Advanced"
              desc="사용자 간의 선호와 비선호 패턴까지 함께 분석해 가장 가까운 취향의 유저들이 좋아한 게임을 추천하는 방식."
              placeholder="예: 11764552"
              value={advUserId}
              setValue={setAdvUserId}
              onClick={handleUserBasedAdvanced}
              gradient="#4c1d95, #7c3aed"
              fullWidth
              disabled={loading}
              currentLabel={currentLabel}
            />
          </div>

          {/* 4행: Item-based Advanced */}
          <div style={{ display: "flex", width: "100%", marginBottom: "20px" }}>
            <Card
              title="4. Item-based Advanced"
              desc="게임 간의 긍정 평가 패턴과 인기도를 함께 고려해 단순 인기보다 진짜 취향이 비슷한 게임을 정교하게 추천하는 방식."
              placeholder="예: 11764552"
              value={advItemUserId}
              setValue={setAdvItemUserId}
              onClick={handleItemBasedAdvanced}
              gradient="#0f766e, #14b8a6"
              fullWidth
              disabled={loading}
              currentLabel={currentLabel}
            />
          </div>

          {/* 5행: Model-based */}
          <div style={{ display: "flex", width: "100%", marginBottom: "10px" }}>
            <Card
              title="5. Model-based"
              desc="머신러닝으로 사용자와 게임의 상호작용을 학습해 선호도를 예측하고 가장 종합적으로 개인화된 추천을 제공하는 방식."
              placeholder="예: 11764552"
              value={modelUserId}
              setValue={setModelUserId}
              onClick={handleModelBased}
              gradient="#d97706, #f59e0b"
              fullWidth
              disabled={loading}
              currentLabel={currentLabel}
            />
          </div>

          {/* LOADING BOX */}
          {loading && (
            <div
              style={{
                padding: "14px 16px",
                marginBottom: "18px",
                background: "#eff6ff",
                borderRadius: "10px",
                border: "1px solid #bfdbfe",
                color: "#1d4ed8",
              }}
            >
              <div style={{ fontWeight: 600 }}>
                현재 계산 중입니다. 조금만 기다려주세요.
              </div>
              <div
                style={{
                  marginTop: 4,
                  fontSize: "0.9rem",
                  color: "#4b5563",
                }}
              >
                데이터 양이 많아 시간이 조금 걸릴 수 있습니다. ( 최소: 1초 ~
                최대: 2분 30초 )
              </div>
              {currentLabel && (
                <div
                  style={{
                    marginTop: 6,
                    fontSize: "0.82rem",
                    color: "#6b7280",
                  }}
                >
                  요청 유형: <b>{currentLabel}</b>
                </div>
              )}

              <div
                style={{
                  marginTop: 10,
                  height: 8,
                  background: "#dbeafe",
                  borderRadius: 9999,
                  overflow: "hidden",
                }}
              >
                <div
                  style={{
                    width: `${progress}%`,
                    height: "100%",
                    background:
                      "linear-gradient(90deg, #38bdf8, #6366f1, #a855f7)",
                    transition: "width 0.2s ease-out",
                    boxShadow: "0 0 6px rgba(59,130,246,0.8)",
                  }}
                />
              </div>
            </div>
          )}

          {/* ERROR */}
          {error && (
            <div
              style={{
                padding: "12px",
                background: "#fee2e2",
                border: "1px solid #fecaca",
                borderRadius: "8px",
                color: "#b91c1c",
                marginBottom: "20px",
              }}
            >
              <b>ERROR:</b> {error}
            </div>
          )}

          {/* RESULT TABLE */}
          {result && (
            <div style={{ marginTop: "20px" }}>
              <h2 style={{ margin: 0, fontSize: "1.1rem", color: "#111827" }}>
                추천 결과
              </h2>
              <div
                style={{
                  marginTop: "12px",
                  borderRadius: "12px",
                  border: "1px solid #e5e7eb",
                  overflow: "hidden",
                  background: "#f9fafb",
                }}
              >
                <table
                  style={{
                    width: "100%",
                    borderCollapse: "collapse",
                    fontSize: "0.93rem",
                  }}
                >
                  <thead>
                    <tr style={{ background: "#e5e7eb" }}>
                      <th style={{ padding: "10px 12px", textAlign: "left" }}>
                        #
                      </th>
                      <th style={{ padding: "10px 12px", textAlign: "left" }}>
                        Title
                      </th>
                      <th
                        style={{
                          padding: "10px 12px",
                          textAlign: "right",
                        }}
                      >
                        Similarity
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    {result.result?.map((item, idx) => (
                      <tr
                        key={idx}
                        style={{
                          background: idx % 2 === 0 ? "white" : "#f9fafb",
                          borderTop: "1px solid #e5e7eb",
                        }}
                      >
                        <td style={{ padding: "9px 12px" }}>{idx + 1}</td>
                        <td
                          style={{
                            padding: "9px 12px",
                            whiteSpace: "nowrap",
                            textOverflow: "ellipsis",
                            overflow: "hidden",
                            maxWidth: 0,
                          }}
                          title={item.title}
                        >
                          {item.title}
                        </td>
                        <td
                          style={{
                            padding: "9px 12px",
                            textAlign: "right",
                            fontFamily: "monospace",
                          }}
                        >
                          {formatScore(item)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

/** 공통 Card 컴포넌트 */
function Card({
  title,
  desc,
  placeholder,
  value,
  setValue,
  onClick,
  gradient,
  fullWidth = false,
  disabled = false,
  currentLabel,
}) {
  return (
    <div
      style={{
        flex: fullWidth ? "1 1 100%" : "1 1 400px",
        padding: "18px 18px 16px",
        background: "linear-gradient(135deg, #f9fafb, #eef2ff)",
        borderRadius: "14px",
        border: "1px solid #e5e7eb",
        boxShadow: "0 8px 20px rgba(148,163,184,0.22)",
      }}
    >
      <h3 style={{ margin: 0, fontSize: "1rem", color: "#111827" }}>{title}</h3>
      <p
        style={{ margin: "6px 0 12px", color: "#6b7280", fontSize: "0.85rem" }}
      >
        {desc}
      </p>
      <div style={{ display: "flex", gap: "10px" }}>
        <input
          placeholder={placeholder}
          value={value}
          onChange={(e) => setValue(e.target.value)}
          disabled={disabled}
          style={{
            flex: 1,
            padding: "10px",
            borderRadius: "8px",
            border: "1px solid #d1d5db",
            backgroundColor: disabled ? "#e5e7eb" : "white",
            fontSize: "0.85rem",
          }}
        />
        <button
          onClick={onClick}
          disabled={disabled}
          style={{
            padding: "10px 16px",
            background: disabled
              ? "#9ca3af"
              : `linear-gradient(135deg, ${gradient})`,
            color: "white",
            borderRadius: "8px",
            border: "none",
            cursor: disabled ? "not-allowed" : "pointer",
            fontSize: "0.85rem",
            fontWeight: 600,
            boxShadow: disabled ? "none" : "0 0 10px rgba(129,140,248,0.7)",
          }}
        >
          {disabled && currentLabel === title ? "실행 중..." : "실행"}
        </button>
      </div>
    </div>
  );
}

export default App;
