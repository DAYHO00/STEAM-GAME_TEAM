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

  // 로딩 상태 + 어떤 추천을 돌리는지 라벨
  const [loading, setLoading] = useState(false);
  const [currentLabel, setCurrentLabel] = useState("");

  // 게이지용 progress 상태 (0 ~ 100)
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

      // ✅ result가 비어 있으면 "아이디가 없습니다" 메시지 표시
      if (!data || (Array.isArray(data.result) && data.result.length === 0)) {
        setError("아이디가 없습니다.");
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

  const handleUserBased = () =>
    handleFetch(`${BASE_URL}/recommend/user/${userId}`, "User Based");

  const handleItemBased = () =>
    handleFetch(`${BASE_URL}/recommend/item/${appId}`, "Item Based");

  const handleModelBased = () =>
    handleFetch(`${BASE_URL}/recommend/model/${modelUserId}`, "Model Based");

  const handleUserBasedAdvanced = () =>
    handleFetch(
      `${BASE_URL}/recommend/user-advanced/${advUserId}`,
      "User Based Advanced"
    );

  const handleItemBasedAdvanced = () =>
    handleFetch(
      `${BASE_URL}/recommend/item-advanced/${advItemUserId}`,
      "Item Based Advanced"
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

  // loading 상태에 따라 게이지 애니메이션
  useEffect(() => {
    let intervalId;

    if (loading) {
      // 로딩 시작: 0부터 서서히 90%까지 증가
      setProgress(0);
      intervalId = setInterval(() => {
        setProgress((prev) => {
          if (prev >= 90) return prev;
          return prev + 2;
        });
      }, 200);
    } else {
      // 로딩 끝나면 0으로 리셋
      setProgress(0);
    }

    return () => {
      if (intervalId) clearInterval(intervalId);
    };
  }, [loading]);

  return (
    <div
      style={{
        minHeight: "100vh",
        background: "#f3f4f6",
        padding: "40px 20px",
        fontFamily: "'Segoe UI', sans-serif",
      }}
    >
      {/* HEADER */}
      <div style={{ maxWidth: "1200px", margin: "0 auto 35px" }}>
        <h1 style={{ fontSize: "2rem", margin: 0 }}>Steam Recommendation</h1>
        <p style={{ margin: "6px 0 0", color: "#6b7280" }}>
          User-based / Item-based / Advanced / Model-based 추천 결과를
          확인해보세요.
        </p>
      </div>

      {/* MAIN CARD */}
      <div
        style={{
          width: "100%",
          maxWidth: "1200px",
          margin: "0 auto",
          background: "white",
          borderRadius: "16px",
          padding: "32px",
          boxShadow: "0 10px 25px rgba(0,0,0,0.08)",
        }}
      >
        {/* 1행: User-based / Item-based */}
        <div
          style={{
            display: "flex",
            gap: "20px",
            width: "100%",
            flexWrap: "wrap",
            marginBottom: "25px",
          }}
        >
          <Card
            title="User-based 추천"
            desc="추천을 받을 user_id를 입력하세요."
            placeholder="예: 123456"
            value={userId}
            setValue={setUserId}
            onClick={handleUserBased}
            gradient="#2563eb, #4f46e5"
            disabled={loading}
          />

          <Card
            title="Item-based 추천"
            desc="추천을 받을 user_id를 입력하세요."
            placeholder="예: 570"
            value={appId}
            setValue={setAppId}
            onClick={handleItemBased}
            gradient="#059669, #10b981"
            disabled={loading}
          />
        </div>

        {/* 2행: User-based Advanced / Item-based Advanced */}
        <div
          style={{
            display: "flex",
            gap: "20px",
            width: "100%",
            flexWrap: "wrap",
            marginBottom: "25px",
          }}
        >
          <Card
            title="User-based Advanced"
            desc="추천을 받을 user_id를 입력하세요."
            placeholder="예: 123456"
            value={advUserId}
            setValue={setAdvUserId}
            onClick={handleUserBasedAdvanced}
            gradient="#4c1d95, #7c3aed"
            disabled={loading}
          />

          <Card
            title="Item-based Advanced"
            desc="추천을 받을 user_id를 입력하세요."
            placeholder="예: 123456"
            value={advItemUserId}
            setValue={setAdvItemUserId}
            onClick={handleItemBasedAdvanced}
            gradient="#0f766e, #14b8a6"
            disabled={loading}
          />
        </div>

        {/* 3행: Model-based 한 줄 전체 폭 */}
        <div
          style={{
            display: "flex",
            width: "100%",
            marginBottom: "10px",
          }}
        >
          <Card
            title="Model-based 추천"
            desc="추천을 받을 user_id를 입력하세요."
            placeholder="예: 123456"
            value={modelUserId}
            setValue={setModelUserId}
            onClick={handleModelBased}
            gradient="#d97706, #f59e0b"
            fullWidth
            disabled={loading}
          />
        </div>

        {/* LOADING BOX + 게이지 */}
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
            <div style={{ marginTop: 4, fontSize: "0.9rem", color: "#4b5563" }}>
              데이터 양이 많아 시간이 조금 걸릴 수 있습니다.
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

            {/* 게이지 바 */}
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
                  background: "#2563eb",
                  transition: "width 0.2s ease-out",
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
            <h2>추천 결과</h2>
            <table
              style={{
                width: "100%",
                borderCollapse: "collapse",
                marginTop: "12px",
                fontSize: "0.93rem",
              }}
            >
              <thead>
                <tr style={{ background: "#f3f4f6" }}>
                  <th style={{ padding: "12px", textAlign: "left" }}>#</th>
                  <th style={{ padding: "12px", textAlign: "left" }}>Title</th>
                  <th style={{ padding: "12px", textAlign: "right" }}>
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
                      borderBottom: "1px solid #eee",
                    }}
                  >
                    <td style={{ padding: "10px" }}>{idx + 1}</td>
                    <td style={{ padding: "10px" }}>{item.title}</td>
                    <td
                      style={{
                        padding: "10px",
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
        )}
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
}) {
  return (
    <div
      style={{
        flex: fullWidth ? "1 1 100%" : "1 1 400px",
        padding: "20px",
        background: "#f9fafb",
        borderRadius: "12px",
        border: "none", // ✅ 테두리 제거
      }}
    >
      <h3>{title}</h3>
      <p style={{ margin: "6px 0 12px", color: "#6b7280" }}>{desc}</p>
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
          }}
        >
          {disabled ? "계산 중..." : "실행"}
        </button>
      </div>
    </div>
  );
}

export default App;
