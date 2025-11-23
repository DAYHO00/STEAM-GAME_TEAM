// frontend/src/App.js
import React, { useState } from "react";

function App() {
  const [userId, setUserId] = useState("");
  const [appId, setAppId] = useState("");
  const [modelUserId, setModelUserId] = useState("");
  const [advUserId, setAdvUserId] = useState("");
  const [advItemUserId, setAdvItemUserId] = useState("");
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const BASE_URL = "http://127.0.0.1:8000";

  const handleFetch = async (url, label) => {
    try {
      setError(null);
      setResult(null);
      const res = await fetch(url);
      if (!res.ok) throw new Error("서버 응답 오류");
      const data = await res.json();
      console.log(`${label} result:`, data);
      setResult(data);
    } catch (err) {
      console.error(err);
      setError(`Failed to fetch (${label})`);
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
        <h1 style={{ fontSize: "2rem", margin: 0 }}>Steam 추천 테스트</h1>
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
          />

          <Card
            title="Item-based 추천"
            desc="기준이 될 app_id를 입력하세요."
            placeholder="예: 570"
            value={appId}
            setValue={setAppId}
            onClick={handleItemBased}
            gradient="#059669, #10b981"
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
            desc="고급 User-based로 추천 받을 user_id를 입력하세요."
            placeholder="예: 123456"
            value={advUserId}
            setValue={setAdvUserId}
            onClick={handleUserBasedAdvanced}
            gradient="#4c1d95, #7c3aed"
          />

          <Card
            title="Item-based Advanced"
            desc="고급 Item-based로 추천 받을 user_id를 입력하세요."
            placeholder="예: 123456"
            value={advItemUserId}
            setValue={setAdvItemUserId}
            onClick={handleItemBasedAdvanced}
            gradient="#0f766e, #14b8a6"
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
            fullWidth // 🔸 한 줄 전체를 쓰도록
          />
        </div>

        {/* ---- ERROR ---- */}
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

        {/* ---- RESULT TABLE ---- */}
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
}) {
  return (
    <div
      style={{
        flex: fullWidth ? "1 1 100%" : "1 1 400px", // 🔸 fullWidth면 한 줄 전체
        padding: "20px",
        background: "#f9fafb",
        borderRadius: "12px",
        border: "1px solid #e5e7eb",
      }}
    >
      <h3>{title}</h3>
      <p style={{ margin: "6px 0 12px", color: "#6b7280" }}>{desc}</p>
      <div style={{ display: "flex", gap: "10px" }}>
        <input
          placeholder={placeholder}
          value={value}
          onChange={(e) => setValue(e.target.value)}
          style={{
            flex: 1,
            padding: "10px",
            borderRadius: "8px",
            border: "1px solid #d1d5db",
          }}
        />
        <button
          onClick={onClick}
          style={{
            padding: "10px 16px",
            background: `linear-gradient(135deg, ${gradient})`,
            color: "white",
            borderRadius: "8px",
            border: "none",
            cursor: "pointer",
          }}
        >
          실행
        </button>
      </div>
    </div>
  );
}

export default App;
