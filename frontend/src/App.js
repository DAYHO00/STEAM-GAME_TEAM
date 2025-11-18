import React, { useState } from "react";

function App() {
  const [userId, setUserId] = useState("");
  const [appId, setAppId] = useState("");
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const BASE_URL = "http://127.0.0.1:8000";

  const handleUserBased = async () => {
    try {
      setError(null);
      setResult(null);

      const res = await fetch(`${BASE_URL}/recommend/user/${userId}`);
      if (!res.ok) throw new Error("서버 응답 오류");

      const data = await res.json();
      console.log("User-based result:", data);
      setResult(data);
    } catch (err) {
      console.error(err);
      setError("Failed to fetch (User Based)");
    }
  };

  const handleItemBased = async () => {
    try {
      setError(null);
      setResult(null);

      const res = await fetch(`${BASE_URL}/recommend/item/${appId}`);
      if (!res.ok) throw new Error("서버 응답 오류");

      const data = await res.json();
      console.log("Item-based result:", data);
      setResult(data);
    } catch (err) {
      console.error(err);
      setError("Failed to fetch (Item Based)");
    }
  };

  // 점수 필드를 안전하게 꺼내서 포맷팅하는 헬퍼
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
      {/* 상단 제목 영역 */}
      <div style={{ maxWidth: "1200px", margin: "0 auto 35px" }}>
        <h1 style={{ fontSize: "2rem", margin: 0 }}>Steam 추천 테스트</h1>
        <p style={{ margin: "6px 0 0", color: "#6b7280" }}>
          User-based / Item-based 추천 결과를 확인해보세요.
        </p>
      </div>

      {/* 메인 카드 */}
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
        {/* 입력 카드 영역 */}
        <div
          style={{
            display: "flex",
            gap: "20px",
            width: "100%",
            flexWrap: "wrap",
            marginBottom: "25px",
          }}
        >
          {/* User-based */}
          <div
            style={{
              flex: "1 1 400px",
              padding: "20px",
              background: "#f9fafb",
              borderRadius: "12px",
              border: "1px solid #e5e7eb",
            }}
          >
            <h3>User-based 추천</h3>
            <p style={{ margin: "6px 0 12px", color: "#6b7280" }}>
              추천을 받을 <b>user_id</b>를 입력하세요.
            </p>

            <div style={{ display: "flex", gap: "10px" }}>
              <input
                placeholder="예: 123456"
                value={userId}
                onChange={(e) => setUserId(e.target.value)}
                style={{
                  flex: 1,
                  padding: "10px",
                  borderRadius: "8px",
                  border: "1px solid",
                }}
              />
              <button
                onClick={handleUserBased}
                style={{
                  padding: "10px 16px",
                  background: "linear-gradient(135deg, #2563eb, #4f46e5)",
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

          {/* Item-based */}
          <div
            style={{
              flex: "1 1 400px",
              padding: "20px",
              background: "#f9fafb",
              borderRadius: "12px",
              border: "1px solid #e5e7eb",
            }}
          >
            <h3>Item-based 추천</h3>
            <p style={{ margin: "6px 0 12px", color: "#6b7280" }}>
              기준이 될 <b>user_id</b>를 입력하세요.
            </p>

            <div style={{ display: "flex", gap: "10px" }}>
              <input
                placeholder="예: 730"
                value={appId}
                onChange={(e) => setAppId(e.target.value)}
                style={{
                  flex: 1,
                  padding: "10px",
                  borderRadius: "8px",
                  border: "1px solid #d1d5db",
                }}
              />
              <button
                onClick={handleItemBased}
                style={{
                  padding: "10px 16px",
                  background: "linear-gradient(135deg, #059669, #10b981)",
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
        </div>

        {/* 에러 메시지 */}
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

        {/* 추천 결과 테이블 */}
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

export default App;
