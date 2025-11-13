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

      const res = await fetch(`${BASE_URL}/recommend/user/${userId}`);
      if (!res.ok) throw new Error("서버 응답 오류");

      const data = await res.json();
      setResult(data);
    } catch (err) {
      console.error("User-Based Error:", err);
      setError("Failed to fetch (User Based)");
    }
  };

  const handleItemBased = async () => {
    try {
      setError(null);

      const res = await fetch(`${BASE_URL}/recommend/item/${appId}`);
      if (!res.ok) throw new Error("서버 응답 오류");

      const data = await res.json();
      setResult(data);
    } catch (err) {
      console.error("Item-Based Error:", err);
      setError("Failed to fetch (Item Based)");
    }
  };

  return (
    <div style={{ padding: 40 }}>
      <h1>추천 테스트 화면</h1>

      <div>
        <h3>User-based 추천</h3>
        <input
          placeholder="user_id 입력"
          value={userId}
          onChange={(e) => setUserId(e.target.value)}
        />
        <button onClick={handleUserBased}>User Based 실행</button>
      </div>

      <div style={{ marginTop: 30 }}>
        <h3>Item-based 추천</h3>
        <input
          placeholder="app_id 입력"
          value={appId}
          onChange={(e) => setAppId(e.target.value)}
        />
        <button onClick={handleItemBased}>Item Based 실행</button>
      </div>

      {error && (
        <div style={{ marginTop: 20, color: "red" }}>
          <b>ERROR:</b> {error}
        </div>
      )}

      {result && (
        <div style={{ marginTop: 30 }}>
          <h2>결과</h2>
          <pre>{JSON.stringify(result, null, 2)}</pre>
        </div>
      )}
    </div>
  );
}

export default App;
