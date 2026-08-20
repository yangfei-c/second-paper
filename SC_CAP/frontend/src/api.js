function detailMessage(detail, status) {
  if (typeof detail === "string" && detail) return detail;
  if (Array.isArray(detail)) {
    return detail.map((item) => item.msg || JSON.stringify(item)).join("；");
  }
  if (detail && typeof detail === "object") {
    return detail.reason || detail.message || JSON.stringify(detail);
  }
  return `请求失败：${status}`;
}

async function request(path, payload) {
  let response;
  try {
    response = await fetch(`/api${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
  } catch {
    throw new Error("无法连接推荐服务，请确认后端已启动（端口 8081）。");
  }

  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(detailMessage(body.detail, response.status));
  return body;
}

export const firstRecommend = (payload) => request("/first/recommend", payload);
export const firstNext = (sessionId, payload) => request(`/first/${sessionId}/next`, payload);
export const firstFeedback = (payload) => request("/first/feedback", payload);
export const startSecond = (payload) => request("/second/sessions", payload);
export const sendSecondFeedback = (sessionId, payload) => request(`/second/sessions/${sessionId}/feedback`, payload);
export const finishSecond = (sessionId, payload) => request(`/second/sessions/${sessionId}/finish`, payload);
