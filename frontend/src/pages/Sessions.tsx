import { useEffect, useState } from "react";
import { api, type Session } from "../api";
import { useSession } from "../hooks/useSession";

export default function SessionsPage() {
  const { sessionId, switchSession, createSession } = useSession();
  const [sessions, setSessions] = useState<Session[]>([]);
  const [newName, setNewName] = useState("");
  const [loading, setLoading] = useState(true);

  const load = async () => {
    setLoading(true);
    try {
      const data = await api<Session[]>("GET", "/api/sessions");
      setSessions(data);
    } catch {
      /* ignore */
    }
    setLoading(false);
  };

  useEffect(() => {
    load();
  }, []);

  const handleCreate = async () => {
    const name = newName.trim();
    if (!name) return;
    try {
      await createSession(name);
      setNewName("");
      load();
    } catch (e: unknown) {
      alert((e as Error).message);
    }
  };

  const handleSwitch = async (id: number) => {
    await switchSession(id);
    load();
  };

  return (
    <div>
      <div className="page-header">
        <div className="page-title">会话管理</div>
        <div className="page-desc">不同学习主题数据互相隔离</div>
      </div>

      <div className="card">
        <div className="card-title">创建新会话</div>
        <div style={{ display: "flex", gap: 10 }}>
          <input
            type="text"
            placeholder="如 golang"
            value={newName}
            onChange={(e) => setNewName(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleCreate()}
            style={{ flex: 1 }}
          />
          <button className="btn btn-primary" onClick={handleCreate}>
            创建
          </button>
        </div>
      </div>

      <div className="card">
        <div className="card-title">已有会话</div>
        {loading ? (
          <div className="loading">加载中...</div>
        ) : sessions.length === 0 ? (
          <div style={{ color: "var(--text-secondary)", fontSize: 14, padding: "12px 0" }}>
            暂无会话
          </div>
        ) : (
          sessions.map((s) => {
            const isCurrent = String(s.id) === sessionId;
            return (
              <div
                key={s.id}
                className={`session-item${isCurrent ? " current" : ""}`}
              >
                <div>
                  <div className="session-name">
                    {s.name}
                    {isCurrent && (
                      <span className="tag tag-blue" style={{ marginLeft: 8 }}>
                        当前
                      </span>
                    )}
                  </div>
                  <div className="session-meta">创建于 {s.created_at}</div>
                </div>
                {!isCurrent && (
                  <button
                    className="btn btn-sm"
                    onClick={() => handleSwitch(s.id)}
                  >
                    切换
                  </button>
                )}
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}
