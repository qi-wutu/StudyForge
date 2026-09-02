import { useState } from "react";
import { api } from "../api";

type View = "options" | "pasting" | "loading" | "result";

interface KP {
  title: string;
  content: string;
}

export default function ImportPage() {
  const [view, setView] = useState<View>("options");
  const [title, setTitle] = useState("");
  const [content, setContent] = useState("");
  const [kps, setKps] = useState<KP[]>([]);
  const [docTitle, setDocTitle] = useState("");

  const handleFile = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setView("loading");
    try {
      const formData = new FormData();
      formData.append("file", file);
      const sid = localStorage.getItem("sf-session-id");
      let uploadUrl = "/api/import/file";
      if (sid) uploadUrl += `?session_id=${sid}`;
      const res = await fetch(uploadUrl, { method: "POST", body: formData });
      if (!res.ok) throw new Error((await res.json()).detail || "导入失败");
      const data = await res.json();
      setKps(data.knowledge_points || []);
      setDocTitle(file.name);
      setView("result");
    } catch (e: unknown) {
      alert("导入失败：" + (e as Error).message);
      setView("options");
    }
  };

  const handlePaste = async () => {
    if (!content.trim()) return;
    const t = title.trim() || "粘贴内容";
    setView("loading");
    try {
      const data = await api<{ knowledge_points: KP[] }>("POST", "/api/import", {
        content,
        title: t,
      });
      setKps(data.knowledge_points || []);
      setDocTitle(t);
      setView("result");
    } catch (e: unknown) {
      alert("导入失败：" + (e as Error).message);
      setView("pasting");
    }
  };

  const reset = () => {
    setView("options");
    setTitle("");
    setContent("");
    setKps([]);
  };

  return (
    <div>
      <div className="page-header">
        <div className="page-title">导入资料</div>
        <div className="page-desc">AI 自动提取知识点，BM25 语义去重</div>
      </div>

      {view === "options" && (
        <div className="import-options">
          <div
            className="import-pane"
            onClick={() => document.getElementById("fileInput")?.click()}
          >
            <div className="import-pane-icon">📄</div>
            <div className="import-pane-title">上传文件</div>
            <div className="import-pane-desc">支持 .md 或 .txt 格式</div>
            <input
              id="fileInput"
              type="file"
              accept=".md,.txt"
              style={{ display: "none" }}
              onChange={handleFile}
            />
          </div>
          <div className="import-pane" onClick={() => setView("pasting")}>
            <div className="import-pane-icon">📝</div>
            <div className="import-pane-title">粘贴内容</div>
            <div className="import-pane-desc">直接粘贴学习资料文本</div>
          </div>
        </div>
      )}

      {view === "pasting" && (
        <div className="card">
          <div className="card-title">粘贴内容</div>
          <textarea
            placeholder="在此粘贴学习资料内容..."
            value={content}
            onChange={(e) => setContent(e.target.value)}
            style={{ minHeight: 200 }}
          />
          <div style={{ marginTop: 10 }}>
            <input
              type="text"
              placeholder="标题（可选）"
              value={title}
              onChange={(e) => setTitle(e.target.value)}
              style={{ maxWidth: 300 }}
            />
          </div>
          <div style={{ marginTop: 12, display: "flex", gap: 10 }}>
            <button className="btn btn-primary" onClick={handlePaste}>
              导入
            </button>
            <button className="btn" onClick={reset}>
              返回
            </button>
          </div>
        </div>
      )}

      {view === "loading" && (
        <div className="loading">正在提取知识点...</div>
      )}

      {view === "result" && (
        <div className="card">
          <div className="card-title">导入完成</div>
          <div className="success-box">
            成功提取 <strong>{kps.length}</strong> 个知识点
          </div>
          <div className="kp-list">
            {kps.length === 0 && (
              <div style={{ color: "var(--text-secondary)" }}>
                未提取到知识点
              </div>
            )}
            {kps.map((kp, i) => (
              <div key={i} className="kp-item">
                <div className="kp-item-title">{kp.title}</div>
                <div
                  style={{
                    color: "var(--text-secondary)",
                    fontSize: 12,
                    marginTop: 2,
                  }}
                >
                  {kp.content}
                </div>
              </div>
            ))}
          </div>
          <div style={{ marginTop: 16 }}>
            <button className="btn btn-primary" onClick={reset}>
              继续导入
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
