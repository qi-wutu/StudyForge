import { useEffect, useState } from "react";
import { api, type Stats, type Session } from "../api";
import { useSession } from "../hooks/useSession";

export default function Dashboard() {
  const { sessionId } = useSession();
  const [stats, setStats] = useState<Stats | null>(null);
  const [sessionName, setSessionName] = useState("未设置");

  useEffect(() => {
    Promise.all([
      api<Stats>("GET", "/api/stats"),
      api<Session[]>("GET", "/api/sessions"),
    ])
      .then(([s, sessions]) => {
        setStats(s);
        const current = sessions.find((s) => String(s.id) === sessionId);
        if (current) setSessionName(current.name);
      })
      .catch(() => {});
  }, [sessionId]);

  return (
    <div>
      <div className="page-header">
        <div className="page-title">概览</div>
        <div className="page-desc">StudyForge AI 自适应复习系统</div>
      </div>

      <div className="stats-grid">
        <StatCard value={stats?.kp_count ?? 0} label="知识点" />
        <StatCard value={stats?.review_count ?? 0} label="答题记录" />
        <StatCard
          value={stats?.avg_score ?? "-"}
          label="平均分"
          color={
            stats?.avg_score
              ? stats.avg_score >= 60
                ? "var(--success)"
                : "var(--danger)"
              : undefined
          }
        />
        <StatCard value={stats?.doc_count ?? 0} label="导入文档" />
        <StatCard
          value={stats?.weak_kp_count ?? 0}
          label="薄弱知识点"
          color={
            stats?.weak_kp_count && stats.weak_kp_count > 0
              ? "var(--danger)"
              : undefined
          }
        />
      </div>

      <div className="card">
        <div className="card-title">当前会话：{sessionName}</div>
        <div style={{ marginTop: 6, fontSize: 13, color: "var(--text-secondary)" }}>
          想边聊边学？在「对话」里说「开始复习 / 什么是 GMP / 我哪里薄弱」都会自动分发
        </div>
        <div style={{ display: "flex", gap: 10, flexWrap: "wrap", marginTop: 12 }}>
          <a href="#/chat" className="btn btn-primary">
            💬 开始对话
          </a>
          <a href="#/review" className="btn btn-success">
            开始复习
          </a>
          <a href="#/import" className="btn">
            导入资料
          </a>
          <a href="#/analysis" className="btn">
            薄弱分析
          </a>
          <a href="#/sessions" className="btn">
            切换会话
          </a>
        </div>
      </div>

      <div className="card">
        <div className="card-title">快速入门</div>
        <ol style={{ paddingLeft: 20, fontSize: 14, lineHeight: 2 }}>
          <li>
            <strong>创建会话</strong> — 在"会话"页面创建一个新会话（如 golang）
          </li>
          <li>
            <strong>导入资料</strong> — 上传 Markdown 文件或粘贴内容，AI 自动提取知识点
          </li>
          <li>
            <strong>自然语言交流</strong> — 去「对话」说「开始复习」「什么是 GMP」，系统自动分发到复习/问答
          </li>
          <li>
            <strong>查看分析</strong> — 在「对话」说「我哪里薄弱」，或到「分析」页看报告
          </li>
        </ol>
      </div>
    </div>
  );
}

function StatCard({
  value,
  label,
  color,
}: {
  value: number | string;
  label: string;
  color?: string;
}) {
  return (
    <div className="stat-card">
      <div className="stat-value" style={color ? { color } : undefined}>
        {value}
      </div>
      <div className="stat-label">{label}</div>
    </div>
  );
}
