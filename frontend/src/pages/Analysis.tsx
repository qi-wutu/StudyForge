import { useEffect, useState } from "react";
import { api, type AnalyzeData, type GlobalStats } from "../api";

export default function Analysis() {
  const [data, setData] = useState<AnalyzeData | null>(null);

  useEffect(() => {
    api<AnalyzeData>("GET", "/api/analyze")
      .then(setData)
      .catch(() => {});
  }, []);

  if (!data) {
    return <div className="loading">正在生成分析报告...</div>;
  }

  const kp_stats = data.kp_stats || [];
  const globalStats: GlobalStats = data.global_stats || {};
  const llm_report = data.llm_report || "";
  const missingFreq = globalStats.missing_kps_freq ?? [];
  const weakFreq = globalStats.weakness_freq ?? [];
  const weak = kp_stats.filter((s) => s.avg_score < 60);
  const mid = kp_stats.filter((s) => s.avg_score >= 60 && s.avg_score < 75);
  const strong = kp_stats.filter((s) => s.avg_score >= 75);

  return (
    <div>
      <div className="page-header">
        <div className="page-title">薄弱分析报告</div>
        <div className="page-desc">
          基于 {globalStats.total_records || 0} 条答题记录
        </div>
      </div>

      <div className="stats-grid">
        <StatCard value={globalStats.total_records || 0} label="总答题数" />
        <StatCard
          value={globalStats.avg_score_all ?? "-"}
          label="全局平均分"
        />
        <StatCard value={weak.length} label='薄弱（&lt;60）' color="var(--danger)" />
        <StatCard
          value={mid.length}
          label="待加强（60-75）"
          color="var(--warning)"
        />
        <StatCard
          value={strong.length}
          label="良好（&ge;75）"
          color="var(--success)"
        />
      </div>

      {weak.length > 0 && (
        <div className="card">
          <div className="card-title">薄弱知识点排行榜</div>
          {weak.map((s) => (
            <div key={s.title}>
              <div className="weak-bar">
                <div className="weak-bar-label">{s.title}</div>
                <div style={{ flex: 1 }}>
                  <div
                    style={{
                      background: "var(--gray-100)",
                      borderRadius: 4,
                      overflow: "hidden",
                    }}
                  >
                    <div
                      className="weak-bar-fill"
                      style={{
                        width: `${(s.avg_score / 100) * 100}%`,
                        background: "var(--danger)",
                      }}
                    />
                  </div>
                </div>
                <div
                  className="weak-bar-score"
                  style={{ color: "var(--danger)" }}
                >
                  {s.avg_score}
                </div>
                <TrendBadge trend={s.score_trend} />
              </div>
              {s.top_missing_kps?.length > 0 && (
                <div
                  style={{
                    margin: "0 0 8px 112px",
                    fontSize: 12,
                    color: "var(--text-secondary)",
                  }}
                >
                  缺失：
                  {s.top_missing_kps.map((m) => (
                    <span
                      key={m}
                      className="tag tag-gray"
                      style={{ margin: "0 2px" }}
                    >
                      {m}
                    </span>
                  ))}
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      {mid.length > 0 && (
        <div className="card">
          <div className="card-title">待加强知识点</div>
          {mid.map((s) => (
            <div key={s.title} className="weak-bar">
              <div className="weak-bar-label">{s.title}</div>
              <div style={{ flex: 1 }}>
                <div
                  style={{
                    background: "var(--gray-100)",
                    borderRadius: 4,
                    overflow: "hidden",
                  }}
                >
                  <div
                    className="weak-bar-fill"
                    style={{
                      width: `${s.avg_score}%`,
                      background: "var(--warning)",
                    }}
                  />
                </div>
              </div>
              <div
                className="weak-bar-score"
                style={{ color: "var(--warning)" }}
              >
                {s.avg_score}
              </div>
              <span
                style={{
                  fontSize: 12,
                  minWidth: 60,
                  color: "var(--text-secondary)",
                }}
              >
                {s.score_trend || ""}
              </span>
            </div>
          ))}
        </div>
      )}

      {missingFreq && missingFreq.length > 0 && (
        <div className="card">
          <div className="card-title">高频缺失知识点 Top 10</div>
          <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
            {missingFreq?.map(([item, count]) => (
              <span
                key={item}
                className="tag tag-gray"
                style={{ fontSize: 13, padding: "4px 12px" }}
              >
                {item} <strong>&times;{count}</strong>
              </span>
            ))}
          </div>
        </div>
      )}

      {weakFreq && weakFreq.length > 0 && (
        <div className="card">
          <div className="card-title">高频弱点 Top 10</div>
          <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
            {weakFreq?.map(([item, count]) => (
              <span
                key={item}
                className="tag tag-gray"
                style={{ fontSize: 13, padding: "4px 12px" }}
              >
                {item} <strong>&times;{count}</strong>
              </span>
            ))}
          </div>
        </div>
      )}

      {llm_report && (
        <div className="card">
          <div className="card-title">AI 分析报告</div>
          <div className="llm-report">{llm_report}</div>
        </div>
      )}

      {!weak.length && !mid.length && !kp_stats.length && (
        <div className="card">
          <div className="empty-state">
            <div className="empty-state-text">
              暂无答题记录，开始复习后才有分析数据
            </div>
            <a href="#/review" className="btn btn-primary">
              去复习
            </a>
          </div>
        </div>
      )}
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

function TrendBadge({ trend }: { trend?: string }) {
  if (!trend) return null;
  const cls = trend.includes("上升")
    ? "trend-up"
    : trend.includes("下降")
      ? "trend-down"
      : "trend-stable";
  return (
    <span
      className={cls}
      style={{ fontSize: 12, minWidth: 60 }}
    >
      {trend}
    </span>
  );
}
