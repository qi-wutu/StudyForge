import { useState } from "react";
import { useReview } from "../hooks/useReview";
import { api, type KnowledgePoint } from "../api";
import { useEffect } from "react";

export default function ReviewPage() {
  const { state, start, submitAnswer, nextQuestion, exitReview } = useReview();
  const [kps, setKps] = useState<KnowledgePoint[]>([]);

  // 进入 idle 时查一次知识点数
  useEffect(() => {
    if (state.phase === "idle") {
      api<KnowledgePoint[]>("GET", "/api/knowledge-points")
        .then(setKps)
        .catch(() => {});
    }
  }, [state.phase]);

  switch (state.phase) {
    case "idle":
      return <ReviewStart kps={kps} onStart={start} />;
    case "loading":
      return <ReviewLoading text="出题中..." />;
    case "answering":
      return (
        <ReviewAnswer
          state={state}
          onSubmit={submitAnswer}
          onExit={exitReview}
        />
      );
    case "submitting":
      return <ReviewLoading text="AI 判分中..." />;
    case "evaluated":
      return (
        <ReviewEvaluated
          state={state}
          onNext={nextQuestion}
          onExit={exitReview}
        />
      );
    case "ended":
      return <ReviewEnded />;
    default:
      return null;
  }
}

// ===== Idle — 开始复习 =====
function ReviewStart({
  kps,
  onStart,
}: {
  kps: KnowledgePoint[];
  onStart: () => void;
}) {
  if (kps.length === 0) {
    return (
      <div>
        <div className="page-header">
          <div className="page-title">复习</div>
          <div className="page-desc">
            AI 出题 → 自由回答 → AI 评分 + 指出薄弱点
          </div>
        </div>
        <div className="card">
          <div className="empty-state">
            <div className="empty-state-icon">📚</div>
            <div className="empty-state-text">
              当前会话没有知识点，请先导入资料
            </div>
            <a href="#/import" className="btn btn-primary">
              去导入
            </a>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div>
      <div className="page-header">
        <div className="page-title">复习</div>
        <div className="page-desc">
          AI 出题 → 自由回答 → AI 评分 + 指出薄弱点
        </div>
      </div>
      <div className="card" style={{ textAlign: "center", padding: 40 }}>
        <div style={{ fontSize: 48, marginBottom: 12 }}>🎯</div>
        <div style={{ fontSize: 16, marginBottom: 20 }}>
          共 <strong>{kps.length}</strong> 个知识点可复习
        </div>
        <button className="btn btn-primary btn-lg" onClick={onStart}>
          开始复习
        </button>
      </div>
    </div>
  );
}

// ===== Loading / Submitting =====
function ReviewLoading({ text }: { text: string }) {
  return (
    <div>
      <div className="page-header">
        <div className="page-title">复习</div>
      </div>
      <div className="loading">{text}</div>
    </div>
  );
}

// ===== Answering — 答题 =====
function ReviewAnswer({
  state,
  onSubmit,
  onExit,
}: {
  state: {
    question: string;
    kpTitle: string;
    kpContent: string;
    reviewReason: string;
    questionNo: number;
  };
  onSubmit: (answer: string) => void;
  onExit: () => void;
}) {
  const [answer, setAnswer] = useState("");

  const handleSubmit = () => {
    if (!answer.trim()) return;
    onSubmit(answer);
  };

  return (
    <div>
      <div className="page-header">
        <div className="page-title">复习</div>
        <div className="page-desc">
          第 {state.questionNo} 题
        </div>
      </div>

      <div className="card">
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: 8,
            flexWrap: "wrap",
            marginBottom: 10,
          }}
        >
          <span className="review-reason">
            {state.reviewReason || "常规复习"}
          </span>
          <span className="tag tag-blue">{state.kpTitle}</span>
        </div>

        <div className="review-question">{state.question}</div>

        <div className="review-kp-content">
          <strong>知识点参考：</strong>
          {state.kpContent || ""}
        </div>

        <div className="review-answer-area">
          <textarea
            placeholder="输入你的回答..."
            value={answer}
            onChange={(e) => setAnswer(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && e.shiftKey) {
                e.preventDefault();
                handleSubmit();
              }
            }}
          />
        </div>

        <div className="review-actions">
          <button className="btn btn-primary" onClick={handleSubmit}>
            提交回答
          </button>
          <button className="btn btn-danger" onClick={onExit}>
            结束复习
          </button>
        </div>
      </div>
    </div>
  );
}

// ===== Evaluated — 评分展示 =====
function ReviewEvaluated({
  state,
  onNext,
  onExit,
}: {
  state: {
    evaluation: { score: number; comment: string; strengths: string[]; weaknesses: string[]; missing_kps?: string[] } | null;
    questionNo: number;
  };
  onNext: () => void;
  onExit: () => void;
}) {
  const ev = state.evaluation;
  if (!ev) return null;

  const sClass =
    ev.score >= 75 ? "score-high" : ev.score >= 60 ? "score-mid" : "score-low";

  const items = (arr?: string[]) => {
    if (!arr || arr.length === 0)
      return '<li style="color:var(--text-secondary);">无</li>';
    return arr.map((i) => `<li>${escapeHtml(i)}</li>`).join("");
  };

  return (
    <div>
      <div className="page-header">
        <div className="page-title">复习</div>
        <div className="page-desc">第 {state.questionNo} 题</div>
      </div>

      <div className="card">
        <div className="eval-header">
          <div className={`score-circle ${sClass}`}>{ev.score}</div>
          <div className="eval-score-text">
            <h3>评分</h3>
            <p>满分 100 分</p>
          </div>
        </div>

        {ev.comment && <div className="eval-comment">{ev.comment}</div>}

        <div className="eval-sections">
          <div className="eval-section strengths">
            <div className="eval-section-title">优点</div>
            <ul dangerouslySetInnerHTML={{ __html: items(ev.strengths) }} />
          </div>
          <div className="eval-section weaknesses">
            <div className="eval-section-title">不足</div>
            <ul dangerouslySetInnerHTML={{ __html: items(ev.weaknesses) }} />
          </div>
          {ev.missing_kps && ev.missing_kps.length > 0 && (
            <div className="eval-section missing">
              <div className="eval-section-title">缺失知识点</div>
              <ul
                dangerouslySetInnerHTML={{ __html: items(ev.missing_kps) }}
              />
            </div>
          )}
        </div>

        <div className="review-actions">
          <button className="btn btn-primary" onClick={onNext}>
            下一题
          </button>
          <button className="btn btn-danger" onClick={onExit}>
            结束复习
          </button>
        </div>
      </div>
    </div>
  );
}

// ===== Ended — 复习结束 =====
function ReviewEnded() {
  return (
    <div>
      <div className="page-header">
        <div className="page-title">复习</div>
      </div>
      <div className="card" style={{ textAlign: "center", padding: 60 }}>
        <div style={{ fontSize: 48, marginBottom: 12 }}>🎉</div>
        <div style={{ fontSize: 16, marginBottom: 20 }}>复习结束</div>
        <a href="#/" className="btn btn-primary">
          回首页
        </a>
      </div>
    </div>
  );
}

function escapeHtml(text: string) {
  const d = document.createElement("div");
  d.textContent = text;
  return d.innerHTML;
}
