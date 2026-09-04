import { useEffect, useRef, useState } from "react";
import { api, type ChatResult, type Evaluation } from "../api";

/** 对话式复习（V1.1）—— 自然语言入口。 */

type QuestionCard = { question: string; kp_title: string; kp_content: string; review_reason: string };

type Msg = { id: number; role: "user" | "assistant"; text: string };

let nextId = 1;

export default function ChatPage() {
  const [msgs, setMsgs] = useState<Msg[]>([]);
  const [question, setQuestion] = useState<QuestionCard | null>(null);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [msgs, question]);

  const addMsg = (role: "user" | "assistant", text: string) =>
    setMsgs((m) => [...m, { id: nextId++, role, text }]);

  async function send(text?: string) {
    const content = (text ?? input).trim();
    if (!content || busy) return;
    setBusy(true);
    setInput("");
    addMsg("user", content);
    try {
      const res = await api<ChatResult>("POST", "/api/chat", { message: content });
      handleResult(res);
    } catch (e) {
      addMsg("assistant", `出错了：${(e as Error).message}`);
    } finally {
      setBusy(false);
    }
  }

  function handleResult(res: ChatResult) {
    switch (res.type) {
      case "question":
        // V1.3：LLM 的自然语言前言（如"给你出道题"）先出气泡，再钉住题目卡
        setQuestion(res.data);
        addMsg("assistant", res.text || `出题：${res.data.question}`);
        break;
      case "review_result": {
        addCard("eval", res.evaluation);
        if (res.next) {
          setQuestion(res.next);
          addMsg("assistant", res.text || `下一题：${res.next.question}`);
        } else {
          setQuestion(null);
          addMsg("assistant", res.text || "复习结束了。还想练或有别的需要告诉我。");
        }
        break;
      }
      case "analysis": {
        const txt = res.data.llm_report || "数据还不多，先多复习几轮再来看分析。";
        addMsg("assistant", res.text || "你的薄弱分析：");
        addCard("report", txt);
        break;
      }
      case "imported":
        addMsg("assistant", res.text || `已导入 ${res.data.count} 个知识点，可以开始复习咯。`);
        break;
      case "answer":
      case "chat":
      default:
        addMsg("assistant", res.text || "");
        break;
    }
  }

  function addCard(kind: "eval" | "report", payload: Evaluation | string) {
    setMsgs((m) => [...m, { id: nextId++, role: "assistant", text: "", _card: kind, _payload: payload } as Msg & { _card: string; _payload: unknown }]);
  }

  return (
    <div>
      <div className="page-header">
        <div className="page-title">对话</div>
        <div className="page-desc">自然语言交流——开始复习、提问、做分析、导资料</div>
      </div>

      <div className="chat-wrap">
        <div className="chat-view">
          {msgs.length === 0 && !question && (
            <div className="empty-state">
              <div className="empty-state-icon">💬</div>
              <div className="empty-state-text">
                打个招呼体验一下，比如：「复习一下」「什么是 GMP 模型」「我哪里薄弱」
              </div>
            </div>
          )}

          {msgs.map((m) =>
            (m as Msg & { _card?: string; _payload?: unknown })._card ? (
              <div className="chat-card" key={m.id}>
                {(m as Msg & { _card: string })._card === "eval"
                  ? <EvalCard ev={(m as Msg & { _payload: Evaluation })._payload as Evaluation} />
                  : <ReportCard text={(m as Msg & { _payload: string })._payload as string} />}
              </div>
            ) : (
              <div className={`bubble ${m.role === "user" ? "bubble-user" : "bubble-assistant"}`} key={m.id}>
                {m.text}
              </div>
            )
          )}
          {busy && <div className="bubble bubble-assistant bubble-typing">思考中…</div>}
          <div ref={bottomRef} />
        </div>

        {question && <QuestionBar q={question} />}

        <div className="chat-input">
          <textarea
            placeholder={question ? "输入你的回答，或提问/退出…" : "输入想说的，比如「开始复习」"}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && e.shiftKey) { e.preventDefault(); send(); }
              if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); }
            }}
          />
          <button className="btn btn-primary" onClick={() => send()} disabled={busy || !input.trim()}>
            发送
          </button>
        </div>
      </div>

      <style>{chatCss}</style>
    </div>
  );
}

// ===== 渲染子组件 =====

function QuestionBar({ q }: { q: QuestionCard }) {
  return (
    <div className="chat-question">
      <div className="chat-question-head">
        <span className="review-reason">{q.review_reason || "复习中"}</span>
        <span className="tag tag-blue">{q.kp_title}</span>
      </div>
      <div className="review-question">{q.question}</div>
      {q.kp_content && <div className="review-kp-content"><strong>知识点参考：</strong>{q.kp_content}</div>}
    </div>
  );
}

function EvalCard({ ev }: { ev: Evaluation }) {
  const sClass = ev.score >= 75 ? "score-high" : ev.score >= 60 ? "score-mid" : "score-low";
  const items = (arr?: string[]) =>
    arr && arr.length > 0 ? arr.map((i) => escapeHtml(i)) : "无";
  return (
    <div className="card">
      <div className="eval-header">
        <div className={`score-circle ${sClass}`}>{ev.score}</div>
        <div className="eval-score-text"><h3>评分</h3><p>满分 100 分</p></div>
      </div>
      {ev.comment && <div className="eval-comment">{ev.comment}</div>}
      <div className="eval-sections">
        <div className="eval-section strengths"><div className="eval-section-title">优点</div><ul dangerouslySetInnerHTML={{ __html: items(ev.strengths) }} /></div>
        <div className="eval-section weaknesses"><div className="eval-section-title">不足</div><ul dangerouslySetInnerHTML={{ __html: items(ev.weaknesses) }} /></div>
        {ev.missing_kps && ev.missing_kps.length > 0 && (
          <div className="eval-section missing"><div className="eval-section-title">缺失知识点</div><ul dangerouslySetInnerHTML={{ __html: items(ev.missing_kps) }} /></div>
        )}
      </div>
    </div>
  );
}

function ReportCard({ text }: { text: string }) {
  return <div className="card"><div className="report-text">{text}</div></div>;
}

function escapeHtml(s: string) {
  const d = document.createElement("div");
  d.textContent = s;
  return d.innerHTML;
}

const chatCss = `
.chat-wrap { display: flex; flex-direction: column; height: calc(100vh - 180px); min-height: 420px; }
.chat-view { flex: 1; overflow-y: auto; padding: 8px 4px 16px; display: flex; flex-direction: column; gap: 12px; }
.bubble { max-width: 78%; padding: 10px 14px; border-radius: 12px; line-height: 1.55; white-space: pre-wrap; word-break: break-word; font-size: 14px; }
.bubble-user { align-self: flex-end; background: var(--primary, #3b82f6); color: #fff; border-bottom-right-radius: 4px; }
.bubble-assistant { align-self: flex-start; background: var(--bg-secondary, #f1f5f9); color: var(--text-primary, #111); border-bottom-left-radius: 4px; }
.bubble-typing { color: var(--text-secondary, #888); font-style: italic; }
.chat-card { max-width: 96%; }
.chat-card .card { margin: 0; }
.chat-card ul { margin: 0; padding-left: 18px; }
.chat-question { background: var(--bg-secondary, #f1f5f9); border: 1px solid var(--border, #e2e8f0); border-radius: 10px; padding: 14px; margin: 4px 0 10px; }
.chat-question-head { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; margin-bottom: 8px; }
.report-text { white-space: pre-wrap; line-height: 1.7; font-size: 14px; }
.chat-input { display: flex; gap: 8px; padding: 10px 0 0; }
.chat-input textarea { flex: 1; resize: vertical; min-height: 44px; border-radius: 8px; padding: 10px 12px; font-size: 14px; border: 1px solid var(--border, #e2e8f0); background: var(--bg, #fff); color: var(--text-primary, #111); font-family: inherit; }
.chat-input button { align-self: flex-end; }
`;