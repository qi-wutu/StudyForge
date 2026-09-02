/* ===== API 辅助函数 ===== */

const SESSION_KEY = "sf-session-id";

const API_BASE = "";

export async function api<T = unknown>(
  method: string,
  path: string,
  body?: unknown
): Promise<T> {
  const sid = localStorage.getItem(SESSION_KEY);
  if (sid && path.startsWith("/api/")) {
    const sep = path.includes("?") ? "&" : "?";
    path = `${path}${sep}session_id=${sid}`;
  }
  const opts: RequestInit = {
    method,
    headers: { "Content-Type": "application/json" },
  };
  if (body !== undefined) opts.body = JSON.stringify(body);
  const res = await fetch(API_BASE + path, opts);
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || `HTTP ${res.status}`);
  }
  return res.json();
}

// ===== 类型定义 =====

export interface Session {
  id: number;
  name: string;
  created_at: string;
}

export interface Stats {
  session_name: string;
  kp_count: number;
  review_count: number;
  avg_score: number | null;
  doc_count: number;
  weak_kp_count: number;
}

export interface KnowledgePoint {
  id: number;
  title: string;
  content: string;
  avg_score: number | null;
  review_count: number;
}

export interface ReviewStartData {
  thread_id: string;
  question: string;
  kp_title: string;
  kp_content: string;
  review_reason: string;
}

export interface ReviewNextData {
  question: string;
  kp_title: string;
  kp_content: string;
  review_reason: string;
}

export interface Evaluation {
  score: number;
  comment: string;
  strengths: string[];
  weaknesses: string[];
  missing_kps: string[];
}

export interface ReviewAnswerData {
  evaluation: Evaluation;
  exit: boolean;
}

export interface KPStat {
  title: string;
  avg_score: number;
  review_count: number;
  score_trend: string;
  top_missing_kps: string[];
}

export interface GlobalStats {
  total_records?: number;
  avg_score_all?: number;
  missing_kps_freq?: [string, number][];
  weakness_freq?: [string, number][];
}

export interface AnalyzeData {
  kp_stats: KPStat[];
  global_stats: GlobalStats;
  llm_report?: string;
}

// ===== 对话（V1.1 自然语言入口） =====

export type ChatResult =
  | { type: "chat"; text: string }
  | {
      type: "question";
      data: { question: string; kp_title: string; kp_content: string; review_reason: string };
    }
  | {
      type: "review_result";
      evaluation: Evaluation;
      exit: boolean;
      next: { question: string; kp_title: string; kp_content: string; review_reason: string } | null;
    }
  | { type: "answer"; text: string; has_context: boolean }
  | { type: "analysis"; data: AnalyzeData }
  | { type: "imported"; data: { count: number } };
