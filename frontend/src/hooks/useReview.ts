import { useReducer, useCallback } from "react";
import { api, type ReviewStartData, type ReviewNextData, type ReviewAnswerData, type Evaluation } from "../api";

// ===== State =====
export type ReviewPhase =
  | "idle"
  | "loading"
  | "answering"
  | "submitting"
  | "evaluated"
  | "ended";

export interface ReviewState {
  phase: ReviewPhase;
  questionNo: number;
  threadId: string | null;
  question: string;
  kpTitle: string;
  kpContent: string;
  reviewReason: string;
  evaluation: Evaluation | null;
  error: string | null;
}

const initialState: ReviewState = {
  phase: "idle",
  questionNo: 0,
  threadId: null,
  question: "",
  kpTitle: "",
  kpContent: "",
  reviewReason: "",
  evaluation: null,
  error: null,
};

// ===== Actions =====
type Action =
  | { type: "START" }
  | { type: "READY"; payload: ReviewStartData }
  | { type: "SUBMIT" }
  | { type: "EVALUATED"; payload: Evaluation }
  | { type: "NEXT" }
  | { type: "CONTINUE"; payload: ReviewNextData }
  | { type: "END" }
  | { type: "RESET" }
  | { type: "ERROR"; payload: string };

function reviewReducer(state: ReviewState, action: Action): ReviewState {
  switch (action.type) {
    case "START":
      return { ...initialState, phase: "loading" };
    case "READY":
      return {
        ...state,
        phase: "answering",
        questionNo: 1,
        threadId: action.payload.thread_id,
        question: action.payload.question,
        kpTitle: action.payload.kp_title,
        kpContent: action.payload.kp_content,
        reviewReason: action.payload.review_reason,
      };
    case "SUBMIT":
      return { ...state, phase: "submitting" };
    case "EVALUATED":
      return { ...state, phase: "evaluated", evaluation: action.payload };
    case "NEXT":
      return { ...state, phase: "loading" };
    case "CONTINUE":
      return {
        ...state,
        phase: "answering",
        questionNo: state.questionNo + 1,
        question: action.payload.question,
        kpTitle: action.payload.kp_title,
        kpContent: action.payload.kp_content,
        reviewReason: action.payload.review_reason,
        evaluation: null,
      };
    case "END":
      return { ...state, phase: "ended" };
    case "RESET":
      return initialState;
    case "ERROR":
      return { ...state, phase: "idle", error: action.payload };
    default:
      return state;
  }
}

// ===== Hook =====
export function useReview() {
  const [state, dispatch] = useReducer(reviewReducer, initialState);

  const start = useCallback(async () => {
    dispatch({ type: "START" });
    try {
      const data = await api<ReviewStartData>("POST", "/api/review/start");
      dispatch({ type: "READY", payload: data });
    } catch (e: unknown) {
      dispatch({ type: "ERROR", payload: (e as Error).message });
    }
  }, []);

  const submitAnswer = useCallback(
    async (answer: string) => {
      if (!state.threadId) return;
      dispatch({ type: "SUBMIT" });
      try {
        const data = await api<ReviewAnswerData>(
          "POST",
          `/api/review/${state.threadId}/answer`,
          { answer }
        );
        dispatch({ type: "EVALUATED", payload: data.evaluation });
        if (data.exit) {
          setTimeout(() => dispatch({ type: "END" }), 2000);
        }
      } catch (e: unknown) {
        dispatch({ type: "ERROR", payload: (e as Error).message });
      }
    },
    [state.threadId]
  );

  const nextQuestion = useCallback(async () => {
    if (!state.threadId) return;
    dispatch({ type: "NEXT" });
    try {
      const data = await api<ReviewNextData>(
        "GET",
        `/api/review/${state.threadId}/next`
      );
      dispatch({ type: "CONTINUE", payload: data });
    } catch (e: unknown) {
      dispatch({ type: "ERROR", payload: (e as Error).message });
    }
  }, [state.threadId]);

  const exitReview = useCallback(async () => {
    if (state.threadId) {
      try {
        await api("POST", `/api/review/${state.threadId}/exit`);
      } catch {
        /* ignore */
      }
    }
    dispatch({ type: "RESET" });
  }, [state.threadId]);

  return { state, start, submitAnswer, nextQuestion, exitReview };
}
