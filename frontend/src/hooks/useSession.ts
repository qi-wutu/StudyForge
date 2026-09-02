import { useState, useCallback } from "react";
import { api, type Session } from "../api";

const SESSION_KEY = "sf-session-id";

function readSessionId(): string | null {
  return localStorage.getItem(SESSION_KEY);
}

function writeSessionId(id: number) {
  localStorage.setItem(SESSION_KEY, String(id));
}

export function useSession() {
  const [sessionId, setSessionId] = useState<string | null>(() => readSessionId());

  const switchSession = useCallback(async (id: number) => {
    await api<{ id: number; name: string }>("POST", `/api/sessions/${id}/switch`);
    writeSessionId(id);
    setSessionId(String(id));
  }, []);

  const createSession = useCallback(async (name: string): Promise<Session> => {
    const s = await api<Session>("POST", "/api/sessions", { name });
    writeSessionId(s.id);
    setSessionId(String(s.id));
    return s;
  }, []);

  const initSession = useCallback(async () => {
    if (!readSessionId()) {
      try {
        const s = await api<{ id: number; name: string }>("GET", "/api/sessions/current");
        writeSessionId(s.id);
        setSessionId(String(s.id));
      } catch {
        // 后端未启动，忽略
      }
    }
  }, []);

  return { sessionId, switchSession, createSession, initSession };
}
