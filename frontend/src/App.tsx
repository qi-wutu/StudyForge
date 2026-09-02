import { useEffect, useState } from "react";
import { Routes, Route, NavLink, useLocation, useNavigate } from "react-router-dom";
import { useSession } from "./hooks/useSession";
import { api, type Session } from "./api";
import Dashboard from "./pages/Dashboard";
import Sessions from "./pages/Sessions";
import Import from "./pages/Import";
import Review from "./pages/Review";
import Analysis from "./pages/Analysis";

const navItems = [
  { path: "dashboard", label: "概览", icon: "■" },
  { path: "sessions", label: "会话", icon: "☰" },
  { path: "import", label: "导入", icon: "↑" },
  { path: "review", label: "复习", icon: "▶" },
  { path: "analysis", label: "分析", icon: "♻" },
];

export default function App() {
  const { sessionId, initSession } = useSession();
  const navigate = useNavigate();
  const location = useLocation();

  // 首次访问初始化 session
  useEffect(() => {
    initSession();
  }, [initSession]);

  // 默认重定向
  useEffect(() => {
    const hash = location.pathname;
    if (hash === "/" || hash === "") {
      navigate("/dashboard", { replace: true });
    }
  }, [location.pathname, navigate]);

  // 主题切换
  const toggleTheme = () => {
    const html = document.documentElement;
    const isDark = html.getAttribute("data-theme") === "dark";
    if (isDark) {
      html.removeAttribute("data-theme");
      localStorage.setItem("sf-theme", "light");
    } else {
      html.setAttribute("data-theme", "dark");
      localStorage.setItem("sf-theme", "dark");
    }
  };

  return (
    <div className="app-layout">
      <header className="topbar">
        <div className="topbar-inner">
          <div className="logo">
            <span className="logo-icon">&#9881;</span>
            <span className="logo-text">StudyForge</span>
          </div>
          <div className="topbar-right">
            <SessionBadge sessionId={sessionId} />
          </div>
        </div>
      </header>

      <nav className="sidebar">
        {navItems.map((item) => (
          <NavLink
            key={item.path}
            to={`/${item.path}`}
            className={({ isActive }) =>
              `nav-item${isActive ? " active" : ""}`
            }
          >
            <span className="nav-icon">{item.icon}</span>
            <span className="nav-label">{item.label}</span>
          </NavLink>
        ))}
        <div className="nav-spacer" />
        <button
          className="nav-item"
          onClick={toggleTheme}
          style={{
            cursor: "pointer",
            background: "none",
            border: "none",
            width: "100%",
            textAlign: "left",
            color: "var(--text-secondary)",
            fontSize: 13,
          }}
        >
          <span className="nav-icon">&#9788;</span>
          <span className="nav-label">切换主题</span>
        </button>
      </nav>

      <main className="main-content">
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/sessions" element={<Sessions />} />
          <Route path="/import" element={<Import />} />
          <Route path="/review" element={<Review />} />
          <Route path="/analysis" element={<Analysis />} />
          <Route
            path="*"
            element={
              <div className="empty-state">
                <div className="empty-state-text">页面不存在</div>
                <a href="#/" className="btn">
                  回首页
                </a>
              </div>
            }
          />
        </Routes>
      </main>
    </div>
  );
}

function SessionBadge({ sessionId }: { sessionId: string | null }) {
  const [name, setName] = useState("default");

  useEffect(() => {
    if (!sessionId) return;
    api<{ id: number; name: string }>("GET", "/api/sessions/current")
      .then((s) => setName(s.name))
      .catch(() => {});
  }, [sessionId]);

  return <span className="session-badge">{name}</span>;
}
