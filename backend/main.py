"""StudyForge FastAPI 应用

只做三件事：
1. 创建 FastAPI app
2. 挂载 CORS 中间件
3. 挂载路由 + 静态文件
"""

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.router import router
from storage.db import init_db

app = FastAPI(title="StudyForge API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.on_event("startup")
def startup():
    init_db()


frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
static_dir = frontend_dir / "dist"
if static_dir.exists():
    app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="frontend")
elif frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")
