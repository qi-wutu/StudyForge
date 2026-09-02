"""项目配置

本文件负责从 .env 读取配置项。
其他地方不直接读 .env，都通过这里的变量来拿。

这样做的目的：
  - 配置集中管理，改一处就行
  - 如果将来换成环境变量 / 配置文件 / 配置中心，只改这里
  - 方便写测试时 mock 配置
"""

import os
from dotenv import load_dotenv

load_dotenv()  # 从 .env 文件加载到 os.environ

# ===== MySQL 连接配置 =====
MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "studyforge")

# ===== LLM 配置（任意 OpenAI 兼容大模型） =====
# 项目用 langchain-openai 的 ChatOpenAI —— 凡是支持 OpenAI 兼容协议的大模型都能用，
# 只需在 .env 里改 LLM_BASE_URL 和 LLM_MODEL，代码零改动。
# 适用：DeepSeek / 通义千问 Qwen / 智谱 GLM / Moonshot / MiniMax / OpenAI / Kimi 等。
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.deepseek.com")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-chat")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))
