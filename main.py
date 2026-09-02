"""StudyForge 服务入口

只做启动服务器这一件事。
"""

import uvicorn
from dotenv import load_dotenv

load_dotenv()


def main():
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()
