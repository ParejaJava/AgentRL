"""对已启动的 API 执行健康检查和一轮真实 Agent SSE smoke。"""

from __future__ import annotations

import argparse
import json
import urllib.request
from uuid import uuid4


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--message", default="你好，请用一句话说明你能做什么")
    args = parser.parse_args()

    with urllib.request.urlopen(f"{args.base_url}/health", timeout=10) as response:
        health = json.loads(response.read())
    if health.get("status") != "ok":
        raise RuntimeError(f"健康检查失败：{health}")

    payload = json.dumps(
        {
            "message": args.message,
            "shopping_session_id": f"smoke-shopping-{uuid4().hex[:8]}",
            "buyer_id": "smoke-buyer",
            "thread_id": f"smoke-thread-{uuid4().hex[:8]}",
        }
    ).encode()
    request = urllib.request.Request(
        f"{args.base_url}/api/agent",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    seen_final = False
    with urllib.request.urlopen(request, timeout=180) as response:
        for raw_line in response:
            line = raw_line.decode().strip()
            if line.startswith("data: "):
                event = json.loads(line[6:])
                print(json.dumps(event, ensure_ascii=False))
                seen_final |= event.get("type") == "final.result"
    if not seen_final:
        raise RuntimeError("SSE 流未返回 final.result")


if __name__ == "__main__":
    main()
