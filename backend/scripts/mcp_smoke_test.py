"""P0 Gmail 手工 smoke：走 MCPClientManager + send_guard 真发一封邮件到已录入的联系人。

使用步骤：
1. `.env` 里配置好 MCP_ENABLED=true / GMAIL_* / AGENT_ENGINE=graph。
2. `uv run python -m scripts.seed_contacts --name 张三 --email you@example.com --alias zhangsan`
3. `uv run python -m scripts.mcp_smoke_test --to you@example.com`
"""
import argparse
import asyncio

from app.mcp.client import mcp_manager
from app.mcp.context import mcp_call_context


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--to", required=True)
    parser.add_argument("--subject", default="MCP Gmail smoke test")
    parser.add_argument("--body", default="This is a local MCP Gmail smoke test.")
    args = parser.parse_args()

    await mcp_manager.start()
    try:
        tools = mcp_manager.get_tools(["gmail_send_email"])
        if not tools:
            raise SystemExit("gmail_send_email tool is not loaded")
        tool = tools[0]
        async with mcp_call_context(user_id="smoke", session_id="smoke"):
            result = await tool.ainvoke(
                {"to": [args.to], "subject": args.subject, "body": args.body}
            )
        print(result)
    finally:
        await mcp_manager.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
