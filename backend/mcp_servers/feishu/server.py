"""Feishu / Lark MCP server（P2）——只暴露发送能力。

工具集（tool_name_prefix=true 后对外可见为 feishu_*）:
- send_message_to_user
- send_message_to_group
- send_file_to_user
- send_file_to_group

file 工具是原子操作：先 upload_file 拿 file_key，再 send_file_message。
失败时抛 LarkAPIError → 转成 RuntimeError(dict payload)，FastMCP 将返回 isError=True，
send_guard 会跳过成功审计。
"""
from mcp.server.fastmcp import FastMCP

from mcp_servers.feishu import lark_client
from mcp_servers.feishu.lark_client import LarkAPIError


mcp = FastMCP("feishu")


def _reject_unsupported_msg_type(msg_type: str) -> None:
    """P2 只支持 text；post/interactive/… 明确拒绝，避免 LLM 或调用方
    传入无法验证的复杂 content 造成漏审计。"""
    if msg_type != "text":
        raise RuntimeError(
            {
                "error": "UNSUPPORTED_MSG_TYPE",
                "channel": "feishu",
                "message": f"P2 only supports msg_type=text, got {msg_type!r}",
            }
        )


@mcp.tool()
async def send_message_to_user(
    open_id: str,
    content: str,
    msg_type: str = "text",
) -> dict[str, str]:
    """Send a text message to a specific Feishu user by open_id."""
    _reject_unsupported_msg_type(msg_type)
    try:
        return await lark_client.send_text_message("open_id", open_id, content)
    except LarkAPIError as error:
        raise RuntimeError(error.to_dict()) from error


@mcp.tool()
async def send_message_to_group(
    chat_id: str,
    content: str,
    msg_type: str = "text",
) -> dict[str, str]:
    """Send a text message to a Feishu group chat by chat_id."""
    _reject_unsupported_msg_type(msg_type)
    try:
        return await lark_client.send_text_message("chat_id", chat_id, content)
    except LarkAPIError as error:
        raise RuntimeError(error.to_dict()) from error


@mcp.tool()
async def send_file_to_user(
    open_id: str,
    file_path: str,
    file_name: str | None = None,
) -> dict[str, str]:
    """Upload a Markdown artifact and send it as a file to a Feishu user.

    file_path must be an absolute path inside ARTIFACT_ROOT/send/. Validation
    runs before any network call so an invalid attachment never triggers
    Feishu billing / audit.
    """
    try:
        upload = await lark_client.upload_file(file_path, file_name)
        file_key = upload["file_key"]
        result = await lark_client.send_file_message("open_id", open_id, file_key)
        return {"message_id": result["message_id"], "file_key": file_key}
    except LarkAPIError as error:
        raise RuntimeError(error.to_dict()) from error


@mcp.tool()
async def send_file_to_group(
    chat_id: str,
    file_path: str,
    file_name: str | None = None,
) -> dict[str, str]:
    """Upload a Markdown artifact and send it as a file to a Feishu group."""
    try:
        upload = await lark_client.upload_file(file_path, file_name)
        file_key = upload["file_key"]
        result = await lark_client.send_file_message("chat_id", chat_id, file_key)
        return {"message_id": result["message_id"], "file_key": file_key}
    except LarkAPIError as error:
        raise RuntimeError(error.to_dict()) from error


if __name__ == "__main__":
    mcp.run()
