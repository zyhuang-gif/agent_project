"""确定性的 send preflight。

原则（spec §3、§6）：
- P0 只发 Gmail 正文；显式要求飞书或群聊时直接拒发。
- 只允许 exact/alias 命中且有邮箱的 person；fuzzy/多命中/群聊/无邮箱一律拒发，
  由 send_report 返回需要确认的提示，不做部分发送。
- 输出结构固定：可发时 gmail_to/recipients 一致；不可发时 user_message 是给用户看的中文说明。
"""
from dataclasses import dataclass, field
from typing import Optional

from app.services.contact_service import contact_service, normalize_email


@dataclass(frozen=True)
class ResolvedRecipient:
    name: str
    email: str
    channel: str = "gmail"


@dataclass(frozen=True)
class PreflightResult:
    can_send: bool
    recipients: list[ResolvedRecipient] = field(default_factory=list)
    gmail_to: list[str] = field(default_factory=list)
    user_message: str = ""
    reason: str = ""


def _channels(channels: list[str]) -> list[str]:
    normalized = [c.strip().lower() for c in channels or [] if c and c.strip()]
    return normalized or ["gmail"]


async def resolve_send_targets(
    recipients: list[str], channels: list[str], identity: Optional[object] = None
) -> PreflightResult:
    wanted_channels = _channels(channels)
    if any(channel != "gmail" for channel in wanted_channels):
        return PreflightResult(
            can_send=False,
            user_message="飞书发送暂未启用；P0 只支持 Gmail 正文发送。",
            reason="unsupported_channel",
        )

    if not recipients:
        return PreflightResult(
            can_send=False,
            user_message="我还不知道要发给谁，请补充明确联系人。",
            reason="missing_recipient",
        )

    resolved: list[ResolvedRecipient] = []
    for raw_name in recipients:
        name = (raw_name or "").strip()
        result = await contact_service.resolve(name)
        if not result.matches:
            return PreflightResult(
                False,
                user_message=f"没有找到联系人：{name}。",
                reason="recipient_not_found",
            )
        if result.ambiguous or len(result.matches) != 1:
            return PreflightResult(
                False,
                user_message=f"联系人“{name}”存在多个或模糊匹配，需要确认后再发送。",
                reason="ambiguous_recipient",
            )

        match = result.matches[0]
        if match.kind == "group":
            return PreflightResult(
                False,
                user_message="P0 暂不支持群聊或飞书群发送，请去掉群聊后重试。",
                reason="group_unsupported",
            )
        if match.match_type not in ("exact", "alias"):
            return PreflightResult(
                False,
                user_message=f"联系人“{name}”只是模糊匹配，需要确认后再发送。",
                reason="fuzzy_recipient",
            )
        email = normalize_email(match.email)
        if not email:
            return PreflightResult(
                False,
                user_message=f"联系人“{name}”没有可用邮箱。",
                reason="missing_email",
            )
        resolved.append(ResolvedRecipient(name=match.name, email=email, channel="gmail"))

    return PreflightResult(
        can_send=True,
        recipients=resolved,
        gmail_to=[r.email for r in resolved],
    )
