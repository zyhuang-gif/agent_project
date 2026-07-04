"""确定性的 send preflight。

原则（spec §3、§6、P2 扩展）：
- 支持 channels=[]（默认 gmail，兼容 P0/P1）/ ["gmail"] / ["feishu"] /
  ["gmail","feishu"]；其它通道名 fail-closed。
- 只允许 exact/alias 命中的 person 或 group（group 仅在 feishu 通道下）。
- 任一目标 / 任一选中通道字段缺失（email、feishu_open_id、chat_id、匹配类型）
  → fail-closed 整体不发；绝不做部分发送。
- 输出结构固定：可发时 channels、gmail_to、feishu_users、feishu_groups 反映实际
  执行时的目标集合；不可发时 user_message 是给用户看的中文说明。
"""
from dataclasses import dataclass, field
from typing import Optional

from app.services.contact_service import contact_service, normalize_email


SUPPORTED_CHANNELS: frozenset[str] = frozenset({"gmail", "feishu"})


@dataclass(frozen=True)
class ResolvedRecipient:
    name: str
    email: Optional[str] = None
    feishu_open_id: Optional[str] = None
    chat_id: Optional[str] = None
    kind: str = "person"  # "person" | "group"
    # legacy alias（P0/P1 tests 依赖这个默认值）；描述该 recipient 主要参与的通道，
    # 混合通道模式下没有 authoritative 意义。
    channel: str = "gmail"


@dataclass(frozen=True)
class PreflightResult:
    can_send: bool
    channels: list[str] = field(default_factory=list)
    recipients: list[ResolvedRecipient] = field(default_factory=list)
    gmail_to: list[str] = field(default_factory=list)
    feishu_users: list[ResolvedRecipient] = field(default_factory=list)
    feishu_groups: list[ResolvedRecipient] = field(default_factory=list)
    user_message: str = ""
    reason: str = ""


def _channels(channels: list[str]) -> list[str]:
    normalized = [c.strip().lower() for c in channels or [] if c and c.strip()]
    return normalized or ["gmail"]


def _dedup(seq: list[str]) -> list[str]:
    """保留顺序去重，避免 ["gmail","gmail"] 展开出重复。"""
    seen: set[str] = set()
    out: list[str] = []
    for item in seq:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


async def resolve_send_targets(
    recipients: list[str], channels: list[str], identity: Optional[object] = None
) -> PreflightResult:
    wanted_channels = _dedup(_channels(channels))
    unsupported = [c for c in wanted_channels if c not in SUPPORTED_CHANNELS]
    if unsupported:
        return PreflightResult(
            can_send=False,
            user_message=f"暂不支持的通道：{', '.join(unsupported)}。",
            reason="unsupported_channel",
        )

    if not recipients:
        return PreflightResult(
            can_send=False,
            user_message="我还不知道要发给谁，请补充明确联系人。",
            reason="missing_recipient",
        )

    resolved: list[ResolvedRecipient] = []
    gmail_to: list[str] = []
    feishu_users: list[ResolvedRecipient] = []
    feishu_groups: list[ResolvedRecipient] = []

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
        if match.match_type not in ("exact", "alias"):
            return PreflightResult(
                False,
                user_message=f"联系人“{name}”只是模糊匹配，需要确认后再发送。",
                reason="fuzzy_recipient",
            )

        if match.kind == "group":
            # 群只能走飞书；显式包含 gmail 时 fail-closed（不部分发送）。
            if "gmail" in wanted_channels:
                return PreflightResult(
                    False,
                    user_message=(
                        f"“{name}”是飞书群，Gmail 不支持群聊。"
                        "请去掉群或只使用飞书通道后再重试。"
                    ),
                    reason="gmail_group_unsupported",
                )
            if "feishu" not in wanted_channels:
                return PreflightResult(
                    False,
                    user_message=(
                        f"“{name}”是飞书群，请在通道中包含飞书后再重试。"
                    ),
                    reason="group_needs_feishu_channel",
                )
            chat_id = (match.chat_id or "").strip()
            if not chat_id:
                return PreflightResult(
                    False,
                    user_message=f"飞书群“{name}”缺少 chat_id。",
                    reason="missing_chat_id",
                )
            group = ResolvedRecipient(
                name=match.name,
                kind="group",
                chat_id=chat_id,
                channel="feishu",
            )
            feishu_groups.append(group)
            resolved.append(group)
            continue

        # person：按被选中的通道分别校验字段
        email_val: Optional[str] = None
        open_id_val: Optional[str] = None

        if "gmail" in wanted_channels:
            email_val = normalize_email(match.email)
            if not email_val:
                return PreflightResult(
                    False,
                    user_message=f"联系人“{name}”没有可用邮箱。",
                    reason="missing_email",
                )
            gmail_to.append(email_val)

        if "feishu" in wanted_channels:
            open_id_val = (match.feishu_open_id or "").strip() or None
            if not open_id_val:
                return PreflightResult(
                    False,
                    user_message=f"联系人“{name}”没有可用飞书 open_id。",
                    reason="missing_open_id",
                )

        person = ResolvedRecipient(
            name=match.name,
            email=email_val,
            feishu_open_id=open_id_val,
            kind="person",
            channel="gmail" if "gmail" in wanted_channels else "feishu",
        )
        if open_id_val:
            feishu_users.append(person)
        resolved.append(person)

    return PreflightResult(
        can_send=True,
        channels=wanted_channels,
        recipients=resolved,
        gmail_to=gmail_to,
        feishu_users=feishu_users,
        feishu_groups=feishu_groups,
    )
