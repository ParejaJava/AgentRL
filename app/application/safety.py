"""与具体 Agent 框架无关的输入输出安全策略。"""

from __future__ import annotations

import re
from dataclasses import dataclass


class UnsafePromptError(ValueError):
    """输入明确要求越权读取系统指令或密钥时抛出。"""


@dataclass(frozen=True, slots=True)
class SafetyPolicy:
    """执行保守的提示词注入拦截和敏感输出脱敏。"""

    max_input_chars: int = 20_000

    _INJECTION_PATTERNS = (
        re.compile(r"(?is)ignore\s+(all\s+)?previous\s+instructions"),
        re.compile(r"(?is)(reveal|print|show).{0,40}(system\s+prompt|api[_ -]?key)"),
        re.compile(r"(?is)(忽略|无视).{0,20}(之前|以上).{0,20}(指令|提示词)"),
        re.compile(r"(?is)(输出|泄露|展示).{0,20}(系统提示词|api[_ -]?key|密钥)"),
    )
    _SECRET_PATTERNS = (
        (re.compile(r"(?i)sk-[a-z0-9_-]{16,}"), "[REDACTED]"),
        (
            re.compile(
                r"(?i)((?:api[_ -]?key|secret[_ -]?key)\s*[:=]\s*)[^\s,;]+"
            ),
            r"\1[REDACTED]",
        ),
    )

    def validate_input(self, message: str) -> None:
        """拒绝过长输入和明确的越权提示词注入。"""

        if len(message) > self.max_input_chars:
            raise UnsafePromptError("输入超过平台允许的最大长度")
        if any(pattern.search(message) for pattern in self._INJECTION_PATTERNS):
            raise UnsafePromptError("检测到试图覆盖系统规则或读取敏感配置的输入")

    def sanitize_output(self, text: str) -> str:
        """对最终输出中的常见密钥形态做确定性脱敏。"""

        sanitized = text
        for pattern, replacement in self._SECRET_PATTERNS:
            sanitized = pattern.sub(replacement, sanitized)
        return sanitized
