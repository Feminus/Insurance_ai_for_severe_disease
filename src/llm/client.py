"""LLM 客户端(OpenAI 兼容协议,使用 requests 直连)。

为什么不用 openai SDK:某些环境下 httpx 装坏,导致 base_url 解析失败。
requests 更稳定。

环境变量:
- INSURANCE_LLM_BASE_URL
- INSURANCE_LLM_API_KEY
- INSURANCE_LLM_MODEL
- INSURANCE_LLM_TEMPERATURE(默认 0)
- INSURANCE_LLM_TIMEOUT(默认 30s)

自动加载: 如果 LLM_API_KEY 未设置,尝试从项目根目录的 .env 文件加载。

重试:对服务端瞬时错误(SSL / Timeout / Connection)做指数退避重试,默认 2 次。
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

import requests


# 兜底:如果用户没 source .env,自动从项目根目录加载
_ENV_LOADED = False


def _ensure_env_loaded() -> None:
    """如果 INSURANCE_LLM_API_KEY 还没设置,尝试从 .env 加载(项目根目录)。

    修复 UX 问题: 用户跑 `python -m scripts.batch_run ...` 时,
    经常忘记先 `source .env`,导致 API_KEY 配置丢失。
    """
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    if os.getenv("INSURANCE_LLM_API_KEY"):
        _ENV_LOADED = True
        return

    project_root = Path(__file__).resolve().parents[2]
    env_path = project_root / ".env"
    if env_path.exists():
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and not os.getenv(key):
                    os.environ[key] = value
    _ENV_LOADED = True


@dataclass
class LLMConfig:
    base_url: str
    api_key: str
    model: str
    temperature: float = 0.0
    timeout: float = 60.0  # 60s:大 payload + 真实场景下 30s 太短


def load_config() -> LLMConfig:
    _ensure_env_loaded()  # 兜底:如果没 source .env,自动加载
    return LLMConfig(
        base_url=os.getenv("INSURANCE_LLM_BASE_URL", "https://api.minimaxi.com/v1"),
        api_key=os.getenv("INSURANCE_LLM_API_KEY", ""),
        model=os.getenv("INSURANCE_LLM_MODEL", "MiniMax-Text-01"),
        temperature=float(os.getenv("INSURANCE_LLM_TEMPERATURE", "0")),
        timeout=float(os.getenv("INSURANCE_LLM_TIMEOUT", "60")),
    )


# =============================================================
# 重试配置
# =============================================================
# 这些异常是服务端瞬时波动,值得重试
RETRYABLE_EXCEPTIONS = (
    requests.exceptions.SSLError,
    requests.exceptions.Timeout,
    requests.exceptions.ConnectionError,
)
DEFAULT_MAX_RETRIES = 2  # 失败后重试 2 次(共 3 次调用)
RETRY_BACKOFF_BASE = 1.0  # 第一次重试前等 1s,第二次 2s


def _strip_code_fence(text: str) -> str:
    """去掉 ```json ... ``` / <think>...</think> 包裹。"""
    s = text.strip()
    # 去掉 <think>...</think> 块(MiniMax 模型特有)
    s = re.sub(r"<think>.*?</think>", "", s, flags=re.DOTALL).strip()
    # 去掉 ```json ... ``` 包裹
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```\s*$", "", s)
    return s.strip()


def chat(
    messages: list[dict],
    config: LLMConfig | None = None,
    *,
    max_retries: int = DEFAULT_MAX_RETRIES,
    **kwargs,
) -> str:
    """调 LLM chat-completion,返回文本(原始回复,可能含 markdown 包裹)。

    对服务端瞬时错误(SSL/Timeout/Connection)做指数退避重试。

    关键:传 response_format={"type": "json_object"} 强制 LLM 返回 JSON,
    避免 LLM 在 prompt 说"严格 JSON"时仍返回自然语言回答(导致 30s+ 长响应 + 解析失败)。
    """
    cfg = config or load_config()
    if not cfg.api_key:
        raise RuntimeError(
            "INSURANCE_LLM_API_KEY 未配置。请设置环境变量后再调用 LLM。"
        )

    url = cfg.base_url.rstrip("/") + "/chat/completions"
    params = {
        "model": cfg.model,
        "messages": messages,
        "temperature": kwargs.get("temperature", cfg.temperature),
        # 强制 JSON 输出:避免 LLM 返回自然语言长文(常见超时根因)
        "response_format": {"type": "json_object"},
    }
    headers = {
        "Authorization": f"Bearer {cfg.api_key}",
        "Content-Type": "application/json",
    }

    last_err: Exception | None = None
    total_attempts = max_retries + 1
    for attempt in range(1, total_attempts + 1):
        try:
            resp = requests.post(url, json=params, headers=headers, timeout=cfg.timeout)
            if resp.status_code != 200:
                raise RuntimeError(
                    f"LLM 调用失败 status={resp.status_code}: {resp.text[:300]}"
                )
            data = resp.json()
            return data["choices"][0]["message"]["content"] or ""
        except RETRYABLE_EXCEPTIONS as e:
            last_err = e
            if attempt < total_attempts:
                backoff = RETRY_BACKOFF_BASE * (2 ** (attempt - 1))
                time.sleep(backoff)
                continue
            # 重试耗尽
            raise RuntimeError(
                f"LLM 调用失败(已重试 {max_retries} 次): {type(e).__name__}: {e}"
            ) from e

    # 理论上走不到这里,保险起见
    raise RuntimeError(f"LLM 调用异常: {last_err}")


def chat_json(
    messages: list[dict],
    config: LLMConfig | None = None,
    *,
    max_retries: int = DEFAULT_MAX_RETRIES,
    **kwargs,
) -> dict:
    """调 LLM 并解析 JSON 返回 dict(自动剥 code fence + 容错 + 重试)。"""
    last_err: Exception | None = None
    total_attempts = max_retries + 1
    for attempt in range(1, total_attempts + 1):
        try:
            raw = chat(messages, config=config, max_retries=max_retries, **kwargs)
            cleaned = _strip_code_fence(raw)
            return json.loads(cleaned)
        except (ValueError, json.JSONDecodeError) as e:
            # JSON 解析失败 → 试一次兜底:从原文抓 {...} 块
            last_err = e
            if attempt < total_attempts:
                # 直接重试(可能是上次 response 被截断)
                time.sleep(RETRY_BACKOFF_BASE * (2 ** (attempt - 1)))
                continue
            raise ValueError(
                f"LLM 返回无法解析为 JSON(已重试 {max_retries} 次): {e}"
            ) from e
        except RETRYABLE_EXCEPTIONS as e:
            last_err = e
            if attempt < total_attempts:
                time.sleep(RETRY_BACKOFF_BASE * (2 ** (attempt - 1)))
                continue
            raise RuntimeError(
                f"LLM 调用失败(已重试 {max_retries} 次): {type(e).__name__}: {e}"
            ) from e

    raise RuntimeError(f"LLM 调用异常: {last_err}")
