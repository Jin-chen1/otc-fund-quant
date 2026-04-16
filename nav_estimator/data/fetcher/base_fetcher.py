"""基础数据获取器。"""

import hashlib
import json
import os
import threading
import time
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable

from loguru import logger
import pandas as pd
import requests

from ..cache.cache_manager import CacheManager
from ...config.settings import PROXY_MODE


class BaseFetcher:
    """基础数据获取器，提供缓存装饰器和通用方法。"""

    cache = CacheManager()
    _request_state = threading.local()
    PROXY_ENV_KEYS = (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    )
    PROXY_ERROR_SIGNATURES = (
        "ProxyError",
        "Unable to connect to proxy",
        "Cannot connect to proxy",
    )
    NON_RETRYABLE_ERROR_SIGNATURES = (
        "资产配置",
        "股票仓位信息",
        "比例字段",
        "报告期字段",
        "映射到多个A股指数代码",
        "无法映射到A股指数代码",
        "指数名称不能为空",
    )

    @staticmethod
    def _normalize_for_cache(value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, dict):
            return {
                str(key): BaseFetcher._normalize_for_cache(val)
                for key, val in sorted(value.items(), key=lambda item: str(item[0]))
            }
        if isinstance(value, (list, tuple)):
            return [BaseFetcher._normalize_for_cache(item) for item in value]
        if isinstance(value, set):
            normalized_items = [BaseFetcher._normalize_for_cache(item) for item in value]
            return sorted(normalized_items, key=lambda item: str(item))
        if hasattr(value, "isoformat"):
            try:
                return value.isoformat()
            except Exception:
                pass
        return str(value)

    @staticmethod
    def _build_cache_key(func: Callable, args: tuple, kwargs: dict) -> str:
        normalized_args = []
        for index, arg in enumerate(args):
            if index == 0 and hasattr(arg, "__class__"):
                normalized_args.append({"__instance_class__": arg.__class__.__name__})
            else:
                normalized_args.append(BaseFetcher._normalize_for_cache(arg))

        normalized_kwargs = {
            str(key): BaseFetcher._normalize_for_cache(value)
            for key, value in sorted(kwargs.items(), key=lambda item: str(item[0]))
        }

        payload = {"args": normalized_args, "kwargs": normalized_kwargs}
        payload_text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        payload_hash = hashlib.sha256(payload_text.encode("utf-8")).hexdigest()
        return f"{func.__module__}.{func.__name__}:{payload_hash}"

    @staticmethod
    def _normalize_error_message(error: Exception) -> str:
        return " ".join(str(error).split())

    @staticmethod
    def _is_empty_dataframe(value: Any) -> bool:
        return isinstance(value, pd.DataFrame) and value.empty

    @staticmethod
    def _is_proxy_related_error(error: Exception) -> bool:
        message = BaseFetcher._normalize_error_message(error)
        return any(signature in message for signature in BaseFetcher.PROXY_ERROR_SIGNATURES)

    @staticmethod
    def _is_non_retryable_error(error: Exception) -> bool:
        message = BaseFetcher._normalize_error_message(error)
        return any(signature in message for signature in BaseFetcher.NON_RETRYABLE_ERROR_SIGNATURES)

    @staticmethod
    def _is_direct_mode_latched() -> bool:
        return bool(getattr(BaseFetcher._request_state, "direct_mode_latched", False))

    @staticmethod
    def _set_direct_mode_latched(value: bool):
        BaseFetcher._request_state.direct_mode_latched = value

    @staticmethod
    def _get_strict_mode_latched() -> bool | None:
        return getattr(BaseFetcher._request_state, "strict_mode_latched", None)

    @staticmethod
    def _set_strict_mode_latched(value: bool | None):
        BaseFetcher._request_state.strict_mode_latched = value

    @staticmethod
    def _is_retry_log_suppressed() -> bool:
        return bool(getattr(BaseFetcher._request_state, "suppress_retry_logs", False))

    @staticmethod
    def _set_retry_log_suppressed(value: bool):
        BaseFetcher._request_state.suppress_retry_logs = value

    @staticmethod
    @contextmanager
    def request_scope(strict: bool | None = None):
        previous_direct_mode = BaseFetcher._is_direct_mode_latched()
        previous_strict_mode = BaseFetcher._get_strict_mode_latched()
        BaseFetcher._set_direct_mode_latched(False)
        BaseFetcher._set_strict_mode_latched(strict)
        try:
            yield
        finally:
            BaseFetcher._set_direct_mode_latched(previous_direct_mode)
            BaseFetcher._set_strict_mode_latched(previous_strict_mode)

    @staticmethod
    @contextmanager
    def suppress_retry_logs():
        previous_value = BaseFetcher._is_retry_log_suppressed()
        BaseFetcher._set_retry_log_suppressed(True)
        try:
            yield
        finally:
            BaseFetcher._set_retry_log_suppressed(previous_value)

    @staticmethod
    def _resolve_proxy_mode() -> str:
        proxy_mode = os.environ.get("OTC_FUND_QUANT_PROXY_MODE", PROXY_MODE).strip().lower() or PROXY_MODE
        if proxy_mode not in {"auto", "env", "direct"}:
            return PROXY_MODE
        return proxy_mode

    @staticmethod
    @contextmanager
    def _temporary_proxy_mode(disable_proxy: bool):
        if not disable_proxy:
            yield
            return

        tracked_keys = {
            key.lower()
            for key in BaseFetcher.PROXY_ENV_KEYS
        }
        previous_values = {
            key: value
            for key, value in os.environ.items()
            if key.lower() in tracked_keys
        }
        for key in list(previous_values.keys()):
            os.environ.pop(key, None)

        original_session_init = requests.sessions.Session.__init__

        def _patched_session_init(session_self, *args, **kwargs):
            original_session_init(session_self, *args, **kwargs)
            session_self.trust_env = False
            try:
                session_self.proxies.clear()
            except Exception:
                session_self.proxies = {}

        try:
            requests.sessions.Session.__init__ = _patched_session_init
            yield
        finally:
            requests.sessions.Session.__init__ = original_session_init
            for key, value in previous_values.items():
                os.environ[key] = value

    @staticmethod
    def build_live_payload(
        *,
        value: float,
        source: str,
        source_priority: int,
        raw: dict[str, Any],
        data_as_of_date: str | None = None,
        warnings: list[str] | None = None,
    ) -> dict[str, Any]:
        return {
            "value": float(value),
            "data_as_of_date": data_as_of_date,
            "source": source,
            "source_priority": int(source_priority),
            "raw": raw,
            "warnings": list(warnings or []),
            "source_disagreements": [],
        }

    @staticmethod
    def resolve_live_source(
        *,
        primary_fetcher: Callable[[], dict[str, Any]],
        backup_fetcher: Callable[[], dict[str, Any]],
        strict: bool,
        label: str,
        threshold: float,
        probe_backup_on_primary_success: bool = False,
    ) -> dict[str, Any]:
        warnings: list[str] = []
        disagreements: list[str] = []
        primary_result: dict[str, Any] | None = None
        backup_result: dict[str, Any] | None = None
        primary_error: Exception | None = None
        backup_error: Exception | None = None

        try:
            primary_result = primary_fetcher()
        except Exception as exc:
            primary_error = exc

        should_probe_backup = primary_result is None or probe_backup_on_primary_success
        if should_probe_backup:
            try:
                backup_result = backup_fetcher()
            except Exception as exc:
                backup_error = exc

        if primary_result is not None:
            result = dict(primary_result)
            if backup_result is not None:
                diff = abs(float(primary_result["value"]) - float(backup_result["value"]))
                if diff > threshold:
                    disagreement = (
                        f"{label} 主备源差异过大: {primary_result['source']}={float(primary_result['value']):+.4f}, "
                        f"{backup_result['source']}={float(backup_result['value']):+.4f}, diff={diff:.4f}"
                    )
                    disagreements.append(disagreement)
                    if strict:
                        raise ValueError(disagreement)
                    warnings.append(disagreement)
            result["warnings"] = list(result.get("warnings", [])) + warnings
            result["source_disagreements"] = list(result.get("source_disagreements", [])) + disagreements
            return result

        if backup_result is not None:
            result = dict(backup_result)
            fallback_warning = f"{label} 主源不可用，已切换至备源 {backup_result['source']}"
            logger.warning(fallback_warning)
            result["warnings"] = list(result.get("warnings", [])) + [fallback_warning]
            result["source_disagreements"] = list(result.get("source_disagreements", []))
            return result

        raise ValueError(
            f"{label} 主备实时源均不可用；主源异常: {primary_error}; 备源异常: {backup_error}"
        )

    @staticmethod
    def with_cache(
        ttl: int = 300,
        *,
        refresh_on_cached_empty_dataframe: bool = False,
        cache_empty_dataframe: bool = True,
    ):
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs) -> Any:
                cache_key = BaseFetcher._build_cache_key(func, args, kwargs)
                cached = BaseFetcher.cache.get(cache_key)
                if cached is not None:
                    if refresh_on_cached_empty_dataframe and BaseFetcher._is_empty_dataframe(cached):
                        BaseFetcher.cache.delete(cache_key)
                    else:
                        return cached
                try:
                    result = func(*args, **kwargs)
                    if result is not None:
                        if BaseFetcher._is_empty_dataframe(result) and not cache_empty_dataframe:
                            BaseFetcher.cache.delete(cache_key)
                            return result
                        BaseFetcher.cache.set(cache_key, result, ttl)
                    return result
                except Exception as e:
                    if not BaseFetcher._is_retry_log_suppressed():
                        logger.error(f"函数执行失败 {func.__name__}: {e}")
                    raise

            return wrapper

        return decorator

    @staticmethod
    def retry_on_error(max_retries: int = 3, delay: float = 1.0):
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs) -> Any:
                proxy_mode = BaseFetcher._resolve_proxy_mode()
                force_direct = proxy_mode == "direct" or (
                    proxy_mode == "auto" and BaseFetcher._is_direct_mode_latched()
                )
                strict_mode = BaseFetcher._get_strict_mode_latched()
                effective_max_retries = max_retries
                effective_delay = delay
                if strict_mode is False:
                    effective_max_retries = min(max_retries, 2)
                    effective_delay = min(delay, 0.2)
                proxy_failed = False
                for attempt in range(effective_max_retries):
                    try:
                        disable_proxy = force_direct or (proxy_mode == "auto" and proxy_failed)
                        if disable_proxy and not BaseFetcher._is_retry_log_suppressed():
                            logger.debug(f"函数 {func.__name__} 本次重试使用直连模式")
                        with BaseFetcher._temporary_proxy_mode(disable_proxy):
                            return func(*args, **kwargs)
                    except Exception as e:
                        if proxy_mode == "auto" and BaseFetcher._is_proxy_related_error(e):
                            if not proxy_failed and not BaseFetcher._is_retry_log_suppressed():
                                logger.warning(
                                    f"函数 {func.__name__} 检测到代理连接失败，后续重试将禁用代理直连: {e}"
                                )
                            proxy_failed = True
                            BaseFetcher._set_direct_mode_latched(True)
                        if BaseFetcher._is_non_retryable_error(e):
                            if not BaseFetcher._is_retry_log_suppressed():
                                logger.error(f"函数 {func.__name__} 遇到不可重试错误，直接失败: {e}")
                            raise
                        if attempt < effective_max_retries - 1:
                            if not BaseFetcher._is_retry_log_suppressed():
                                logger.warning(
                                    f"函数 {func.__name__} 执行失败 (尝试 {attempt + 1}/{effective_max_retries}): {e}"
                                )
                            time.sleep(effective_delay)
                        else:
                            if not BaseFetcher._is_retry_log_suppressed():
                                logger.error(f"函数 {func.__name__} 重试{effective_max_retries}次后仍失败: {e}")
                            raise

            return wrapper

        return decorator
