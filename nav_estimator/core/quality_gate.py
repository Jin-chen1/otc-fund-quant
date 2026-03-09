"""质量优先模式辅助工具。"""

from typing import Any


def quality_policy(strict: bool) -> str:
    return "quality_first" if strict else "best_effort"


def build_quality_metadata(
    strict: bool,
    *,
    passed: bool,
    reasons: list[str] | None = None,
    used_sources: dict[str, Any] | None = None,
    source_disagreements: list[str] | None = None,
    live_quote_coverage_pct: float | None = None,
    live_quote_weight_pct: float | None = None,
) -> dict[str, Any]:
    return {
        "quality_policy": quality_policy(strict),
        "quality_gate_passed": passed,
        "quality_gate_failed_reasons": list(reasons or []),
        "used_sources": used_sources or {},
        "source_disagreements": list(source_disagreements or []),
        "live_quote_coverage_pct": live_quote_coverage_pct,
        "live_quote_weight_pct": live_quote_weight_pct,
    }


class QualityGateError(ValueError):
    """质量门槛未通过。"""

    def __init__(
        self,
        reasons: list[str],
        *,
        strict: bool = True,
        used_sources: dict[str, Any] | None = None,
        source_disagreements: list[str] | None = None,
        live_quote_coverage_pct: float | None = None,
        live_quote_weight_pct: float | None = None,
    ):
        normalized_reasons = list(reasons or ["质量门槛未通过"])
        self.quality_gate_failed_reasons = normalized_reasons
        self.quality_details = build_quality_metadata(
            strict,
            passed=False,
            reasons=normalized_reasons,
            used_sources=used_sources,
            source_disagreements=source_disagreements,
            live_quote_coverage_pct=live_quote_coverage_pct,
            live_quote_weight_pct=live_quote_weight_pct,
        )
        super().__init__("质量门槛未通过: " + "；".join(normalized_reasons))
