"""
将 legacy strategy_params.json 迁移为 strategy_profiles.json。
"""

from __future__ import annotations

import argparse
import json
import os
import sys


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARENT_OF_PROJECT = os.path.dirname(PROJECT_ROOT)
for path in [PROJECT_ROOT, PARENT_OF_PROJECT]:
    if path not in sys.path:
        sys.path.insert(0, path)

from otc_fund_quant.config.loader import build_profile_config_from_legacy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="将旧策略配置迁移为画像分层配置")
    parser.add_argument(
        "--input",
        default=os.path.join(PROJECT_ROOT, "config", "strategy_params.json"),
        help="旧配置文件路径",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(PROJECT_ROOT, "config", "strategy_profiles.json"),
        help="新配置文件输出路径",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="允许覆盖已存在的输出文件",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not os.path.exists(args.input):
        print(f"旧配置文件不存在: {args.input}")
        return 1

    if os.path.exists(args.output) and not args.force:
        print(f"输出文件已存在，请使用 --force 覆盖: {args.output}")
        return 1

    with open(args.input, "r", encoding="utf-8") as f:
        legacy_config = json.load(f)

    profile_config = build_profile_config_from_legacy(legacy_config)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(profile_config, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"已生成画像配置: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
