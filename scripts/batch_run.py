"""批量跑案例 CLI。

用法:
    # 单个案例
    python -m scripts.batch_run --case <CASE_ID> --dir <路径/到/案例目录>

    # 整个目录(自动把子目录当案例)
    python -m scripts.batch_run --batch-dir <路径/到/案例根目录> --limit 3

    # 与打标结果对比
    python -m scripts.batch_run --batch-dir <路径/到/案例根目录> --compare-xlsx <打标xlsx路径>
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.executor import judge_one_case, run_batch


def setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def cmd_single(args) -> int:
    """单案例命令。"""
    result = judge_one_case(
        case_id=args.case,
        case_dir=args.dir,
        output_path=args.out,
    )
    print(f"\n=== 案例 {args.case} ===")
    print(f"  结论: {result['judge']['conclusion']}")
    if result['judge'].get('confidence') is not None:
        print(f"  置信度: {result['judge']['confidence']:.3f}")
    if result['judge'].get('matched_rule_id'):
        print(f"  命中规则: {result['judge']['matched_rule_id']} {result['judge'].get('matched_rule_name', '')}")
    print(f"  风险码: {[w['code'] for w in result['judge'].get('risk_warnings', [])]}")
    print(f"  诊断: {result['extraction']['diagnosis']['diagnosis'] if result['extraction'] else '(抽取失败)'}")
    print(f"  ICD-10: {result['extraction']['diagnosis']['icd10'] if result['extraction'] else '(抽取失败)'}")
    print(f"  耗时: {result['elapsed_sec']}s")
    if result.get('error'):
        print(f"  ❌ {result['error']}")
    return 0 if not result.get('error') else 1


def cmd_batch(args) -> int:
    """批跑命令。"""
    root = Path(args.batch_dir)
    if not root.exists():
        print(f"❌ 目录不存在: {root}", file=sys.stderr)
        return 1

    # 自动把每个子目录当一个案例
    case_dirs = [d for d in sorted(root.iterdir()) if d.is_dir()]
    if args.limit:
        case_dirs = case_dirs[:args.limit]
    if not case_dirs:
        print(f"❌ {root} 下没有子目录", file=sys.stderr)
        return 1

    print(f"将批跑 {len(case_dirs)} 个案例,落盘到: {args.out_dir or 'data/output/'}")
    case_specs = [(d.name, d) for d in case_dirs]
    t0 = time.time()
    results = run_batch(case_specs, output_dir=args.out_dir)
    elapsed = time.time() - t0

    # 汇总
    ok = sum(1 for r in results if not r.get("error"))
    failed = len(results) - ok
    print(f"\n=== 批跑完成: 成功 {ok}, 失败 {failed}, 总耗时 {elapsed:.1f}s ===")
    for r in results:
        if r.get("error"):
            print(f"  ❌ {r['case_id']}: {r['error']}")
        else:
            # confidence 可能是 None(无法认定)或缺失
            conf = r['judge'].get('confidence')
            conf_str = f"{conf:.2f}" if isinstance(conf, (int, float)) else "N/A"
            print(
                f"  {r['case_id']}: {r['judge']['conclusion']:<8} "
                f"conf={conf_str} "
                f"({r['elapsed_sec']}s)"
            )

    # 与打标结果对比
    if args.compare_xlsx:
        try:
            from scripts.eval_official_label import compare_results
            compare_results(results, args.compare_xlsx, summary_only=not args.compare_full)
        except Exception as e:
            print(f"⚠️ 对比失败: {e}", file=sys.stderr)

    return 0 if failed == 0 else 2


def main() -> int:
    parser = argparse.ArgumentParser(description="Insurance AI 批跑脚本")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # single
    p1 = sub.add_parser("single", help="跑单个案例")
    p1.add_argument("--case", required=True, help="案例 ID")
    p1.add_argument("--dir", required=True, help="案例目录或单文件")
    p1.add_argument("--out", help="结果 JSON 输出路径")
    p1.add_argument("-v", "--verbose", action="store_true")
    p1.set_defaults(func=cmd_single)

    # batch
    p2 = sub.add_parser("batch", help="批跑一个目录下的所有子目录")
    p2.add_argument("--batch-dir", required=True, help="案例根目录")
    p2.add_argument("--limit", type=int, help="最多跑 N 个案例")
    p2.add_argument("--out-dir", help="结果输出目录")
    p2.add_argument("--compare-xlsx", help="打标 xlsx 路径,做对比")
    p2.add_argument("--compare-full", action="store_true", help="输出每条详细对比")
    p2.add_argument("-v", "--verbose", action="store_true")
    p2.set_defaults(func=cmd_batch)

    args = parser.parse_args()
    setup_logging(args.verbose)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
