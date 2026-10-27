"""
Insurance AI 判定系统 WebUI(Streamlit)。

启动方式:
    python -m streamlit run src/webui/app.py

布局:
  左栏: 输入(上传 / demo 按钮 / 提交)
  右栏: 4 步流水线状态 + 最终判定结果(结论/置信度/命中规则/风险码)
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Optional

# ===== sys.path 必须在任何业务 import 之前 =====
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st  # noqa: E402

from src.webui.demo_cases import DEMO_CASES, DemoCase, case_count, SAMPLES_BASE  # noqa: E402
from src.executor import judge_one_case  # noqa: E402
from src.utils.step_logger import StepLogger  # noqa: E402


# ===== 页面配置 =====
st.set_page_config(
    page_title="Insurance AI 重疾认定",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ===== 样式 =====
CUSTOM_CSS = """
<style>
    .stage-box {
        padding: 12px 16px;
        border-radius: 8px;
        background-color: #f0f2f6;
        margin-bottom: 8px;
        font-size: 14px;
    }
    .stage-done {
        background-color: #d4edda;
        color: #155724;
    }
    .stage-running {
        background-color: #fff3cd;
        color: #856404;
    }
    .stage-error {
        background-color: #f8d7da;
        color: #721c24;
    }
    .verdict-yes {
        font-size: 32px;
        color: #28a745;
        font-weight: bold;
    }
    .verdict-no {
        font-size: 32px;
        color: #dc3545;
        font-weight: bold;
    }
    .verdict-unknown {
        font-size: 32px;
        color: #ffc107;
        font-weight: bold;
    }
    .risk-flag {
        padding: 8px 12px;
        border-left: 3px solid #ffc107;
        background-color: #fff8e1;
        margin-bottom: 6px;
        border-radius: 4px;
        font-family: monospace;
        font-size: 13px;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ===== Pipeline 步骤配置(5 个 stage) =====
PIPELINE_STEPS = [
    ("1_ocr", "OCR 识别", "📷"),
    ("2_aggregate", "聚合", "🔗"),
    ("3_llm_extract", "LLM 抽取", "🧠"),
    ("4_judge", "判定引擎", "⚖️"),
    ("5_evaluate", "评析", "💬"),
]


def _render_stage_status(stage_name: str, step_state: dict) -> str:
    """根据 step_state 渲染 stage 状态(emoji + class)。"""
    name = next((s[1] for s in PIPELINE_STEPS if s[0] == stage_name), stage_name)
    icon = next((s[2] for s in PIPELINE_STEPS if s[0] == stage_name), "•")

    if step_state.get("error"):
        return (
            f'<div class="stage-box stage-error">❌ {icon} {name} — 失败:'
            f'{step_state["error"][:200]}</div>'
        )
    if step_state.get("done"):
        ms = step_state.get("duration_ms", 0)
        return f'<div class="stage-box stage-done">✅ {icon} {name} — {ms:.0f}ms</div>'
    return f'<div class="stage-box stage-running">⏳ {icon} {name} — 运行中...</div>'


def _render_verdict(conclusion: str) -> str:
    """渲染大字号判定结论标签。"""
    if conclusion == "是":
        return '<div class="verdict-yes">🟢 是(本次确诊属于重疾)</div>'
    if conclusion == "否":
        return '<div class="verdict-no">🔴 否(本次确诊不属于重疾)</div>'
    return '<div class="verdict-unknown">🟡 无法认定(关键信息缺失)</div>'


def _fmt(v) -> str:
    """格式化为短字符串。"""
    if v is None:
        return "None"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, str):
        return v if len(v) <= 200 else v[:200] + f"...(共 {len(v)} 字符)"
    if isinstance(v, (list, tuple)):
        return f"[{len(v)} items]" if v else "[]"
    if isinstance(v, dict):
        return f"{{{len(v)} keys}}"
    s = str(v)
    return s if len(s) <= 200 else s[:200] + "..."


def _format_step_io(step: dict) -> str:
    """把 step 的 inputs/outputs 格式化成可读文本。"""
    lines = []
    if step.get("inputs"):
        lines.append("📥 INPUT:")
        for k, v in step["inputs"].items():
            lines.append(f"  {k}: {_fmt(v)}")
    if step.get("outputs"):
        lines.append("📤 OUTPUT:")
        for k, v in step["outputs"].items():
            lines.append(f"  {k}: {_fmt(v)}")
    if step.get("error"):
        lines.append(f"❌ ERROR: {step['error']}")
    return "\n".join(lines) if lines else "(空)"


def _run_pipeline(case_id: str, image_paths: list[Path]) -> dict:
    """执行完整 pipeline,返回结果 + step_states + 完整 step log。"""
    step_states = {s[0]: {"done": False, "duration_ms": 0, "error": None}
                   for s in PIPELINE_STEPS}

    progress_bar = st.progress(0.0, text="准备开始...")
    step_placeholders = {sid: st.empty() for sid, _, _ in PIPELINE_STEPS}
    state_holder = {"step_states": step_states}

    def _refresh_progress() -> None:
        done_count = sum(1 for s in state_holder["step_states"].values() if s.get("done"))
        total = len(PIPELINE_STEPS)
        progress_bar.progress(done_count / total, text=f"已完成 {done_count}/{total} 步")
        for sid, _, _ in PIPELINE_STEPS:
            step_placeholders[sid].markdown(
                _render_stage_status(sid, state_holder["step_states"][sid]),
                unsafe_allow_html=True,
            )

    _refresh_progress()

    # 关键修复:把 step_state 注册到 step_logger 的 on_step_end 回调里
    # 让每完成一个 stage 就实时刷新 UI(而不是最后一次性刷新)
    def _on_step_done(step_record) -> None:
        state_holder["step_states"][step_record.name] = {
            "done": True,
            "duration_ms": step_record.duration_ms or 0,
            "error": step_record.error,
        }
        _refresh_progress()

    try:
        # StepLogger 内部会落盘 logs/case_<id>.log.json,供事后审计
        # on_step_end 回调让每完成一个 stage 就通知 streamlit 刷新 UI
        step_logger = StepLogger(
            case_id=case_id,
            verbose=False,
            on_step_end=_on_step_done,
        )
        result = _run_with_image_paths(
            case_id=case_id,
            image_paths=image_paths,
            step_logger=step_logger,
        )

        # 把 step_logger 内部的 step 状态同步到 step_states(给 UI 状态栏)
        for s in step_logger.steps:
            step_states[s.name] = {
                "done": True,
                "duration_ms": s.duration_ms or 0,
                "error": s.error,
            }
        _refresh_progress()
        progress_bar.progress(1.0, text="✅ 全部完成")

        return {
            "ok": True,
            "result": result,
            "step_states": step_states,
            "step_log": step_logger.to_dict(),
        }
    except Exception as e:
        progress_bar.progress(1.0, text=f"❌ 失败:{e}")
        return {"ok": False, "error": str(e), "step_states": state_holder["step_states"]}


def _run_with_image_paths(
    case_id: str,
    image_paths: list[Path],
    step_logger: StepLogger,
):
    """把 image_paths 复制到临时目录再调 judge_one_case。"""
    import shutil

    # 在 data/output/ 下面建一个临时 case 目录
    tmp_root = PROJECT_ROOT / "data" / "output" / "webui_uploads"
    tmp_root.mkdir(parents=True, exist_ok=True)
    case_dir = tmp_root / f"{case_id}_{int(time.time())}"
    case_dir.mkdir(parents=True, exist_ok=True)

    for src in image_paths:
        dst = case_dir / src.name
        shutil.copy2(src, dst)

    try:
        return judge_one_case(
            case_id=case_id,
            case_dir=case_dir,
            step_logger=step_logger,
            save_step_log=True,  # WebUI 跑也落盘
        )
    finally:
        # WebUI 跑完后清理临时图片目录(仅 webui_uploads 下,不动 resources)
        # 用 ignore_errors=True 防止权限问题;exist_ok=True 防止目录不存在
        try:
            shutil.rmtree(case_dir, ignore_errors=True)
        except Exception:
            pass


def _render_result(result_payload: dict) -> None:
    """右侧:渲染最终结果 + 每个 stage 的输入/输出详情。"""
    result = result_payload["result"]
    step_log = result_payload.get("step_log", {})
    steps = step_log.get("steps", [])

    judgment = result["judge"]
    extraction = result.get("extraction") or {}
    diag = extraction.get("diagnosis", {}) or {}

    # ===== 判定结果 =====
    st.markdown("### 📋 判定结果")
    st.markdown(_render_verdict(judgment.get("conclusion", "无法认定")),
                unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        conf = judgment.get("confidence")
        st.metric("置信度", f"{conf:.3f}" if conf is not None else "N/A")
    with col2:
        rule_id = judgment.get("matched_rule_id")
        st.metric("命中规则", f"{rule_id}" if rule_id else "—")
    with col3:
        st.metric("总耗时", f"{result.get('elapsed_sec', 0):.1f}s")

    # ===== 诊断信息 =====
    st.markdown("### 🩺 诊断信息")
    st.write(f"**疾病名**:`{diag.get('diagnosis') or '(未识别)'}`")
    if diag.get("icd10"):
        st.write(f"**ICD-10**:`{diag.get('icd10')}`")
    if diag.get("histology_type"):
        st.write(f"**组织学类型**:`{diag.get('histology_type')}`")
    if diag.get("pathology_diagnosis"):
        st.write(f"**病理诊断**:`{diag.get('pathology_diagnosis')}`")

    st.write(f"**材料类型**:`{', '.join(extraction.get('doc_types', [])) or '(未识别)'}`")
    st.write(f"**关键指标数**:`{len(extraction.get('lab_indicators', []))}`")
    st.write(f"**影像发现数**:`{len(extraction.get('imaging_findings', []))}`")

    # ===== 风险码 =====
    risks = judgment.get("risk_warnings", []) or []
    if risks:
        st.markdown("### ⚠️ 风险码")
        for w in risks:
            st.markdown(
                f'<div class="risk-flag">[{w.get("code", "?")}] {w.get("message", "")}</div>',
                unsafe_allow_html=True,
            )

    # ===== 投票详情 =====
    vote_confs = judgment.get("vote_confidences") or []
    if vote_confs:
        st.markdown("### 🗳️ 投票详情")
        st.write(f"**轮数**:{judgment.get('vote_rounds', len(vote_confs))} 轮")
        st.write(f"**置信度序列**:`{vote_confs}`")
        st.write(f"**是否一致**:`{'是' if judgment.get('vote_agree') else '否'}`")
        st.write(f"**硬规则命中**:`{judgment.get('hit_hard_rules', [])}`")

    # ===== 评析(LLM 综合解读) =====
    evaluation = result.get("evaluation") or {}
    if evaluation:
        st.markdown("### 💬 评析")
        summary = evaluation.get("summary", "").strip()
        callouts = evaluation.get("risk_callouts") or []
        if summary:
            st.info(summary)
        else:
            st.warning("评析文本为空,可能 LLM 调用失败。")
        if callouts:
            st.markdown("**关键要点:**")
            for c in callouts:
                st.markdown(f"- {c}")
        used = evaluation.get("risk_codes_used") or []
        if used:
            st.caption(f"参考风险码: {', '.join(used)}")
        if evaluation.get("error"):
            st.caption(f"⚠️ 评析生成异常: {evaluation['error']}")

    # ===== 4 个 stage 详细输入输出 =====
    st.markdown("### 🔍 流水线步骤详情")
    for step in steps:
        title = f"{step['name']} — {step.get('duration_ms', 0):.0f}ms"
        title += " ❌" if step.get("error") else " ✅"
        with st.expander(title, expanded=False):
            st.code(_format_step_io(step), language="text")


def main() -> None:
    st.title("🏥 Insurance AI 重疾认定系统")
    st.caption("OCR + LLM 抽取 + 100 条重疾规则 + LLM 投票 3 次 · 端到端可视化")

    # 两栏布局
    left_col, right_col = st.columns([1, 2])

    with left_col:
        st.markdown("### 📥 输入")

        case_id_input = st.text_input(
            "案例 ID",
            value="",
            placeholder="留空则使用 demo 或文件名",
            help="自定义案例 ID,留空将使用 demo 的 case_id",
        )

        st.markdown("#### 方式 1:上传文件")
        uploaded_files = st.file_uploader(
            "选择图片(可多选)",
            type=["jpg", "jpeg", "png", "bmp", "tiff", "tif", "webp"],
            accept_multiple_files=True,
            help="上传后点击下方提交",
        )

        st.markdown(f"#### 方式 2:预置 Demo ({case_count()} 个)")
        # 用 selectbox 展示所有 case,避免按钮铺开
        demo_options = [d for d in DEMO_CASES if d.image_paths]
        if demo_options:
            # 默认选第 1 个
            default_idx = 0
            if "selected_demo_idx" in st.session_state:
                default_idx = st.session_state["selected_demo_idx"]
            # selectbox 显示 label
            selected_label = st.selectbox(
                "选择 demo 案例",
                options=[d.label for d in demo_options],
                index=min(default_idx, len(demo_options) - 1),
                help="按 case_id + 疾病名展示,xlsx 真实标签在描述里",
            )
            # 找到选中的 demo
            selected_demo = next(
                (d for d in demo_options if d.label == selected_label),
                demo_options[0],
            )
            # 记录选中 index
            st.session_state["selected_demo_idx"] = demo_options.index(selected_demo)
            st.session_state["pending_demo"] = selected_demo
            st.caption(f"📝 {selected_demo.description}")
        else:
            st.warning(f"未找到任何 case 目录(扫描路径:{SAMPLES_BASE})")

        st.markdown("---")

        run_clicked = st.button(
            "▶ 提交并运行",
            type="primary",
            use_container_width=True,
            disabled=not (uploaded_files or st.session_state.get("pending_demo")),
        )

    with right_col:
        step_container = st.container()
        result_container = st.container()

        pending = st.session_state.get("pending_demo")
        if pending:
            with step_container:
                st.info(
                    f"已选择 Demo:**{pending.label}** ({len(pending.image_paths)} 张图)。"
                    f"点击 ▶ 提交运行"
                )

        if run_clicked:
            # 决定 case_id 和 image_paths
            # 关键修复:uploaded_files 优先于 pending_demo —— 只要用户上传了文件,
            # 就用上传的;只有未上传时才回退到 demo。
            case_id: Optional[str] = None
            image_paths: list[Path] = []

            if case_id_input.strip():
                case_id = case_id_input.strip()
            elif uploaded_files:
                case_id = f"upload_{int(time.time())}"
            elif pending:
                case_id = pending.case_id
            else:
                case_id = f"upload_{int(time.time())}"

            if uploaded_files:
                # 上传文件 → 保存到临时目录
                upload_dir = PROJECT_ROOT / "data" / "output" / "webui_uploads" / case_id
                upload_dir.mkdir(parents=True, exist_ok=True)
                for f in uploaded_files:
                    save_path = upload_dir / f.name
                    save_path.write_bytes(f.read())
                    image_paths.append(save_path)
            elif pending and pending.image_paths:
                image_paths = list(pending.image_paths)
            else:
                pass

            if not image_paths:
                with result_container:
                    st.error("❌ 没有可用的图片文件")
                return

            with step_container:
                st.markdown("### ⏳ 执行进度")

            res = _run_pipeline(case_id, image_paths)

            st.session_state["last_result"] = res
            st.session_state["last_case_id"] = case_id
            st.session_state.pop("pending_demo", None)

        # 渲染历史结果
        last = st.session_state.get("last_result")
        if last:
            with step_container:
                for sid, _, _ in PIPELINE_STEPS:
                    state = last["step_states"].get(sid, {})
                    st.markdown(_render_stage_status(sid, state),
                                unsafe_allow_html=True)

            with result_container:
                if last.get("ok"):
                    _render_result(last)
                else:
                    st.error(f"Pipeline 失败:{last.get('error')}")

            # 落盘位置提示
            case_id = st.session_state.get("last_case_id")
            if case_id:
                log_path = PROJECT_ROOT / "logs" / f"case_{case_id}.log.json"
                if log_path.exists():
                    with result_container:
                        st.caption(
                            f"📁 步骤日志:`{log_path.relative_to(PROJECT_ROOT)}`"
                        )
        else:
            with result_container:
                st.info("👈 请在左侧上传文件或选择 Demo,点击 ▶ 提交")


if __name__ == "__main__":
    main()
