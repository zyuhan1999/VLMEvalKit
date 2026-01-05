import argparse
import json
import os
import re
from typing import Dict, List, Optional, Tuple

import pandas as pd


def _safe_model_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", str(name))


def _iou(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    max0 = max(a[0], b[0])
    min0 = min(a[0], b[0])
    max1 = max(a[1], b[1])
    min1 = min(a[1], b[1])
    denom = (max1 - min0)
    if denom <= 0:
        return 0.0
    return max(min1 - max0, 0.0) / denom


def _extract_time_rule(pred_text: str) -> List[Tuple[float, float]]:
    # Reuse VLMEvalKit implementation to stay consistent with current timelens.py
    from vlmeval.dataset.timelens import _extract_time

    return [(float(s), float(e)) for (s, e) in _extract_time(pred_text)]


def _llm_extract_span(
    judge_model,
    pred_text: str,
    duration: float,
) -> Tuple[float, float]:
    """
    LLM parser: ask for {"start":..,"end":..}. If the judge returns a list like [[s,e],...],
    fall back to rule-based number-pair parsing and take the first span.
    """
    from vlmeval.smp import extract_json_objects

    dur_str = f"{duration:.3f}" if duration and duration > 0 else ""
    prompt = (
        "You are a parser. Extract the predicted temporal segment from the model output.\n"
        "Output MUST be a JSON object with keys: start (number), end (number).\n"
        "Units: seconds. If multiple spans exist, output the first/main one.\n"
        "If no valid span can be extracted, output: {\"start\": -100, \"end\": -100}\n"
        + (f"Video duration (seconds): {dur_str}\n" if dur_str else "")
        + f"Model output:\n{pred_text}\n"
    )
    resp = judge_model.generate(prompt)
    try:
        js = list(extract_json_objects(resp))
        if js:
            obj = js[-1]
            s = float(obj.get("start", -100.0))
            e = float(obj.get("end", -100.0))
            return (s, e)
    except Exception:
        pass

    # fallback: accept list-form outputs
    spans = _extract_time_rule(str(resp))
    if spans:
        return spans[0]
    return (-100.0, -100.0)


def compute_metrics_from_df(
    df: pd.DataFrame,
    *,
    use_llm: bool,
    judge_model=None,
    force_llm: bool = False,
    llm_parse_on_fail_only: bool = True,
) -> Dict[str, float]:
    if "prediction" not in df.columns:
        raise ValueError("input df must contain `prediction` column")
    # allow either gt_start/gt_end or gt_span
    if ("gt_start" not in df.columns) or ("gt_end" not in df.columns):
        if "gt_span" not in df.columns:
            raise ValueError("input df must contain `gt_start/gt_end` or `gt_span`")
        gt = []
        for x in df["gt_span"]:
            if isinstance(x, (list, tuple)) and len(x) == 2:
                gt.append((float(x[0]), float(x[1])))
            else:
                s, e = eval(str(x))
                gt.append((float(s), float(e)))
        df = df.copy()
        df["gt_start"] = [x[0] for x in gt]
        df["gt_end"] = [x[1] for x in gt]

    if "duration" not in df.columns:
        df = df.copy()
        df["duration"] = [0.0] * len(df)

    ious: List[float] = []
    valid = 0
    used_llm = 0

    for _, row in df.iterrows():
        pred = row["prediction"]
        pred_text = "" if (isinstance(pred, float) and pd.isna(pred)) else str(pred)
        g = (float(row["gt_start"]), float(row["gt_end"]))

        p = (-100.0, -100.0)

        if use_llm and judge_model is not None and (force_llm or (not llm_parse_on_fail_only)):
            p = _llm_extract_span(judge_model, pred_text, float(row.get("duration", 0.0) or 0.0))
            used_llm += 1
            if p[0] != -100.0 and p[1] != -100.0:
                valid += 1
            else:
                spans = _extract_time_rule(pred_text)
                if spans:
                    p = spans[0]
                    valid += 1
        else:
            spans = _extract_time_rule(pred_text)
            if spans:
                p = spans[0]
                valid += 1
            elif use_llm and judge_model is not None and llm_parse_on_fail_only:
                p = _llm_extract_span(judge_model, pred_text, float(row.get("duration", 0.0) or 0.0))
                used_llm += 1
                if p[0] != -100.0 and p[1] != -100.0:
                    valid += 1

        # IMPORTANT: no clamp (align with official TimeLens metrics)
        if p[0] >= p[1]:
            ious.append(0.0)
        else:
            ious.append(_iou(g, p))

    num = len(ious)
    recalls = {f"R1@{thr}": float(sum(x >= thr for x in ious)) / num if num else 0.0 for thr in (0.3, 0.5, 0.7)}
    res = dict(
        **recalls,
        mIoU=float(sum(ious) / num) if num else 0.0,
        num_samples=int(num),
        num_valid_pred=int(valid),
        num_llm_parsed=int(used_llm),
    )
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dir",
        type=str,
        required=True,
        help="An outputs eval_id directory, e.g. .../outputs/Qwen3-VL-2B-Instruct/T20251230_Gf127f860",
    )
    ap.add_argument("--judge", type=str, default=None, help="Judge model name, e.g. qwen3-235b-a22b-instruct-2507")
    ap.add_argument(
        "--force-llm",
        action="store_true",
        help="If set, always use LLM parser for every sample (otherwise only parse-on-fail by default).",
    )
    ap.add_argument(
        "--llm-parse-on-fail-only",
        action="store_true",
        default=True,
        help="Use LLM parser only when rule parsing fails (default: True).",
    )
    ap.add_argument(
        "--out-suffix",
        type=str,
        default="noclamp",
        help="Suffix tag to write corrected files, default: noclamp",
    )
    args = ap.parse_args()

    out_dir = os.path.abspath(args.dir)
    model_prefix = os.path.basename(os.path.dirname(out_dir))  # e.g. Qwen3-VL-2B-Instruct

    def p(name: str) -> str:
        return os.path.join(out_dir, name)

    # load per-subset inference results (already exist when TimeLens was evaluated)
    subsets = {
        "TimeLens_Charades": p(f"{model_prefix}_TimeLens_Charades_2fps.xlsx"),
        "TimeLens_ActivityNet": p(f"{model_prefix}_TimeLens_ActivityNet_2fps.xlsx"),
        "TimeLens_QVHighlights": p(f"{model_prefix}_TimeLens_QVHighlights_2fps.xlsx"),
    }

    for k, fp in subsets.items():
        if not os.path.exists(fp):
            raise FileNotFoundError(f"missing required inference file: {fp}")

    judge_model = None
    use_llm = bool(args.judge)
    if use_llm:
        from vlmeval.dataset.utils import build_judge

        judge_model = build_judge(model=args.judge)

    per = {}
    for name, fp in subsets.items():
        df = pd.read_excel(fp)
        per[name] = compute_metrics_from_df(
            df,
            use_llm=use_llm,
            judge_model=judge_model,
            force_llm=bool(args.force_llm),
            llm_parse_on_fail_only=bool(args.llm_parse_on_fail_only),
        )

        # write per-subset score
        tag = args.out_suffix
        if use_llm:
            out_name = f"{model_prefix}_{name.split('_',1)[1]}_2fps_{_safe_model_name(args.judge)}_{tag}_score.json"
        else:
            out_name = f"{model_prefix}_{name.split('_',1)[1]}_2fps_exact_matching_{tag}_score.json"
        with open(p(out_name), "w", encoding="utf-8") as f:
            json.dump(per[name], f, indent=4)

    # overall = concat all subsets
    all_df = pd.concat([pd.read_excel(fp) for fp in subsets.values()], ignore_index=True)
    overall = compute_metrics_from_df(
        all_df,
        use_llm=use_llm,
        judge_model=judge_model,
        force_llm=bool(args.force_llm),
        llm_parse_on_fail_only=bool(args.llm_parse_on_fail_only),
    )

    tag = args.out_suffix
    if use_llm:
        overall_name = f"{model_prefix}_TimeLens__overall_2fps_{_safe_model_name(args.judge)}_{tag}_score.json"
    else:
        overall_name = f"{model_prefix}_TimeLens__overall_2fps_exact_matching_{tag}_score.json"
    with open(p(overall_name), "w", encoding="utf-8") as f:
        json.dump(overall, f, indent=4)

    # summary file (same style as VLMEvalKit TimeLens_2fps_score.json)
    summary = {}
    for ds, r in per.items():
        for kk, vv in r.items():
            summary[f"{ds}:{kk}"] = vv
    for kk, vv in overall.items():
        summary[f"Overall:{kk}"] = vv

    if use_llm:
        sum_name = f"{model_prefix}_TimeLens_2fps_{_safe_model_name(args.judge)}_{tag}_score.json"
    else:
        sum_name = f"{model_prefix}_TimeLens_2fps_exact_matching_{tag}_score.json"
    with open(p(sum_name), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    print("[OK] wrote corrected no-clamp scores into:", out_dir)
    print(" -", sum_name)
    print(" -", overall_name)


if __name__ == "__main__":
    main()


