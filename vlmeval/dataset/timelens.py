import json
import os
import re
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from ..smp import *
from ..smp.file import get_file_extension, get_intermediate_file_path
from .video_base import VideoBaseDataset
from .video_concat_dataset import ConcatVideoDataset
from .utils.video_pyav import ffprobe_video_info, get_video_decode_backend, save_frames_by_indices_pyav


FAIL_MSG = 'Failed to obtain answer via API.'


@dataclass(frozen=True)
class _TimeLensPaths:
    bench_root: str
    ann_json: str
    videos_root: str
    tsv_path: str


def _default_timelens_bench_root() -> str:
    # User-provided default path for this workspace; can be overridden via env var.
    return os.environ.get('TIMELENS_BENCH_ROOT', '/root/s3/videogpu/zhuyuhan/benchmarks/TimeLens/TimeLens-Bench')


def _read_json(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _parse_query(query: str) -> str:
    # Match TimeLens official cleaning: normalize whitespace & strip trailing period.
    return re.sub(r"\s+", " ", str(query)).strip().strip(".").strip()


def _iou(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    max0 = max(a[0], b[0])
    min0 = min(a[0], b[0])
    max1 = max(a[1], b[1])
    min1 = min(a[1], b[1])
    denom = (max1 - min0)
    if denom <= 0:
        return 0.0
    return max(min1 - max0, 0.0) / denom


def _extract_time(paragraph: str) -> List[Tuple[float, float]]:
    """
    Ported (lightly) from TimeLens official `timelens/utils.py`.
    Extract timestamp pairs from model output.
    """
    paragraph = str(paragraph).lower()
    timestamps: List[Tuple[float, float]] = []

    # 1) HH:MM:SS(.xx) or MM:SS(.xx)
    time_regex = re.compile(r"\b(\d{1,2}:\d{2}:\d{2}(?:\.\d+)?|\d{1,2}:\d{2}(?:\.\d+)?)\b")
    time_matches = re.findall(time_regex, paragraph)
    time_matches = time_matches[: len(time_matches) // 2 * 2]
    if time_matches:
        converted: List[float] = []
        for t in time_matches:
            parts = t.split(":")
            if len(parts) == 3:
                h, m = map(int, parts[:2])
                s = float(parts[2])
                converted.append(float(h * 3600 + m * 60 + s))
            else:
                m = int(parts[0])
                s = float(parts[1])
                converted.append(float(m * 60 + s))
        timestamps = [(converted[i], converted[i + 1]) for i in range(0, len(converted), 2)]

    # 2) "m - n" or "m to n"
    if len(timestamps) == 0:
        patterns = [
            r"(\d+\.?\d*)\s*-\s*(\d+\.?\d*)",
            r"(\d+\.?\d*)\s+to\s+(\d+\.?\d*)",
        ]
        for pat in patterns:
            ms = re.findall(pat, paragraph)
            if ms:
                timestamps = [(float(s), float(e)) for s, e in ms]
                break

    # 3) fallback: any numbers -> pair them
    if len(timestamps) == 0:
        num_re = re.compile(r"\b(\d+\.\d+|\d+)\b")
        nums = re.findall(num_re, paragraph)
        nums = nums[: len(nums) // 2 * 2]
        timestamps = [(float(nums[i]), float(nums[i + 1])) for i in range(0, len(nums), 2)]

    return [(float(s), float(e)) for s, e in timestamps]


def _ensure_timelens_tsv(paths: _TimeLensPaths) -> None:
    if os.path.exists(paths.tsv_path):
        return
    if not os.path.exists(paths.ann_json):
        raise FileNotFoundError(f'[TimeLens] annotation json not found: {paths.ann_json}')
    if not os.path.isdir(paths.videos_root):
        raise FileNotFoundError(f'[TimeLens] videos root not found: {paths.videos_root}')

    raw = _read_json(paths.ann_json)
    rows = []
    idx = 0
    for vid, anno in raw.items():
        duration = float(anno['duration'])
        spans = anno['spans']
        queries = anno['queries']
        if len(spans) != len(queries):
            raise ValueError(f'[TimeLens] spans/queries length mismatch for {vid}: {len(spans)} vs {len(queries)}')
        # video file is named exactly as `vid`.mp4 under videos_root
        video_path = os.path.join(paths.videos_root, f'{vid}.mp4')
        if not os.path.exists(video_path):
            raise FileNotFoundError(f'[TimeLens] missing video file: {video_path}')
        for span, query in zip(spans, queries):
            s, e = float(span[0]), float(span[1])
            rows.append(
                dict(
                    index=idx,
                    video=str(vid),
                    question=_parse_query(query),
                    duration=duration,
                    gt_span=str([s, e]),
                    gt_start=s,
                    gt_end=e,
                )
            )
            idx += 1

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(paths.tsv_path), exist_ok=True)
    dump(df, paths.tsv_path)


class _TimeLensGroundingBase(VideoBaseDataset):
    TYPE = 'Video-VTG'
    MODALITY = 'VIDEO'

    # Prompt aligned with TimeLens official evaluation.
    GROUNDER_PROMPT = (
        "Please find the visual event described by the sentence '{query}', determining its starting and ending times. "
        "The format should be: 'The event happens in <start time> - <end time> seconds'."
    )

    GROUNDER_PROMPT_TEXT_TIMESTAMP = (
        "You are given a video with multiple frames. "
        "The numbers before each video frame indicate its sampling timestamp (in seconds). "
    ) + GROUNDER_PROMPT

    def __init__(self, dataset: str, subset: str, nframe=0, fps=-1, frames_limit=2048):
        self._subset = subset
        self.frames_limit = frames_limit  # following Qwen3-VL: cap frames per video for FPS sampling
        super().__init__(dataset=dataset, nframe=nframe, fps=fps)

    def prepare_dataset(self, dataset: str):
        bench_root = _default_timelens_bench_root()
        ann = os.path.join(bench_root, f'{self._subset}-timelens.json')
        videos_root = os.path.join(bench_root, 'videos', self._subset)
        tsv = os.path.join(bench_root, f'{self._subset}-timelens.tsv')
        paths = _TimeLensPaths(bench_root=bench_root, ann_json=ann, videos_root=videos_root, tsv_path=tsv)
        _ensure_timelens_tsv(paths)
        return dict(root=paths.videos_root, data_file=paths.tsv_path)

    def _sample_indices_and_timestamps(self, vid_path: str):
        backend = get_video_decode_backend()
        use_pyav = backend == "pyav"

        vr = None
        try:
            if backend != "pyav":
                import decord
                # Use single-threaded decoder for stability on some long/corrupted videos.
                vr = decord.VideoReader(vid_path, num_threads=1)
                n_frames = int(len(vr))
                video_fps = float(vr.get_avg_fps())
                duration = float(n_frames / video_fps) if video_fps > 0 else 0.0
            else:
                raise RuntimeError("force pyav")
        except Exception:
            n_frames, video_fps, duration = ffprobe_video_info(vid_path)
            use_pyav = True

        video_info = {
            'fps': float(video_fps),
            'n_frames': int(n_frames),
            'duration': float(duration),
            'backend': 'pyav' if use_pyav else 'decord',
        }
        if self.nframe > 0 and self.fps < 0:
            step_size = n_frames / (self.nframe + 1)
            indices = [int(i * step_size) for i in range(1, self.nframe + 1)]
            indices = [min(max(0, x), n_frames - 1) for x in indices]
            timestamps = [idx / video_fps for idx in indices]
            frame_paths = self.frame_paths(os.path.splitext(os.path.basename(vid_path))[0])
        elif self.fps > 0:
            total_duration = duration if duration > 0 else (n_frames / video_fps if video_fps > 0 else 0.0)
            required_frames = int(total_duration * self.fps)
            if required_frames > self.frames_limit:
                step_size = n_frames / (self.frames_limit + 1)
                indices = [int(i * step_size) for i in range(1, self.frames_limit + 1)]
                if backend != "decord":
                    video_info['backend'] = 'pyav'
            else:
                step_size = video_fps / self.fps
                indices = [int(i * step_size) for i in range(required_frames)]
            indices = [min(max(0, x), n_frames - 1) for x in indices]
            timestamps = [idx / video_fps for idx in indices]
            frame_paths = self.frame_paths_fps(os.path.splitext(os.path.basename(vid_path))[0], len(indices))
        else:
            raise ValueError('fps and nframe should be set at least one valid value')
        return vr, indices, timestamps, frame_paths, video_info

    def _save_frames_with_timestamps(self, line) -> Tuple[List[str], List[float]]:
        video = str(line['video'])
        vid_path = os.path.join(self.data_root, f'{video}.mp4')
        vr, indices, timestamps, frame_paths, video_info = self._sample_indices_and_timestamps(vid_path)

        flag = np.all([osp.exists(p) for p in frame_paths])
        if not flag:
            # Avoid creating lock next to the source video (may be on read-only/limited FUSE mount like /root/s3).
            lock_dir = osp.dirname(frame_paths[0]) if len(frame_paths) else self.frame_root
            os.makedirs(lock_dir, exist_ok=True)
            lock_path = osp.join(lock_dir, '.extract.lock')
            # Allow long wait: multiple ranks may trigger the same video's frame extraction.
            with portalocker.Lock(lock_path, 'w', timeout=3600):
                if not np.all([osp.exists(p) for p in frame_paths]):
                    if video_info.get('backend') == 'pyav' or vr is None:
                        save_frames_by_indices_pyav(
                            vid_path=vid_path,
                            indices=[int(x) for x in indices],
                            frame_paths=frame_paths,
                            total_frames=video_info.get('n_frames', None),
                            desc=f"Extracting (pyav): {video}",
                        )
                    else:
                        try:
                            # Stream decode & save to avoid holding many frames in RAM (prevents OOM on long videos).
                            for frame_idx, pth in zip(indices, frame_paths):
                                if osp.exists(pth):
                                    continue
                                arr = vr[int(frame_idx)].asnumpy()
                                Image.fromarray(arr).save(pth)
                        except Exception:
                            save_frames_by_indices_pyav(
                                vid_path=vid_path,
                                indices=[int(x) for x in indices],
                                frame_paths=frame_paths,
                                total_frames=video_info.get('n_frames', None),
                                desc=f"Extracting (pyav): {video}",
                            )
        return frame_paths, timestamps

    def build_prompt(self, line, video_llm=False):
        if isinstance(line, int):
            assert line < len(self)
            line = self.data.iloc[line]

        query = str(line['question'])
        # For Qwen3-VL: when video_llm=True we still pass extracted frames for controllable sampling & caps.
        if video_llm:
            assert self.fps > 0
            video = str(line['video'])
            vid_path = os.path.join(self.data_root, f'{video}.mp4')
            vr, indices, _, frame_paths, video_info = self._sample_indices_and_timestamps(vid_path)
            duration = float(video_info.get('duration', 0.0))
            actual_fps = (self.frames_limit / duration) if (len(frame_paths) == self.frames_limit and duration > 0) else self.fps
            prompt = self.GROUNDER_PROMPT.format(query=query)
            return [
                dict(type='text', value=prompt),
                dict(
                    type='video',
                    value=frame_paths,
                    sample_fps=actual_fps,
                    min_pixels=1 * 2 * 2 * 16 * 16,
                    max_pixels=640 * 32 * 32,
                    total_pixels=224000 * 4 * 16 * 16,
                ),
            ]

        # Multi-image mode: interleave timestamps + frames.
        frames, ts = self._save_frames_with_timestamps(line)
        msg = [dict(type='text', value=self.GROUNDER_PROMPT_TEXT_TIMESTAMP.format(query=query))]
        for t, p in zip(ts, frames):
            msg.append(dict(type='text', value=f'{t:.2f}s'))
            msg.append(dict(type='image', value=p))
        msg.append(dict(type='text', value="Answer with: The event happens in <start time> - <end time> seconds"))
        return msg

    @classmethod
    def evaluate(cls, eval_file, **judge_kwargs):
        # Temporal grounding evaluation is a strict numeric metric (IoU/Recall). We do NOT use LLM to judge
        # correctness directly; however, an optional LLM can be used to normalize/parse the model output into
        # a (start, end) time span (seconds), then we compute metrics as usual.
        model_name = judge_kwargs.get('model', 'exact_matching')
        # Default behavior when judge model is specified: ALWAYS use it as a parser (like Minerva/LVBench style),
        # with rule-based parsing as fallback if LLM fails.
        force_llm = bool(judge_kwargs.get('force_llm_judge', True))
        parse_on_fail_only = bool(judge_kwargs.get('llm_parse_on_fail_only', False))
        judge_model = None
        if model_name not in [None, 'exact_matching']:
            if gpt_key_set():
                try:
                    from .utils import build_judge, DEBUG_MESSAGE
                    judge_model = build_judge(**judge_kwargs)
                    if hasattr(judge_model, 'working') and (not judge_model.working()):
                        warnings.warn('[TimeLens] Judge model is not working properly, will use rule-based parsing only.')
                        warnings.warn(DEBUG_MESSAGE)
                        judge_model = None
                except Exception as e:
                    warnings.warn(
                        f'[TimeLens] Failed to build judge model ({model_name}), will use rule-based parsing only: '
                        f'{type(e)}: {e}'
                    )
                    try:
                        from .utils import DEBUG_MESSAGE
                        warnings.warn(DEBUG_MESSAGE)
                    except Exception:
                        pass
                    judge_model = None
            else:
                warnings.warn('[TimeLens] API key is not set, will use rule-based parsing only.')
                judge_model = None

        # If judge is unavailable, disable forcing.
        if judge_model is None:
            force_llm = False
            parse_on_fail_only = True
        assert get_file_extension(eval_file) in ['xlsx', 'json', 'tsv'], (
            'data file should be an supported format (xlsx/json/tsv) file'
        )
        # Write score file with model name suffix when judge is used (so outputs are distinguishable).
        safe_model_name = re.sub(r'[^a-zA-Z0-9._-]+', '_', str(model_name))
        suffix = f'_{safe_model_name}_score' if model_name not in [None, 'exact_matching'] else '_score'
        score_file = get_intermediate_file_path(eval_file, suffix, 'json')
        judge_cache_file = get_intermediate_file_path(eval_file, f'_{safe_model_name}_judge_cache', 'pkl')
        if not os.path.exists(score_file):
            data = load(eval_file)
            if 'prediction' not in data.columns:
                raise ValueError('[TimeLens] eval file must contain `prediction` column.')
            if 'gt_start' not in data.columns or 'gt_end' not in data.columns:
                # fallback to parse gt_span
                if 'gt_span' not in data.columns:
                    raise ValueError('[TimeLens] eval file must contain `gt_start/gt_end` or `gt_span`.')
                gt = []
                for x in data['gt_span']:
                    if isinstance(x, (list, tuple)) and len(x) == 2:
                        gt.append((float(x[0]), float(x[1])))
                    else:
                        s, e = eval(str(x))
                        gt.append((float(s), float(e)))
                data['gt_start'] = [x[0] for x in gt]
                data['gt_end'] = [x[1] for x in gt]

            if 'duration' not in data.columns:
                data['duration'] = [0.0] * len(data)

            ious = []
            valid = 0
            used_llm = 0  # count REAL LLM calls (cache miss only)

            # Load / init LLM-parse cache
            judge_cache = {}
            if judge_model is not None and os.path.exists(judge_cache_file):
                try:
                    judge_cache = load(judge_cache_file)
                except Exception:
                    judge_cache = {}
            if not isinstance(judge_cache, dict):
                judge_cache = {}
            new_cache_entries = 0
            flush_every = 50  # periodically flush cache for robustness

            def _llm_extract_span(pred_text: str, duration: float, cache_key=None):
                """
                Use LLM to normalize/parse model output into a single (start, end) span in seconds.
                Returns (-100, -100) if failed.
                Returns: (span, used_llm: bool, cache_updated: bool)
                """
                if judge_model is None:
                    return (-100.0, -100.0), False, False

                if cache_key is not None and cache_key in judge_cache:
                    try:
                        s, e = judge_cache[cache_key]
                        return (float(s), float(e)), False, False
                    except Exception:
                        # fallthrough to re-parse if cache entry is malformed
                        pass

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
                    if not js:
                        span = (-100.0, -100.0)
                        if cache_key is not None:
                            judge_cache[cache_key] = list(span)
                        return span, True, True
                    obj = js[-1]
                    s = float(obj.get('start', -100.0))
                    e = float(obj.get('end', -100.0))
                    span = (s, e)
                    if cache_key is not None:
                        judge_cache[cache_key] = [float(s), float(e)]
                    return span, True, True
                except Exception:
                    span = (-100.0, -100.0)
                    if cache_key is not None:
                        judge_cache[cache_key] = list(span)
                    return span, True, True

            for i, row in tqdm(data.iterrows(), total=len(data), desc="Evaluating TimeLens"):
                # prefer stable int index if available
                try:
                    cache_key = int(row.get('index', i))
                except Exception:
                    cache_key = int(i)

                pred = row['prediction']
                if pd.isna(pred):
                    pred = ''
                pred_text = str(pred)
                p = (-100.0, -100.0)
                # LLM-first (forced) or rule-first (fallback-on-fail)
                if judge_model is not None and (force_llm or (not parse_on_fail_only)):
                    dur = float(row.get('duration', 0.0) or 0.0)
                    p, did_call, did_update = _llm_extract_span(pred_text, dur, cache_key=cache_key)
                    if did_call:
                        used_llm += 1
                    if did_update:
                        new_cache_entries += 1
                    if p[0] != -100.0 and p[1] != -100.0:
                        valid += 1
                    else:
                        # fallback to rule-based parsing
                        ts = _extract_time(pred_text)
                        if ts:
                            p = ts[0]
                            valid += 1
                else:
                    # rule-first; optionally fallback to LLM if parsing fails
                    ts = _extract_time(pred_text)
                    if ts:
                        p = ts[0]
                        valid += 1
                    elif judge_model is not None and parse_on_fail_only:
                        dur = float(row.get('duration', 0.0) or 0.0)
                        p, did_call, did_update = _llm_extract_span(pred_text, dur, cache_key=cache_key)
                        if did_call:
                            used_llm += 1
                        if did_update:
                            new_cache_entries += 1
                        if p[0] != -100.0 and p[1] != -100.0:
                            valid += 1

                g = (float(row['gt_start']), float(row['gt_end']))
                if p[0] >= p[1]:
                    ious.append(0.0)
                else:
                    ious.append(_iou(g, p))

                # Periodic cache flush for robustness
                if judge_model is not None and new_cache_entries > 0 and (new_cache_entries % flush_every == 0):
                    dump(judge_cache, judge_cache_file)

            # Final cache flush
            if judge_model is not None and new_cache_entries > 0:
                dump(judge_cache, judge_cache_file)

            num = len(ious)
            recalls = {}
            for thr in [0.3, 0.5, 0.7]:
                recalls[f'R1@{thr}'] = float(np.mean([x >= thr for x in ious])) if num else 0.0
            res = dict(
                **recalls,
                mIoU=float(np.mean(ious)) if num else 0.0,
                num_samples=int(num),
                num_valid_pred=int(valid),
                num_llm_parsed=int(used_llm),
            )
            dump(res, score_file)
        return load(score_file)


class TimeLens_ActivityNet(_TimeLensGroundingBase):
    def __init__(self, dataset='TimeLens_ActivityNet', nframe=0, fps=-1, frames_limit=2048):
        super().__init__(dataset=dataset, subset='activitynet', nframe=nframe, fps=fps, frames_limit=frames_limit)

    @classmethod
    def supported_datasets(cls):
        return ['TimeLens_ActivityNet']


class TimeLens_Charades(_TimeLensGroundingBase):
    def __init__(self, dataset='TimeLens_Charades', nframe=0, fps=-1, frames_limit=2048):
        super().__init__(dataset=dataset, subset='charades', nframe=nframe, fps=fps, frames_limit=frames_limit)

    @classmethod
    def supported_datasets(cls):
        return ['TimeLens_Charades']


class TimeLens_QVHighlights(_TimeLensGroundingBase):
    def __init__(self, dataset='TimeLens_QVHighlights', nframe=0, fps=-1, frames_limit=2048):
        super().__init__(dataset=dataset, subset='qvhighlights', nframe=nframe, fps=fps, frames_limit=frames_limit)

    @classmethod
    def supported_datasets(cls):
        return ['TimeLens_QVHighlights']


class TimeLens(ConcatVideoDataset):
    def __init__(self, dataset='TimeLens', nframe=0, fps=-1):
        self.DATASET_SETS[dataset] = ['TimeLens_Charades', 'TimeLens_ActivityNet', 'TimeLens_QVHighlights']
        super().__init__(dataset=dataset, nframe=nframe, fps=fps)

    @classmethod
    def supported_datasets(cls):
        return ['TimeLens']

    def evaluate(self, eval_file, **judge_kwargs):
        # Split-by-subdataset, compute metrics for each, and also compute overall on concatenated data.
        data_all = load(eval_file)
        results = {}
        per_ds = {}
        for dname in self.datasets:
            tgt = eval_file.replace(self.dataset_name, dname)
            sub = data_all[data_all['SUB_DATASET'] == dname].copy()
            sub.pop('index')
            sub['index'] = sub.pop('original_index')
            sub.pop('SUB_DATASET')
            dump(sub, tgt)
            per_ds[dname] = self.dataset_map[dname].evaluate(tgt, **judge_kwargs)
            results.update({f'{dname}:{k}': v for k, v in per_ds[dname].items()})

        # overall metrics
        # IMPORTANT: do NOT re-run evaluation on an overall concatenated file.
        # We can compute the overall metrics exactly by aggregating per-subdataset results:
        #   - R1@thr and mIoU are averages over samples, so use sample-count weighted average.
        #   - count fields are summed.
        n_total = sum(int(per_ds[d].get('num_samples', 0) or 0) for d in self.datasets)
        if n_total <= 0:
            overall_res = {
                'R1@0.3': 0.0,
                'R1@0.5': 0.0,
                'R1@0.7': 0.0,
                'mIoU': 0.0,
                'num_samples': 0,
                'num_valid_pred': 0,
                'num_llm_parsed': 0,
            }
        else:
            def _wavg(key: str) -> float:
                num = 0.0
                for d in self.datasets:
                    sub_n = float(per_ds[d].get('num_samples', 0) or 0)
                    sub_v = float(per_ds[d].get(key, 0.0) or 0.0)
                    num += sub_n * sub_v
                return float(num / float(n_total))

            overall_res = dict(
                **{k: _wavg(k) for k in ['R1@0.3', 'R1@0.5', 'R1@0.7', 'mIoU']},
                num_samples=int(n_total),
                num_valid_pred=int(sum(int(per_ds[d].get('num_valid_pred', 0) or 0) for d in self.datasets)),
                num_llm_parsed=int(sum(int(per_ds[d].get('num_llm_parsed', 0) or 0) for d in self.datasets)),
            )
        results.update({f'Overall:{k}': v for k, v in overall_res.items()})

        model_name = judge_kwargs.get('model', 'exact_matching')
        safe_model_name = re.sub(r'[^a-zA-Z0-9._-]+', '_', str(model_name))
        suffix = f'_{safe_model_name}_score' if model_name not in [None, 'exact_matching'] else '_score'
        score_file = get_intermediate_file_path(eval_file, suffix, 'json')
        dump(results, score_file)
        return results


