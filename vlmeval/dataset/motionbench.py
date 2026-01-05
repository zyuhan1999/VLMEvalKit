import json
import os
import re
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..smp import *
from ..smp.file import get_intermediate_file_path, get_file_extension
from .video_base import VideoBaseDataset
from .utils.multiple_choice import mcq_vanilla_eval, extract_answer_from_item
from .utils.video_pyav import ffprobe_video_info, get_video_decode_backend, save_frames_by_indices_pyav


FAIL_MSG = 'Failed to obtain answer via API.'


@dataclass(frozen=True)
class _MotionBenchPaths:
    root: str
    ann_json: str
    meta_jsonl: str
    tsv_path: str


def _default_motionbench_root() -> str:
    # User-provided default path; can be overridden via env var.
    return os.environ.get(
        'MOTIONBENCH_ROOT',
        '/root/s3/pnorm2/videochat3/video/MotionBench/MotionBench',
    )


def _polish_answer(answer: str) -> str:
    """
    Ported from MotionBench official `motionbench/utils/answer_util.py`.
    Convert arbitrary model output to a single option letter if possible.
    """
    if answer is None:
        return ''
    s = str(answer).strip()
    if not s:
        return ''
    s = s.split(')')[0].strip()
    if '(' in s:
        try:
            s = s.split('(')[1].strip()
        except Exception:
            pass
    s = s.split(' ')[0]
    s = s.strip()
    return s[0].upper() if len(s) > 0 else ''


def _parse_mcq_from_question(q: str) -> Tuple[str, Dict[str, str]]:
    """
    Parse a MotionBench question string which is typically:
        <question line>\nA. ...\nB. ...\nC. ...
    Returns (question_stem, options_map).
    """
    q = str(q).replace('\r\n', '\n').strip()
    lines = [ln.rstrip() for ln in q.split('\n') if ln.strip() != '']
    stem_lines: List[str] = []
    opts: Dict[str, str] = {}
    opt_pat = re.compile(r'^([A-E])\s*[\.\)]\s*(.*)$')
    for ln in lines:
        m = opt_pat.match(ln.strip())
        if m:
            key = m.group(1).upper()
            val = m.group(2).strip()
            opts[key] = val
        else:
            if len(opts) == 0:
                stem_lines.append(ln)
            else:
                # continuation line for last option (rare)
                if len(opts):
                    last = sorted(opts.keys())[-1]
                    opts[last] = (opts[last] + ' ' + ln.strip()).strip()
    stem = '\n'.join(stem_lines).strip()
    return stem, opts


def _load_meta_map(meta_jsonl: str) -> Dict[str, Dict[str, str]]:
    """
    Build basename -> {question_type, video_type} mapping from MotionBench video_info.meta.jsonl.
    """
    if not os.path.exists(meta_jsonl):
        return {}
    m = {}
    with open(meta_jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            vp = obj.get('video_path', '')
            if not vp:
                continue
            base = os.path.basename(vp)
            m[base] = dict(
                question_type=str(obj.get('question_type', '') or ''),
                video_type=str(obj.get('video_type', '') or ''),
            )
    return m


def _ensure_motionbench_tsv(paths: _MotionBenchPaths) -> None:
    if os.path.exists(paths.tsv_path):
        return
    if not os.path.exists(paths.ann_json):
        raise FileNotFoundError(f'[MotionBench] annotation json not found: {paths.ann_json}')
    if not os.path.isdir(paths.root):
        raise FileNotFoundError(f'[MotionBench] root not found: {paths.root}')

    with open(paths.ann_json, 'r', encoding='utf-8') as f:
        ann = json.load(f)
    if not isinstance(ann, list):
        raise ValueError(f'[MotionBench] annotation file must be a list, got {type(ann)}')

    meta_map = _load_meta_map(paths.meta_jsonl)

    rows = []
    for idx, item in enumerate(ann):
        rel_vp = str(item.get('video_path', '')).lstrip('./')
        if not rel_vp:
            raise ValueError(f'[MotionBench] missing video_path at idx={idx}')
        abs_vp = os.path.join(paths.root, rel_vp)
        if not os.path.exists(abs_vp):
            raise FileNotFoundError(f'[MotionBench] missing video file: {abs_vp}')

        q_raw = item.get('question', '')
        stem, opts = _parse_mcq_from_question(q_raw)
        answer = str(item.get('answer', '')).strip().upper()
        if answer and answer not in list('ABCDE'):
            # best-effort normalize if provided like "(A)" etc.
            answer = _polish_answer(answer)

        base = os.path.basename(rel_vp)
        meta = meta_map.get(base, {})

        # create a stable id for frame cache directory
        video_id = os.path.splitext(rel_vp)[0].replace('/', '__')
        row = dict(
            index=idx,
            video=video_id,
            video_path=rel_vp,
            question=stem if stem else str(q_raw).strip(),
            A=opts.get('A', ''),
            B=opts.get('B', ''),
            C=opts.get('C', ''),
            D=opts.get('D', ''),
            E=opts.get('E', ''),
            answer=answer,
            question_type=meta.get('question_type', ''),
            video_type=meta.get('video_type', ''),
        )
        rows.append(row)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(paths.tsv_path), exist_ok=True)
    dump(df, paths.tsv_path)


class MotionBench(VideoBaseDataset):
    """
    MotionBench: Benchmarking Fine-grained Video Motion Understanding (Video-MCQ).

    This integration uses a local, pre-extracted MotionBench dataset folder which contains:
      - videos under `MOTIONBENCH_ROOT/<subdir>/.../*.mp4`
      - annotations `motionbench_val_new.json` (with GT answers)
      - optional metadata `video_info.meta.jsonl` (question_type / video_type)
    """

    TYPE = 'Video-MCQ'
    MODALITY = 'VIDEO'

    SYS = 'You are an AI assistant responsible for answering multiple-choice questions about a video.'
    FRAMES_TMPL_SYS = (
        'You will receive {} distinct frames uniformly sampled from a video in chronological order.\n'
        'Answer the multiple-choice question based on these frames.\n'
    )
    POST_PROMPT = 'Answer with ONLY the single uppercase letter of the correct option (A/B/C/D/E).'

    def __init__(self, dataset='MotionBench', nframe=0, fps=-1, frames_limit=2048):
        self.frames_limit = frames_limit  # following Qwen3-VL: cap frames per video for FPS sampling
        super().__init__(dataset=dataset, nframe=nframe, fps=fps)

    @classmethod
    def supported_datasets(cls):
        return ['MotionBench']

    def prepare_dataset(self, dataset='MotionBench'):
        root = _default_motionbench_root()
        paths = _MotionBenchPaths(
            root=root,
            ann_json=os.path.join(root, 'motionbench_val_new.json'),
            meta_jsonl=os.path.join(root, 'video_info.meta.jsonl'),
            tsv_path=os.path.join(root, 'MotionBench.tsv'),
        )
        _ensure_motionbench_tsv(paths)
        return dict(root=paths.root, data_file=paths.tsv_path)

    def save_video_frames(self, line, video_llm: bool = False, verbose: bool = False):
        """
        Override: MotionBench videos are stored under subdirectories, thus we must use `video_path`.
        Returns: (frame_paths, indices, video_info)
        """
        if isinstance(line, int):
            line = self.data.iloc[line]

        video_id = str(line['video'])
        rel_vp = str(line['video_path'])
        vid_path = os.path.normpath(os.path.join(self.data_root, rel_vp))
        backend = get_video_decode_backend()
        use_pyav = backend == "pyav"

        vid = None
        try:
            if backend != "pyav":
                import decord
                # Use single-threaded decoder for stability on some long/corrupted videos.
                vid = decord.VideoReader(vid_path, num_threads=1)
                fps = float(vid.get_avg_fps())
                n_frames = int(len(vid))
                duration = (n_frames / fps) if fps > 0 else 0.0
            else:
                raise RuntimeError("force pyav")
        except Exception:
            n_frames, fps, duration = ffprobe_video_info(vid_path)
            use_pyav = True

        video_info = {'fps': float(fps), 'n_frames': int(n_frames), 'duration': float(duration), 'backend': 'pyav' if use_pyav else 'decord'}

        if self.nframe > 0 and self.fps < 0:
            step_size = n_frames / (self.nframe + 1)
            indices = [int(i * step_size) for i in range(1, self.nframe + 1)]
            frame_paths = self.frame_paths(video_id)
        elif self.fps > 0:
            total_duration = duration
            required_frames = int(total_duration * self.fps) if total_duration > 0 else 0
            if required_frames > self.frames_limit:
                if verbose:
                    print(
                        f"Warning: Video `{rel_vp}` requires {required_frames} frames with {self.fps} fps. "
                        f"Truncating to {self.frames_limit} frames."
                    )
                step_size = n_frames / (self.frames_limit + 1)
                indices = [int(i * step_size) for i in range(1, self.frames_limit + 1)]
                frame_root = osp.join(self.frame_root, video_id)
                os.makedirs(frame_root, exist_ok=True)
                frame_paths = [
                    osp.join(frame_root, self.frame_tmpl.format(i, self.frames_limit))
                    for i in range(1, self.frames_limit + 1)
                ]
                if backend != "decord":
                    use_pyav = True
            else:
                step_size = fps / self.fps if self.fps > 0 else 1.0
                indices = [int(i * step_size) for i in range(required_frames)]
                frame_paths = self.frame_paths_fps(video_id, len(indices))
        else:
            raise ValueError('Either nframe > 0 or fps > 0 must be set.')

        # clamp indices
        if n_frames > 0 and len(indices) > 0:
            max_idx = n_frames - 1
            indices = [min(max(0, int(i)), max_idx) for i in indices]

        flag = np.all([osp.exists(p) for p in frame_paths])
        if not flag:
            # Avoid creating lock next to the source video (may be on read-only/limited FUSE mount like /root/s3).
            lock_dir = osp.dirname(frame_paths[0]) if len(frame_paths) else self.frame_root
            os.makedirs(lock_dir, exist_ok=True)
            lock_path = osp.join(lock_dir, '.extract.lock')
            # MotionBench can have multiple QA per video; allow long wait for another process to finish extraction.
            with portalocker.Lock(lock_path, 'w', timeout=3600):
                if not np.all([osp.exists(p) for p in frame_paths]):
                    if use_pyav or vid is None:
                        save_frames_by_indices_pyav(
                            vid_path=vid_path,
                            indices=[int(x) for x in indices],
                            frame_paths=frame_paths,
                            total_frames=video_info.get('n_frames', None),
                            desc=f"Extracting (pyav): {rel_vp}",
                        )
                    else:
                        try:
                            # Stream decode & save to avoid holding many frames in RAM (prevents OOM on long videos).
                            for frame_idx, pth in zip(indices, frame_paths):
                                if osp.exists(pth):
                                    continue
                                arr = vid[int(frame_idx)].asnumpy()
                                Image.fromarray(arr).save(pth)
                        except Exception:
                            save_frames_by_indices_pyav(
                                vid_path=vid_path,
                                indices=[int(x) for x in indices],
                                frame_paths=frame_paths,
                                total_frames=video_info.get('n_frames', None),
                                desc=f"Extracting (pyav): {rel_vp}",
                            )
        return frame_paths, indices, video_info

    def build_prompt(self, line, video_llm=False):
        if isinstance(line, int):
            assert line < len(self)
            line = self.data.iloc[line]

        # Build options string
        opts = []
        for ch in 'ABCDE':
            val = str(line.get(ch, '')).strip()
            if val and val.lower() != 'nan':
                opts.append(f'{ch}. {val}')
        opt_str = '\n'.join(opts)
        q = str(line['question']).strip()

        frames, _, video_info = self.save_video_frames(line, video_llm=video_llm)

        if video_llm:
            assert self.fps > 0
            actual_fps = (
                self.frames_limit / video_info['duration']
                if len(frames) == self.frames_limit and video_info.get('duration', 0) > 0
                else self.fps
            )
            prompt = f'{self.SYS}\n\nQuestion:\n{q}\n\nOptions:\n{opt_str}\n\n{self.POST_PROMPT}'
            return [
                dict(type='text', value=prompt),
                dict(
                    type='video',
                    value=frames,
                    sample_fps=actual_fps,
                    min_pixels=1 * 2 * 2 * 16 * 16,
                    max_pixels=640 * 32 * 32,
                    total_pixels=224000 * 4 * 16 * 16,
                ),
            ]

        msg = [dict(type='text', value=self.FRAMES_TMPL_SYS.format(len(frames)))]
        for im in frames:
            msg.append(dict(type='image', value=im))
        msg.append(dict(type='text', value=f'Question:\n{q}\n\nOptions:\n{opt_str}\n\nAnswer: '))
        msg.append(dict(type='text', value=self.POST_PROMPT))
        return msg

    @classmethod
    def evaluate(cls, eval_file, **judge_kwargs):
        assert get_file_extension(eval_file) in ['xlsx', 'json', 'tsv'], (
            'data file should be an supported format (xlsx/json/tsv) file'
        )

        judge_name = judge_kwargs.setdefault('model', 'exact_matching')
        nproc = judge_kwargs.pop('nproc', 4)

        tmp_file = get_intermediate_file_path(eval_file, f'_{judge_name}_tmp', 'pkl')
        score_file = get_intermediate_file_path(eval_file, f'_{judge_name}_score')
        acc_file = get_intermediate_file_path(eval_file, f'_{judge_name}_acc', 'json')

        if not os.path.exists(score_file):
            data = load(eval_file)
            if 'prediction' not in data.columns:
                raise ValueError('[MotionBench] eval file must contain `prediction` column.')

            if 'answer' not in data.columns:
                raise ValueError('[MotionBench] eval file must contain `answer` (GT letter).')

            meta = data[['index', 'answer']].copy()
            meta['index'] = meta['index'].astype(int)

            # Optional LLM judge: if unavailable/not working, fallback to exact matching (like `videomme.py`).
            use_judge = judge_name not in [None, 'exact_matching']
            judge_model = None
            if use_judge and gpt_key_set():
                try:
                    from .utils import build_judge, DEBUG_MESSAGE
                    judge_model = build_judge(**judge_kwargs)
                    if hasattr(judge_model, 'working') and (not judge_model.working()):
                        warnings.warn('Judge model is not working properly, will use exact matching for evaluation')
                        warnings.warn(DEBUG_MESSAGE)
                        judge_model = None
                except Exception as e:
                    warnings.warn(
                        f'Failed to build judge model ({judge_name}), fallback to exact matching: {type(e)}: {e}'
                    )
                    try:
                        from .utils import DEBUG_MESSAGE
                        warnings.warn(DEBUG_MESSAGE)
                    except Exception:
                        pass
                    judge_model = None
            elif use_judge and (not gpt_key_set()):
                warnings.warn('OPENAI_API_KEY is not set properly, will use exact matching for evaluation')
                judge_model = None

            if judge_model is None:
                # Rule-based exact matching path: polish to single letters first.
                data['prediction'] = [_polish_answer(x) for x in data['prediction']]
                scored = mcq_vanilla_eval(
                    model=None,
                    data=data,
                    meta=meta,
                    nproc=nproc,
                    result_file=tmp_file,
                    dataset_name='MotionBench',
                )
            else:
                # LLM-judge parsing path (similar to LVBench): do NOT pre-polish prediction,
                # let judge extract the option letter from the raw response.
                cache = {} if not osp.exists(tmp_file) else load(tmp_file)
                if not isinstance(cache, dict):
                    cache = {}
                answer_map = {int(i): str(a).strip().upper() for i, a in zip(meta['index'], meta['answer'])}

                updated = False
                for _, row in data.iterrows():
                    idx = int(row['index'])
                    if idx in cache:
                        continue
                    gt = answer_map.get(idx, '')
                    item = {
                        'question': str(row.get('question', '')),
                        'prediction': '' if pd.isna(row.get('prediction', '')) else str(row.get('prediction', '')),
                        'answer': gt,
                    }
                    for ch in 'ABCDE':
                        item[ch] = '' if pd.isna(row.get(ch, '')) else str(row.get(ch, ''))
                    try:
                        res = extract_answer_from_item(judge_model, item, dataset_name='MotionBench')
                        opt = str(res.get('opt', '') or '').strip().upper()
                        match_log = str(res.get('log', '') or '')
                    except Exception as e:
                        opt = ''
                        match_log = f'Exception: {type(e)} {e}'
                    hit = int(opt == gt and gt in list('ABCDE'))
                    cache[idx] = dict(hit=hit, log=f'Match Log: {match_log}. ')
                    updated = True
                if updated:
                    dump(cache, tmp_file)

                data = data[data['index'].astype(int).isin(answer_map)].copy()
                data['hit'] = [cache[int(i)]['hit'] for i in data['index'].astype(int)]
                data['log'] = [cache[int(i)]['log'] for i in data['index'].astype(int)]
                scored = data
            dump(scored, score_file)

        scored = load(score_file)
        overall = float(np.mean(scored['hit'])) if len(scored) else 0.0

        def _group_acc(col: str) -> Dict[str, float]:
            if col not in scored.columns:
                return {}
            out = {}
            vals = [x for x in scored[col] if not pd.isna(x) and str(x).strip() != '']
            for v in sorted(set(vals)):
                sub = scored[scored[col] == v]
                out[str(v)] = float(np.mean(sub['hit'])) if len(sub) else 0.0
            return out

        res = dict(
            overall_accuracy=overall,
            question_type_accuracy=_group_acc('question_type'),
            video_type_accuracy=_group_acc('video_type'),
            num_samples=int(len(scored)),
        )
        dump(res, acc_file)
        return res

