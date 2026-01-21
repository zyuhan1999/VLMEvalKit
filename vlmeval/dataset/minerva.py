import json
import os
import re
import unicodedata
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd

from ..smp import *
from ..smp.file import get_intermediate_file_path, get_file_extension
from .video_base import VideoBaseDataset
from .utils.multiple_choice import mcq_vanilla_eval, extract_answer_from_item
from .utils.video_pyav import ffprobe_video_info, get_video_decode_backend, save_frames_by_indices_pyav


FAIL_MSG = 'Failed to obtain answer via API.'


@dataclass(frozen=True)
class _MinervaPaths:
    root: str
    videos_dir: str
    ann_json: str
    url_title_map_tsv: str
    tsv_path: str


def _default_minerva_root() -> str:
    # User-provided default path for this workspace; can be overridden via env var.
    return os.environ.get('MINERVA_ROOT', '/mnt/petrelfs/zhuyuhan/s3/videogpu/zhuyuhan/benchmarks/Minerva').strip()


def _canonicalize_title(s: str) -> str:
    """Normalize unicode and map some 'look-alike' punctuation back to ASCII for matching."""
    if s is None:
        return ''
    s = unicodedata.normalize('NFKC', str(s)).strip().lower()
    # Unify common downloader replacements back to ASCII equivalents.
    trans = str.maketrans({
        '｜': '|',
        '⧸': '/',
        '／': '/',
        '：': ':',
        '？': '?',
        '＊': '*',
        '＂': '"',
        '“': '"',
        '”': '"',
        '’': "'",
        '‘': "'",
    })
    s = s.translate(trans)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def _sanitize_title_for_filename(title: str) -> str:
    """
    Approximate common yt-dlp/youtube-dl filesystem sanitization for Windows-forbidden chars.
    In this dataset, we observed:
      - '|' -> '｜'
      - '/' -> '⧸'
      - ':' -> '：'
      - '?' -> '？'
      - '*' -> '＊'
      - '"' -> '＂'
    """
    if title is None:
        return ''
    title = str(title)
    trans = str.maketrans({
        '|': '｜',
        '/': '⧸',
        '\\': '⧸',
        ':': '：',
        '?': '？',
        '*': '＊',
        '"': '＂',
        '<': '＜',
        '>': '＞',
    })
    return title.translate(trans)


def _build_video_stem_index(videos_dir: str) -> Dict[str, str]:
    """
    Build a mapping: canonical_title -> stem.
    If multiple stems collide after canonicalization, we keep the first one.
    """
    idx: Dict[str, str] = {}
    with os.scandir(videos_dir) as it:
        for ent in it:
            if not ent.is_file():
                continue
            name = ent.name
            if not name.lower().endswith('.mp4'):
                continue
            stem = name[:-4]
            key = _canonicalize_title(stem)
            if key and key not in idx:
                idx[key] = stem
    return idx


def _resolve_video_stem(
    video_id: str,
    title_raw: str,
    videos_dir: str,
    stem_index: Dict[str, str],
) -> str:
    """
    Resolve a local mp4 stem (filename without extension) for a given (video_id, title_raw).
    """
    candidates: List[str] = []
    if title_raw:
        candidates.append(title_raw)
        candidates.append(_sanitize_title_for_filename(title_raw))
        # A few extra minor normalizations
        candidates.append(_sanitize_title_for_filename(title_raw).strip().rstrip('.'))
    for cand in candidates:
        if not cand:
            continue
        p = os.path.join(videos_dir, f'{cand}.mp4')
        if os.path.exists(p):
            return cand

    # Fallback: canonical lookup among existing filenames.
    can_key = _canonicalize_title(_sanitize_title_for_filename(title_raw))
    if can_key in stem_index:
        return stem_index[can_key]

    # Last fallback: try raw title canonical form.
    can_key2 = _canonicalize_title(title_raw)
    if can_key2 in stem_index:
        return stem_index[can_key2]

    raise FileNotFoundError(
        f'[Minerva] Cannot find local video for video_id={video_id}, title={title_raw!r} under {videos_dir}'
    )


def _read_url_title_map(tsv_path: str) -> pd.DataFrame:
    # Some released mappings may use literal "\\t" instead of real tab characters.
    rows = []
    with open(tsv_path, 'r', encoding='utf-8') as f:
        for raw in f:
            line = raw.rstrip('\n')
            if not line:
                continue
            if '\t' in line:
                parts = line.split('\t')
            else:
                parts = line.split('\\t')
            if len(parts) < 3:
                # best-effort fallback
                parts = (parts + [''] * 3)[:3]
            video_id, title, url = parts[0], parts[1], parts[2]
            rows.append(
                dict(
                    video_id=str(video_id).strip(),
                    title=str(title).strip(),
                    url=str(url).strip(),
                )
            )
    return pd.DataFrame(rows)


def _ensure_minerva_tsv(paths: _MinervaPaths) -> None:
    if os.path.exists(paths.tsv_path):
        # Trust local TSV if it exists; users can delete it to regenerate.
        return

    if not os.path.exists(paths.ann_json):
        raise FileNotFoundError(f'[Minerva] annotation json not found: {paths.ann_json}')
    if not os.path.exists(paths.url_title_map_tsv):
        raise FileNotFoundError(f'[Minerva] url_title_map.tsv not found: {paths.url_title_map_tsv}')
    if not os.path.isdir(paths.videos_dir):
        raise FileNotFoundError(f'[Minerva] videos dir not found: {paths.videos_dir}')

    url_map = _read_url_title_map(paths.url_title_map_tsv)
    id2title = {r['video_id']: r['title'] for _, r in url_map.iterrows()}

    with open(paths.ann_json, 'r', encoding='utf-8') as f:
        records = json.load(f)
    if not isinstance(records, list):
        raise ValueError(f'[Minerva] minerva.json should be a list, got {type(records)}')

    stem_index = _build_video_stem_index(paths.videos_dir)

    rows = []
    for i, r in enumerate(records):
        video_id = str(r.get('video_id', ''))
        title_raw = id2title.get(video_id, '')
        video_stem = _resolve_video_stem(video_id, title_raw, paths.videos_dir, stem_index)

        answer_id = int(r.get('answer_id'))
        gt_letter = chr(ord('A') + answer_id)
        choices = [r.get(f'answer_choice_{k}') for k in range(5)]

        row = dict(
            index=i,
            key=r.get('key', ''),
            video_id=video_id,
            video=video_stem,
            question=r.get('question', ''),
            A=choices[0],
            B=choices[1],
            C=choices[2],
            D=choices[3],
            E=choices[4],
            answer=gt_letter,  # required by VLMEvalKit MCQ evaluator
            answer_id=answer_id,
            answer_text=r.get('answer', choices[answer_id] if 0 <= answer_id < 5 else ''),
            question_type=r.get('question_type', ''),
            split=r.get('split', ''),
            category=r.get('category', ''),
            reasoning=r.get('reasoning', ''),
        )
        rows.append(row)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(paths.tsv_path), exist_ok=True)
    dump(df, paths.tsv_path)


class Minerva(VideoBaseDataset):
    """
    MINERVA: Evaluating Complex Video Reasoning
    - Local dataset root default: /mnt/shared-storage-user/zhuyuhan/temp_datasets/Minerva (override via MINERVA_ROOT)
    - Videos are stored by *title* (sanitized), while annotations use *video_id*; we resolve via url_title_map.tsv.
    """

    TYPE = 'Video-MCQ'

    SYS = (
        "You are an AI assistant responsible for answering questions about a video.\n"
        "Select the best option (A/B/C/D/E) based on the video.\n"
    )

    FRAMES_TMPL_SYS = (
        "You will receive {} distinct frames uniformly sampled from a video in chronological order.\n"
        "Answer the multiple-choice question based on these frames.\n"
    )

    POST_PROMPT = "Answer with ONLY the single uppercase letter of the correct option (A/B/C/D/E)."

    def __init__(self, dataset='Minerva', nframe=0, fps=-1, frames_limit=2048):
        self.frames_limit = frames_limit  # following Qwen3-VL: cap frames per video for FPS sampling
        super().__init__(dataset=dataset, nframe=nframe, fps=fps)

    @classmethod
    def supported_datasets(cls):
        return ['Minerva']

    def prepare_dataset(self, dataset='Minerva'):
        root = _default_minerva_root()
        paths = _MinervaPaths(
            root=root,
            videos_dir=os.path.join(root, 'videos'),
            ann_json=os.path.join(root, 'minerva.json'),
            url_title_map_tsv=os.path.join(root, 'url_title_map.tsv'),
            tsv_path=os.path.join(root, 'Minerva.tsv'),
        )
        _ensure_minerva_tsv(paths)
        # VideoBaseDataset expects videos at: root/<video>.mp4
        return dict(root=paths.videos_dir, data_file=paths.tsv_path)

    def save_video_frames(self, video: str, video_llm: bool = False, verbose: bool = False):
        """
        Override to support frames_limit for FPS sampling and return (frame_paths, indices, video_info).
        """
        vid_path = osp.join(self.data_root, f'{video}.mp4')
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
            # Fallback to ffprobe+PyAV (robust for long/problematic videos).
            n_frames, fps, duration = ffprobe_video_info(vid_path)
            use_pyav = True

        video_info = {'fps': float(fps), 'n_frames': int(n_frames), 'duration': float(duration), 'backend': 'pyav' if use_pyav else 'decord'}

        if self.nframe > 0 and self.fps < 0:
            step_size = n_frames / (self.nframe + 1)
            indices = [int(i * step_size) for i in range(1, self.nframe + 1)]
            frame_paths = self.frame_paths(video)
        elif self.fps > 0:
            total_duration = duration
            required_frames = int(total_duration * self.fps) if total_duration > 0 else 0
            if required_frames > self.frames_limit:
                if verbose:
                    print(
                        f"Warning: Video `{video}` requires {required_frames} frames with {self.fps} fps. "
                        f"Truncating to {self.frames_limit} frames."
                    )
                step_size = n_frames / (self.frames_limit + 1)
                indices = [int(i * step_size) for i in range(1, self.frames_limit + 1)]
                frame_root = osp.join(self.frame_root, video)
                os.makedirs(frame_root, exist_ok=True)
                frame_paths = [
                    osp.join(frame_root, self.frame_tmpl.format(i, self.frames_limit))
                    for i in range(1, self.frames_limit + 1)
                ]
                # Long video: prefer robust sequential decode.
                if backend != "decord":
                    use_pyav = True
            else:
                step_size = fps / self.fps if self.fps > 0 else 1.0
                indices = [int(i * step_size) for i in range(required_frames)]
                frame_paths = self.frame_paths_fps(video, len(indices))
        else:
            raise ValueError('Either nframe > 0 or fps > 0 must be set.')

        # clamp indices
        if n_frames > 0 and len(indices) > 0:
            max_idx = n_frames - 1
            indices = [min(max(0, int(i)), max_idx) for i in indices]

        flag = np.all([osp.exists(p) for p in frame_paths])
        if not flag:
            # NOTE: Do NOT create lock next to the source video. On some clusters `/root/s3/...` is a
            # read-only/limited FUSE mount where creating new files is not permitted.
            # Use the (writable) frame cache directory under LMUData instead.
            lock_dir = osp.dirname(frame_paths[0]) if len(frame_paths) else self.frame_root
            os.makedirs(lock_dir, exist_ok=True)
            lock_path = osp.join(lock_dir, '.extract.lock')
            # Minerva has multiple QA per video; under torchrun different ranks may touch the same video.
            # Extracting 2fps up to 2048 frames can take minutes, so use a large lock timeout.
            with portalocker.Lock(lock_path, 'w', timeout=3600):
                if not np.all([osp.exists(p) for p in frame_paths]):
                    if use_pyav or vid is None:
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
                                arr = vid[int(frame_idx)].asnumpy()
                                Image.fromarray(arr).save(pth)
                        except Exception:
                            # Fallback: robust sequential decode.
                            save_frames_by_indices_pyav(
                                vid_path=vid_path,
                                indices=[int(x) for x in indices],
                                frame_paths=frame_paths,
                                total_frames=video_info.get('n_frames', None),
                                desc=f"Extracting (pyav): {video}",
                            )
        return frame_paths, indices, video_info

    def build_prompt(self, line, video_llm=False):
        if isinstance(line, int):
            assert line < len(self)
            line = self.data.iloc[line]

        q = str(line['question'])
        opts = [
            ('A', str(line.get('A', ''))),
            ('B', str(line.get('B', ''))),
            ('C', str(line.get('C', ''))),
            ('D', str(line.get('D', ''))),
            ('E', str(line.get('E', ''))),
        ]
        opt_str = '\n'.join([f'({k}) {v}' for k, v in opts])
        prompt = f"{self.SYS}\nQuestion: {q}\nOptions:\n{opt_str}\n\n{self.POST_PROMPT}"

        frames, _, video_info = self.save_video_frames(str(line['video']), video_llm=video_llm)

        if video_llm:
            assert self.fps > 0
            actual_fps = (
                self.frames_limit / video_info['duration']
                if len(frames) == self.frames_limit and video_info.get('duration', 0) > 0
                else self.fps
            )
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
        msg.append(dict(type='text', value=f'Question: {q}\nOptions:\n{opt_str}\nAnswer: '))
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

            # Ensure MCQ columns exist for evaluator.
            if not all([c in data.columns for c in list('ABCDE')]):
                for i, ch in enumerate('ABCDE'):
                    src = f'answer_choice_{i}'
                    if src in data.columns:
                        data[ch] = data[src]

            if 'answer' not in data.columns:
                raise ValueError('[Minerva] eval file must contain `answer` (GT letter) for evaluation.')

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
                scored = mcq_vanilla_eval(
                    model=None,
                    data=data,
                    meta=meta,
                    nproc=nproc,
                    result_file=tmp_file,
                    dataset_name='Minerva',
                )
            else:
                # LLM-judge parsing path (similar to LVBench): use LLM to extract final option letter with caching.
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
                        res = extract_answer_from_item(judge_model, item, dataset_name='Minerva')
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
            for v in sorted(set([x for x in scored[col] if not pd.isna(x)])):
                sub = scored[scored[col] == v]
                out[str(v)] = float(np.mean(sub['hit'])) if len(sub) else 0.0
            return out

        res = dict(
            overall_accuracy=overall,
            split_accuracy=_group_acc('split'),
            question_type_accuracy=_group_acc('question_type'),
            category_accuracy=_group_acc('category'),
        )
        dump(res, acc_file)
        return res

