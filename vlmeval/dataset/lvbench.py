from ..smp import *
from .video_base import VideoBaseDataset
import subprocess

class LVBench(VideoBaseDataset):

    """
    LVBench: An Extreme Long Video Understanding Benchmark

    本实现假设数据组织结构与官方仓库一致：
    - 根目录下包含：
      - data/video_info.meta.jsonl
      - videos/（由 video2dataset 下载得到的视频文件）

    在第一次使用时，会根据 `video_info.meta.jsonl` 自动生成
    一个 TSV 文件 `<dataset_name>.tsv`，字段包括：
    - index: 全局题目索引
    - video: 视频的 key（YouTube ID）
    - uid: 官方题目 ID
    - video_path: 相对根目录的视频路径，例如 `videos/00000/00000000.mp4`
    - question: 含选项的完整英文问题文本
    - answer: 正确选项字母（A/B/C/D）
    - question_type: 题目能力类别列表的 JSON 字符串
    - time_reference: 时间标注字符串
    """

    TYPE = 'Video-MCQ'
    SYS = ''

    def __init__(self, dataset: str = 'LVBench', nframe: int = 0, fps: float = -1):
        self.dataset_name = dataset
        super().__init__(dataset=dataset, nframe=nframe, fps=fps)

    @classmethod
    def supported_datasets(cls):
        return ['LVBench']

    def prepare_dataset(
        self,
        dataset_name: str = 'LVBench',
        repo_id: str = '/mnt/shared-storage-user/zhuyuhan/temp_datasets/LVBench',
    ):
        """
        准备 LVBench 数据集。
        """

        dataset_path = repo_id

        def check_integrity(pth: str) -> bool:
            data_file = osp.join(pth, f'{dataset_name}.tsv')
            if not osp.exists(data_file):
                return False

            data = load(data_file)
            if 'video_path' not in data:
                return False

            # 简单检查：随机抽样若干条，确认视频文件存在
            sample_paths = data['video_path'][:16]
            for vp in sample_paths:
                if not osp.exists(osp.join(pth, vp)):
                    return False
            return True

        if not osp.exists(dataset_path):
            raise FileNotFoundError(
                f'LVBench root directory not found: {dataset_path}. '
                f'Please set environment variable LVBENCH_ROOT to your LVBench root.'
            )

        if not check_integrity(dataset_path):

            def _build_video_key_to_path(root: str):
                """
                尝试从 video2dataset 的 parquet 元数据中恢复
                key (YouTube ID) 到本地路径的映射。
                若 parquet 不存在，则退化为按行顺序与 mp4 排序顺序一一对应。
                """
                videos_dir = osp.join(root, 'videos')
                parquet_file = osp.join(videos_dir, '00000.parquet')
                key_to_path = {}

                import pandas as pd

                df = pd.read_parquet(parquet_file)
                if df.empty:
                    raise RuntimeError(f'LVBench parquet file is empty: {parquet_file}')

                key_to_path = {
                    row["url"].split("watch?v=")[1]: osp.join("videos/00000", f"{row['key']}.mp4")
                    for _, row in df.iterrows()
                }

                return key_to_path

            def generate_tsv(pth: str):
                import pandas as pd

                meta_file = osp.join(pth, 'data', 'video_info.meta.jsonl')
                if not osp.exists(meta_file):
                    raise FileNotFoundError(
                        f'LVBench meta file not found: {meta_file}. '
                        f'Please make sure video_info.meta.jsonl is downloaded.'
                    )

                data_file = osp.join(pth, f'{dataset_name}.tsv')

                key_to_path = _build_video_key_to_path(pth)

                rows = []
                with open(meta_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        meta = json.loads(line)
                        key = meta['key']
                        qa_list = meta.get('qa', [])

                        video_rel_path = key_to_path[key]

                        video_path = osp.join(dataset_path, video_rel_path)
                        if not osp.exists(video_path):
                            continue

                        for qa in qa_list:
                            rows.append(
                                dict(
                                    video=key,
                                    uid=str(qa['uid']),
                                    video_path=video_rel_path,
                                    question=qa['question'],
                                    answer=str(qa['answer']).strip(),
                                    question_type=json.dumps(qa.get('question_type', [])),
                                    time_reference=qa.get('time_reference', ''),
                                )
                            )

                if len(rows) == 0:
                    raise RuntimeError('No QA entries parsed from video_info.meta.jsonl')

                df = pd.DataFrame(rows)
                df = df.assign(index=range(len(df)))
                df.to_csv(data_file, sep='\t', index=False)

            generate_tsv(dataset_path)

        data_file = osp.join(dataset_path, f'{dataset_name}.tsv')
        return dict(data_file=data_file, root=dataset_path)

    def save_video_frames(self, video_path: str, video_llm: bool = False, verbose: bool = False):
        """
        根据 nframe 或 fps 从视频中抽帧，支持 AV1（使用 PyAV 解码）。
        """
        def get_frames_and_fps(path):
            cmd = [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-count_packets",
                "-show_entries", "stream=nb_read_packets,avg_frame_rate",
                "-of", "json",
                path,
            ]
            data = subprocess.check_output(cmd)
            info = json.loads(data)["streams"][0]

            frames = int(info["nb_read_packets"])
            num, den = info["avg_frame_rate"].split("/")
            fps = float(num) / float(den)

            return frames, fps

        import av

        vid_path = osp.join(self.data_root, video_path)
        print('video path: ', vid_path)

        n_frames, fps = get_frames_and_fps(vid_path)
        video_info = {'fps': fps, 'n_frames': n_frames}

        # ---- 计算抽帧 index ----
        if self.nframe > 0 and self.fps < 0:
            step_size = n_frames / (self.nframe + 1)
            indices = [int(i * step_size) for i in range(1, self.nframe + 1)]
            frame_paths = self.frame_paths(video_path[:-4])
        elif self.fps > 0:
            total_duration = n_frames / fps
            required_frames = int(total_duration * self.fps)
            step_size = fps / self.fps
            indices = [int(i * step_size) for i in range(required_frames)]
            frame_paths = self.frame_paths_fps(video_path[:-4], len(indices))
        else:
            raise ValueError('Either nframe > 0 or fps > 0 must be set.')

        needed = set(indices)

        if not np.all([osp.exists(p) for p in frame_paths]):
            lock_path = osp.splitext(vid_path)[0] + '.lock'
            with portalocker.Lock(lock_path, 'w', timeout=30):
                if not np.all([osp.exists(p) for p in frame_paths]):
                    container = av.open(vid_path)
                    stream = container.streams.video[0]
                    images = []

                    frame_iter = enumerate(container.decode(stream))
                    for idx, frame in tqdm(frame_iter, 
                                        total=n_frames, 
                                        desc=f"Extracting frames for {video_path}", 
                                        disable=not verbose):
                        if idx in needed:
                            out_idx = indices.index(idx)
                            pth = frame_paths[out_idx]
                            if not osp.exists(pth):
                                frame.to_image().save(pth)

                    container.close()

        return frame_paths, indices, video_info

    def build_prompt(self, line, video_llm: bool):
        """
        构建给 VLM 的多模态输入：
        - 若 video_llm=True，则直接传入整段视频；
        - 否则传入若干采样帧。
        """
        if isinstance(line, int):
            assert line < len(self)
            line = self.data.iloc[line]

        frames, indices, video_info = self.save_video_frames(line['video_path'], video_llm)

        message = [dict(type='text', value=self.SYS)]
        if video_llm:
            message.append(dict(type='video', value=osp.join(self.data_root, line['video_path'])))
        else:
            for im in frames:
                message.append(dict(type='image', value=im))

        # LVBench 的问题文本本身已包含选项 (A/B/C/D)
        prompt = line['question'] + "\nPlease select the best answer from the options above and directly provide the letter representing your choice without giving any explanation."
        message.append(dict(type='text', value=prompt))
        return message

    @staticmethod
    def _polish_answer(ans: str) -> str:
        """
        参考 LVBench 官方 `answer_util.polish_answer`，
        将模型输出归一化为单个大写字母（若可能）。
        """
        ans = (ans or '').strip()
        if not ans:
            return ''

        ans = ans.split(')')[0].strip()

        if '(' in ans:
            try:
                ans = ans.split('(')[1].strip()
            except Exception:
                pass

        ans = ans.split(' ')[0].strip()
        if len(ans) > 0:
            return ans[0].upper()
        return ''

    # It returns a dictionary
    @classmethod
    def evaluate(cls, eval_file, **judge_kwargs):
        """
        直接复刻 LVBench 官方的评测逻辑：
        - 读取 `<model>_LVBench.xlsx/tsv` 或 json，其中包含 prediction 列；
        - 与 TSV 中的 answer 比较，统计整体与各 question_type 的准确率。
        """
        from ..smp.file import get_file_extension

        assert get_file_extension(eval_file) in ['xlsx', 'json', 'tsv'], (
            'data file should be an supported format (xlsx/json/tsv) file'
        )

        data = load(eval_file)
        # 兼容 Excel / TSV：确保 prediction/answer/question_type 存在
        assert 'prediction' in data and 'answer' in data and 'question_type' in data, (
            'LVBench evaluation requires `prediction`, `answer`, and `question_type` columns.'
        )

        total_qa_num = 0
        right_num = 0
        category_right = defaultdict(int)
        category_total = defaultdict(int)

        for _, row in data.iterrows():
            pred_raw = row.get('prediction', None)
            if pd.isna(pred_raw):
                continue
            gt = str(row['answer']).strip().upper()
            if not gt:
                continue

            pred = cls._polish_answer(str(pred_raw))
            if not pred:
                continue

            equal = (pred == gt)

            qtype = row.get('question_type', '')
            # question_type 在 TSV 中以 JSON 字符串形式存储
            qtypes = []
            if isinstance(qtype, str):
                s = qtype.strip()
                if s:
                    try:
                        parsed = json.loads(s)
                        if isinstance(parsed, list):
                            qtypes = [str(x) for x in parsed]
                    except Exception:
                        # 退化处理：用逗号或分号分隔
                        for part in s.replace(';', ',').split(','):
                            part = part.strip().strip('"').strip("'")
                            if part:
                                qtypes.append(part)
            elif isinstance(qtype, (list, tuple)):
                qtypes = [str(x) for x in qtype]

            if len(qtypes) == 0:
                # 若没有类别信息，也至少统计到 overall 中
                category_total['__NO_CATEGORY__'] += 1
                if equal:
                    category_right['__NO_CATEGORY__'] += 1

            for c in qtypes:
                category_total[c] += 1
                if equal:
                    category_right[c] += 1

            if equal:
                right_num += 1
            total_qa_num += 1

        category_acc = {}
        for key in category_total:
            if category_total[key] > 0:
                category_acc[key] = category_right[key] / category_total[key]
            else:
                category_acc[key] = 0.0

        acc = float(right_num) / total_qa_num if total_qa_num > 0 else 0.0
        category_acc.update({'acc': acc})
        return category_acc

