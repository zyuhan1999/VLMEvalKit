import argparse
import os
from vlmeval.dataset import build_dataset, VideoMMMU, VideoMME, LongVideoBench, LVBench, MMVU, TOMATO
from vlmeval.dataset.minerva import Minerva
from vlmeval.dataset.motionbench import MotionBench
from vlmeval.dataset.vidi_vue_tr import VUE_TR
from vlmeval.dataset.timelens import TimeLens_Charades, TimeLens_ActivityNet, TimeLens_QVHighlights
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(
        description="Pre-extract video frames using VLMEvalKit's built-in logic."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="VideoMMMU_8frame",
        help=(
            "Dataset name used by VLMEvalKit, e.g. `VideoMMMU_8frame`, "
            "`VideoMMMU_64frame`, `VideoMMMU_1fps`, etc. "
            "It should be a key in `supported_video_datasets` or an officially supported dataset name."
        ),
    )
    return parser.parse_args()

def run_videomme(dataset):
    data = dataset.data
    print(f"[Info] Total samples to process: {len(data)}")
    print(f"[Info] Frames will be saved under {os.environ['LMUData']}/images/<dataset>/... ")

    # Iterate over each row, mimicking the logic used in `VideoMMMU.build_prompt`
    # for i, (_, line) in enumerate(data.iterrows()):
    for i, (_, line) in enumerate(data.iloc[::-1].iterrows()):
        video_pth = line["video"] if "video" in line else None

        print(f"[{i+1}/{len(data)}] Extracting frames for video...")

        dataset.save_video_frames(video_pth, verbose=True)

    print("[Done] Frame extraction finished.")


def run_videommmu(dataset):
    data = dataset.data
    print(f"[Info] Total samples to process: {len(data)}")
    print(f"[Info] Frames will be saved under {os.environ['LMUData']}/images/<dataset>/... ")

    # Iterate over each row, mimicking the logic used in `VideoMMMU.build_prompt`
    for i, (_, line) in enumerate(data.iterrows()):
        sample_id = line["id"]
        video_pth = line["video"]

        print(f"[{i+1}/{len(data)}] Extracting frames for video: {sample_id}...")

        dataset.save_video_frames(sample_id, video_pth, verbose=True)

    print("[Done] Frame extraction finished.")


from concurrent.futures import ThreadPoolExecutor, as_completed

def run_longvideobench(dataset):
    data = dataset.data
    print(f"[Info] Total samples to process: {len(data)}")
    print(f"[Info] Frames will be saved under {os.environ['LMUData']}/images/<dataset>/... ")

    # Iterate over each row, mimicking the logic used in `VideoMMMU.build_prompt`
    for i, (_, line) in enumerate(data.iterrows()):
        video_pth = line["video_path"] if "video" in line else None

        print(f"[{i+1}/{len(data)}] Extracting frames for video...")

        dataset.save_video_frames(video_pth, verbose=True)

    print("[Done] Frame extraction finished.")

def run_lvbench(dataset):
    data = dataset.data
    print(f"[Info] Total samples to process: {len(data)}")
    print(f"[Info] Frames will be saved under {os.environ['LMUData']}/images/<dataset>/... ")

    # Iterate over each row, mimicking the logic used in `VideoMMMU.build_prompt`
    for i, (_, line) in enumerate(data.iterrows()):
        video_pth = line["video_path"] if "video" in line else None

        print(f"[{i+1}/{len(data)}] Extracting frames for video...")

        dataset.save_video_frames(video_pth, verbose=True)

    print("[Done] Frame extraction finished.")

def run_mmvu(dataset):
    data = dataset.data
    print(f"[Info] Total samples to process: {len(data)}")
    print(f"[Info] Frames will be saved under {os.environ['LMUData']}/images/<dataset>/... ")

    # Iterate over each row, mimicking the logic used in `VideoMMMU.build_prompt`
    for i, (_, line) in enumerate(data.iterrows()):
        video_pth = line['video']
        print(f"[{i+1}/{len(data)}] Extracting frames for video: {video_pth}...")
        dataset.save_video_frames(video_pth, verbose=True)

    print("[Done] Frame extraction finished.")

def run_tomato(dataset):
    data = dataset.data
    print(f"[Info] Total samples to process: {len(data)}")
    print(f"[Info] Frames will be saved under {os.environ['LMUData']}/images/<dataset>/... ")

    # Iterate over each row, mimicking the logic used in `VideoMMMU.build_prompt`
    for i, (_, line) in enumerate(data.iterrows()):
        video_pth = line['video']
        print(f"[{i+1}/{len(data)}] Extracting frames for video: {video_pth}...")
        dataset.save_video_frames(video_pth, verbose=True)

    print("[Done] Frame extraction finished.")

def run_minerva(dataset: Minerva):
    """
    Minerva can have multiple QA per video; dedupe by `video` stem.
    """
    data = dataset.data
    print(f"[Info] Total samples: {len(data)}")
    if 'video' not in data.columns:
        raise ValueError("[Minerva] dataset TSV must contain `video` column.")
    videos = list(dict.fromkeys([str(x) for x in data['video'].tolist()]))
    print(f"[Info] Unique videos to process: {len(videos)}")
    print(f"[Info] Frames will be saved under {os.environ['LMUData']}/images/<dataset>/... ")
    for i, v in enumerate(videos):
        print(f"[{i+1}/{len(videos)}] Extracting frames for video: {v}...")
        try:
            dataset.save_video_frames(v, verbose=True)
        except Exception as e:
            print(f"[Warn] Failed to extract frames for video={v!r}: {type(e).__name__}: {e}")
            continue
    print("[Done] Frame extraction finished.")


def run_motionbench(dataset: MotionBench):
    """
    MotionBench may contain multiple QA per video; dedupe by `video` id.
    """
    data = dataset.data
    print(f"[Info] Total samples: {len(data)}")
    if 'video' not in data.columns:
        raise ValueError("[MotionBench] dataset TSV must contain `video` column.")
    uniq = data.drop_duplicates('video').reset_index(drop=True)
    print(f"[Info] Unique videos to process: {len(uniq)}")
    print(f"[Info] Frames will be saved under {os.environ['LMUData']}/images/<dataset>/... ")
    for i, (_, row) in enumerate(uniq.iterrows()):
        print(f"[{i+1}/{len(uniq)}] Extracting frames for video: {row['video']}...")
        # try:
        dataset.save_video_frames(row, verbose=True)
        # except Exception as e:
        #     print(f"[Warn] Failed to extract frames for video={row.get('video')!r}: {type(e).__name__}: {e}")
        #     continue
    print("[Done] Frame extraction finished.")


def run_vue_tr(dataset: VUE_TR):
    """
    VUE_TR is query-based; dedupe by `video_id` to avoid repeated decode.
    """
    data = dataset.data
    print(f"[Info] Total samples: {len(data)}")
    if 'video_id' not in data.columns:
        raise ValueError("[VUE_TR] dataset TSV must contain `video_id` column.")
    uniq = data.drop_duplicates('video_id').reset_index(drop=True)
    print(f"[Info] Unique videos to process: {len(uniq)}")
    print(f"[Info] Frames will be saved under {os.environ['LMUData']}/images/<dataset>/... ")
    for i, (_, row) in enumerate(uniq.iterrows()):
        print(f"[{i+1}/{len(uniq)}] Extracting frames for video_id: {row['video_id']}...")
        try:
            dataset.save_video_frames(row)
        except FileNotFoundError as e:
            # Some youtube videos are not downloadable; skip them and continue.
            print(f"[Warn] {e}")
            continue
        except Exception as e:
            print(f"[Warn] Failed to extract frames for video_id={row.get('video_id')!r}: {type(e).__name__}: {e}")
            continue
    print("[Done] Frame extraction finished.")


def run_timelens(dataset):
    """
    TimeLens is span/query-based; easiest is to trigger its internal saving by calling build_prompt(video_llm=False).
    Dedupe by video id to avoid repeated decode.
    """
    data = dataset.data
    print(f"[Info] Total samples: {len(data)}")
    if 'video' not in data.columns:
        raise ValueError("[TimeLens] dataset TSV must contain `video` column.")
    uniq = data.drop_duplicates('video').reset_index(drop=True)
    print(f"[Info] Unique videos to process: {len(uniq)}")
    print(f"[Info] Frames will be saved under {os.environ['LMUData']}/images/<dataset>/... ")
    for i, (_, row) in enumerate(uniq.iterrows()):
        print(f"[{i+1}/{len(uniq)}] Extracting frames for video: {row['video']}...")
        # This will save frames (with timestamp interleave) if they don't exist.
    # try:
        dataset.build_prompt(row, video_llm=False)
        # except Exception as e:
        #     print(f"[Warn] Failed to extract frames for video={row.get('video')!r}: {type(e).__name__}: {e}")
        #     continue
    print("[Done] Frame extraction finished.")

def main():
    assert "LMUData" in os.environ
    save_root = os.environ["LMUData"]
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    args = parse_args()

    print(f"[Info] Building dataset: {args.dataset}")
    dataset = build_dataset(args.dataset)
    if dataset is None:
        raise ValueError(f"Failed to build dataset '{args.dataset}'. Please check the name and data files.")

    if not hasattr(dataset, "save_video_frames"):
        raise ValueError(
            f"Dataset '{args.dataset}' does not provide `save_video_frames`. "
            "Make sure you are using a video dataset that supports frame extraction."
        )

    if isinstance(dataset, VideoMMMU):
        run_videommmu(dataset)
    elif isinstance(dataset, VideoMME):
        run_videomme(dataset)
    elif isinstance(dataset, LongVideoBench):
        run_longvideobench(dataset)
    elif isinstance(dataset, LVBench):
        run_lvbench(dataset)
    elif isinstance(dataset, MMVU):
        run_mmvu(dataset)
    elif isinstance(dataset, TOMATO):
        run_tomato(dataset)
    elif isinstance(dataset, Minerva):
        run_minerva(dataset)
    elif isinstance(dataset, MotionBench):
        run_motionbench(dataset)
    elif isinstance(dataset, VUE_TR):
        run_vue_tr(dataset)
    elif isinstance(dataset, (TimeLens_Charades, TimeLens_ActivityNet, TimeLens_QVHighlights)):
        run_timelens(dataset)
    else:
        raise ValueError(f"Dataset '{args.dataset}' is not supported.")


if __name__ == "__main__":
    main()
