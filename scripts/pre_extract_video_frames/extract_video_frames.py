import argparse
import os
from vlmeval.dataset import build_dataset, VideoMMMU, VideoMME, LongVideoBench, LVBench, MMVU, TOMATO
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
    else:
        raise ValueError(f"Dataset '{args.dataset}' is not supported.")


if __name__ == "__main__":
    main()
