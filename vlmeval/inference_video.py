import torch
import torch.distributed as dist
from vlmeval.config import supported_VLM
from vlmeval.utils import track_progress_rich
from vlmeval.smp import *

FAIL_MSG = 'Failed to obtain answer via API.'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, nargs='+', required=True)
    parser.add_argument('--model', type=str, nargs='+', required=True)
    parser.add_argument('--nproc', type=int, default=4, required=True)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    return args


# Only API model is accepted
def infer_data_api(model, work_dir, model_name, dataset, samples_dict={}, api_nproc=4):
    rank, world_size = get_rank_and_world_size()
    assert rank == 0 and world_size == 1
    dataset_name = dataset.dataset_name
    model = supported_VLM[model_name]() if isinstance(model, str) else model
    assert getattr(model, 'is_api', False)

    indices = list(samples_dict.keys())
    if getattr(model,'backend', None) == 'genai':
        if dataset.nframe > 0:
            print(
                'Gemini model (with genai backend) does not support nframe, '
                'will set its VIDEO_LLM to False to enable multi-image input for video.'
            )
            setattr(model, 'VIDEO_LLM', False)
        else:
            print('Gemini model (with genai backend) is a video-llm, '
                  'will reset fps setting in model to match the dataset.')
            setattr(model, 'fps', dataset.fps)
            print(f'The fps is set to {dataset.fps} for the model {model_name}.')
    elif getattr(model,'backend', None) == 'vertex':
        print('Gemini model (with vertex backend) does not support video input, '
              'will set its VIDEO_LLM to False to enable multi-image input for video.')
        setattr(model, 'VIDEO_LLM', False)

    packstr = 'pack' if getattr(dataset, 'pack', False) else 'nopack'
    build_prompt_input = [(samples_dict[idx], getattr(model, 'VIDEO_LLM', False)) for idx in indices]
    if dataset.nframe > 0:
        struct_tmp_file = f'{work_dir}/{model_name}_{dataset_name}_{dataset.nframe}frame_{packstr}_structs.pkl'
    else:
        struct_tmp_file = f'{work_dir}/{model_name}_{dataset_name}_{dataset.fps}fps_{packstr}_structs.pkl'
    structs = track_progress_rich(
        dataset.build_prompt,
        tasks=build_prompt_input,
        nproc=api_nproc,
        save=struct_tmp_file,
        keys=indices,
    )

    if dataset.nframe > 0:
        out_file = f'{work_dir}/{model_name}_{dataset_name}_{dataset.nframe}frame_{packstr}_supp.pkl'
    else:
        out_file = f'{work_dir}/{model_name}_{dataset_name}_{dataset.fps}fps_{packstr}_supp.pkl'
    res = load(out_file) if osp.exists(out_file) else {}

    structs = [s for i, s in zip(indices, structs) if i not in res or res[i] == FAIL_MSG]
    structs = [struct for struct in structs if struct is not None]
    indices = [i for i in indices if i not in res or res[i] == FAIL_MSG]

    gen_func = model.generate
    structs = [dict(message=struct, dataset=dataset_name) for struct in structs]

    if len(structs):
        track_progress_rich(gen_func, structs, nproc=api_nproc, chunksize=api_nproc, save=out_file, keys=indices)

    res = load(out_file)
    return res


def infer_data(
        model,
        model_name,
        work_dir,
        dataset,
        out_file,
        verbose=False,
        api_nproc=4,
        use_vllm=False,
        assigned_indices=None):
    res = load(out_file) if osp.exists(out_file) else {}
    rank, world_size = get_rank_and_world_size()
    dataset_name = dataset.dataset_name

    sample_indices = list(dataset.videos) if getattr(dataset, 'pack', False) else list(dataset.data['index'])
    samples = list(dataset.videos) if getattr(dataset, 'pack', False) else list(range(len(dataset.data)))
    sample_map = {i: s for i, s in zip(sample_indices, samples)}

    if assigned_indices is not None:
        sample_indices_sub = assigned_indices
    else:
        sample_indices_sub = sample_indices[rank::world_size]

    if np.all([idx in res for idx in sample_indices_sub]):
        return model
    sample_indices_subrem = [x for x in sample_indices_sub if x not in res]

    kwargs = {}
    if model_name is not None and (
        'Llama-4' in model_name
        or 'Qwen2-VL' in model_name
        or 'Qwen2.5-VL' in model_name
        or 'Qwen2.5-Omni' in model_name
    ):
        kwargs = {'use_vllm': use_vllm}

    # (25.06.05) In newer version of transformers (after 4.50), with device_map='auto' and torchrun launcher,
    # Transformers automatically adopt TP parallelism, which leads to compatibility problems with VLMEvalKit
    # (In VLMEvalKit, we use torchrun to launch multiple model instances on a single node).
    # To bypass this problem, we unset `WORLD_SIZE` before building the model to not use TP parallel.
    ws_bak = os.environ.pop('WORLD_SIZE', None)
    model = supported_VLM[model_name](**kwargs) if isinstance(model, str) else model
    if ws_bak:
        os.environ['WORLD_SIZE'] = ws_bak

    is_api = getattr(model, 'is_api', False)
    if is_api:
        assert world_size == 1
        supp = infer_data_api(
            model=model,
            work_dir=work_dir,
            model_name=model_name,
            dataset=dataset,
            samples_dict={k: sample_map[k] for k in sample_indices_subrem},
            api_nproc=api_nproc)
        for k in sample_indices_subrem:
            assert k in supp
        res.update(supp)
        dump(res, out_file)
        return model

    assert not getattr(dataset, 'pack', False), 'Current model not supported pack mode!'
    if 'megabench' in dataset_name.lower() and 'llava_onevision' in model_name:
        print(
            'LLaVA-OneVision does not support Megabench dataset as video dataset, '
            'will set its VIDEO_LLM to False to enable multi-image input for video.'
        )
        setattr(model, 'VIDEO_LLM', False)

    for i, idx in tqdm(enumerate(sample_indices_subrem), total=len(sample_indices_subrem), desc=f"[Rank{rank}]"):
        if idx in res:
            continue
        if getattr(model, 'nframe', None) is not None and getattr(model, 'nframe', 0) > 0:
            if dataset.nframe > 0:
                if getattr(model, 'nframe', 0) != dataset.nframe:
                    print(f'{model_name} is a video-llm model, nframe is set to {dataset.nframe}, not using default')
                    setattr(model, 'nframe', dataset.nframe)
            elif getattr(model, 'fps', 0) == 0:
                raise ValueError(f'fps is not suitable for {model_name}')
            else:
                setattr(model, 'nframe', None)
        if getattr(model, 'fps', None) is not None and getattr(model, 'fps', 0) > 0:
            if dataset.fps > 0:
                if getattr(model, 'fps', 0) != dataset.fps:
                    print(f'{model_name} is a video-llm model, fps is set to {dataset.fps}, not using default')
                    setattr(model, 'fps', dataset.fps)
            elif getattr(model, 'nframe', 0) == 0:
                raise ValueError(f'nframe is not suitable for {model_name}')
            else:
                setattr(model, 'fps', None)
        if (
            'Qwen2-VL' in model_name
            or 'Qwen2.5-VL' in model_name
            or 'Qwen2.5-Omni' in model_name
        ):
            if getattr(model, 'nframe', None) is None and dataset.nframe > 0:
                print(f'using {model_name} default setting for video, dataset.nframe is ommitted')
            if getattr(model, 'fps', None) is None and dataset.fps > 0:
                print(f'using {model_name} default setting for video, dataset.fps is ommitted')
        if 'SUB_DATASET' in dataset.data.iloc[sample_map[idx]]:
            dataset_name = dataset.data.iloc[sample_map[idx]]['SUB_DATASET']
        if hasattr(model, 'use_custom_prompt') and model.use_custom_prompt(dataset_name):
            if dataset.nframe == 0:
                raise ValueError(f'nframe must be set for custom prompt, fps is not suitable for {model_name}')
            struct = model.build_prompt(
                dataset.data.iloc[sample_map[idx]], dataset=dataset, video_llm=getattr(model, 'VIDEO_LLM', False)
            )
        else:
            struct = dataset.build_prompt(
                sample_map[idx], video_llm=getattr(model, 'VIDEO_LLM', False)
            )
        if struct is None:
            continue

        # If `SKIP_ERR` flag is set, the model will skip the generation if error is encountered
        if os.environ.get('SKIP_ERR', False) == '1':
            FAIL_MSG = 'Failed to obtain answer'
            try:
                response = model.generate(message=struct, dataset=dataset_name)
            except RuntimeError as err:
                torch.cuda.synchronize()
                warnings.error(f'{type(err)} {str(err)}')
                response = f'{FAIL_MSG}: {type(err)} {str(err)}'
        else:
            response = model.generate(message=struct, dataset=dataset_name)
        torch.cuda.empty_cache()

        if verbose:
            print(response, flush=True)

        res[idx] = response
        if (i + 1) % 20 == 0:
            dump(res, out_file)

    # For the default (non-dynamic) scheduling strategy, keep only the samples
    # belonging to this rank. When dynamic load balancing is enabled
    # (assigned_indices is not None), we keep all cached results in this file,
    # since different ranks may take over unfinished samples from others.
    if assigned_indices is None:
        res = {k: res[k] for k in sample_indices_sub}
    dump(res, out_file)
    return model


# A wrapper for infer_data, do the pre & post processing
def infer_data_job_video(
        model,
        work_dir,
        model_name,
        dataset,
        result_file_name,
        verbose=False,
        api_nproc=4,
        use_vllm=False):

    dataset_name = dataset.dataset_name
    rank, world_size = get_rank_and_world_size()
    result_file = osp.join(work_dir, result_file_name)
    # Dump Predictions to Prev File if result file exists
    if osp.exists(result_file):
        return model

    tmpl = osp.join(work_dir, '{}' + f'{world_size}_{osp.splitext(result_file_name)[0]}.pkl')
    out_file = tmpl.format(rank)

    # -------- Dynamic load-balancing for video inference --------
    # We support resuming from previous partial results (per-rank pkl files)
    # while re-balancing the remaining samples evenly across all ranks.
    # Existing results in 0_*.pkl, 1_*.pkl, ... will be respected and kept.
    assigned_indices = None
    if world_size > 1:
        # Build a global task list and remove all indices that have already
        # been processed by any rank (according to existing per-rank pkl files).
        if rank == 0:
            # All ranks share the same dataset, so we can safely construct the
            # global index list on rank 0 and then broadcast the assignment.
            sample_indices = (
                list(dataset.videos)
                if getattr(dataset, 'pack', False)
                else list(dataset.data['index'])
            )

            done_indices = set()
            for i in range(world_size):
                part_file = tmpl.format(i)
                if osp.exists(part_file):
                    part_res = load(part_file)
                    done_indices.update(part_res.keys())

            remaining_indices = [idx for idx in sample_indices if idx not in done_indices]
            print(f"===== Done {len(done_indices)} | Remaining {len(remaining_indices)} | Total {len(sample_indices)} ======")

            # Evenly assign remaining indices to each rank in a round-robin way.
            assigned_indices_all = [[] for _ in range(world_size)]
            for j, idx in enumerate(remaining_indices):
                assigned_indices_all[j % world_size].append(idx)
        else:
            assigned_indices_all = None

        # Synchronize the assignment to all ranks if distributed is initialized.
        if dist.is_available() and dist.is_initialized():
            obj_list = [assigned_indices_all]
            dist.broadcast_object_list(obj_list, src=0)
            assigned_indices_all = obj_list[0]

        if assigned_indices_all is not None:
            assigned_indices = assigned_indices_all[rank]

    model = infer_data(
        model=model,
        model_name=model_name,
        work_dir=work_dir,
        dataset=dataset,
        out_file=out_file,
        verbose=verbose,
        api_nproc=api_nproc,
        use_vllm=use_vllm,
        assigned_indices=assigned_indices)

    if world_size > 1:
        dist.barrier()

    if rank == 0:
        data_all = {}
        for i in range(world_size):
            data_all.update(load(tmpl.format(i)))

        meta = dataset.data
        if dataset_name == 'MMBench-Video' and getattr(dataset, 'pack', False):
            meta, vstats = dataset.load_pack_answers(data_all)
            print(f'Statitics of Pack Video Inference: {vstats}')
        else:
            for x in meta['index']:
                assert x in data_all
            meta['prediction'] = [str(data_all[x]) for x in meta['index']]
            if 'image' in meta:
                meta.pop('image')

        dump(meta, result_file)
        for i in range(world_size):
            os.remove(tmpl.format(i))
    return model
