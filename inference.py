# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Audio-to-text-and-speech adaptation of vllm/end2end.py.

Source: https://github.com/vllm-project/vllm-omni/blob/main/examples/offline_inference/qwen3_omni/end2end.py

Kept: official audio prompt format, Omni API, stage sampling and output parsing.
Changed: conversational task, one local WAV, stable output names, and a tested
4x48GB deployment overlay. No web server, Transformers backend or scoring.

Run from this directory:
  conda activate vllm
  CUDA_VISIBLE_DEVICES=4,5,6,7 python -u inference.py \
      --audio-path assets/input.wav --output-dir assets/output

The default deployment uses logical GPUs 0,1 for Thinker (TP=2), GPU 2 for
Talker, and GPU 3 for Code2Wav. Use --deploy-config for a different layout.
"""
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import time

HERE = Path(__file__).resolve().parent
SEED = 42
DEFAULT_SYSTEM = (
    'You are Qwen, a virtual human developed by the Qwen Team, Alibaba '
    'Group, capable of perceiving auditory and visual inputs, as well as '
    'generating text and speech.'
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', default='/data/hdd0/xiaobin.rong/pretrained/Qwen3-Omni-30B-A3B-Instruct')
    parser.add_argument('--audio-path', '--input', dest='audio_path', type=Path,
                        default=HERE / 'assets/input.wav')
    parser.add_argument('--output-dir', type=Path, default=HERE / 'assets/output')
    parser.add_argument('--question', default=(
        'Please answer the user in the audio directly, in the same language as '
        'the user. Keep the reply brief and conversational.'))
    parser.add_argument('--max-tokens', type=int, default=512,
                        help='Thinker output budget; official demo uses 1200')
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--deploy-config', type=Path,
                        help='Optional complete deployment YAML; relative base_config is resolved by Omni')
    parser.add_argument('--stage-init-timeout', type=int, default=600)
    parser.add_argument('--nccl-safe-mode', action=argparse.BooleanOptionalAction, default=True,
                        help='Default on: disable NCCL P2P/IB for this machine; may reduce communication speed')
    parser.add_argument('--check', action='store_true',
                        help='Validate paths, GPU count and deployment without loading model weights')
    return parser.parse_args()


def get_audio_query(audio_path, question):
    """Official get_audio_query format, with a conversational question."""
    import numpy as np
    from vllm.multimodal.media.audio import load_audio

    audio_signal, sr = load_audio(str(audio_path), sr=16000)
    prompt = (
        f'<|im_start|>system\n{DEFAULT_SYSTEM}<|im_end|>\n'
        '<|im_start|>user\n<|audio_start|><|audio_pad|><|audio_end|>'
        f'{question}<|im_end|>\n<|im_start|>assistant\n'
    )
    return {
        'prompt': prompt,
        'multi_modal_data': {'audio': (audio_signal.astype(np.float32), sr)},
        'modalities': ['text', 'audio'],
    }


@contextmanager
def deployment_config(custom_config):
    """Use installed official defaults plus only the tested local overrides."""
    if custom_config is not None:
        yield custom_config.resolve(strict=True)
        return

    import yaml
    import vllm_omni
    from vllm_omni.config.stage_config import resolve_deploy_yaml

    base = Path(vllm_omni.__file__).parent / 'deploy/qwen3_omni_moe.yaml'
    config = resolve_deploy_yaml(base)
    config['async_chunk'] = False
    overrides = {
        0: dict(devices='0,1', tensor_parallel_size=2, gpu_memory_utilization=0.85,
                max_num_seqs=1, max_model_len=4096, max_num_batched_tokens=4096,
                enable_prefix_caching=False, enforce_eager=True,
                disable_custom_all_reduce=True,
                # In the installed version, disabling both visual modalities
                # leaves DeepStack buffers on meta. Keep image=1; no image is sent.
                limit_mm_per_prompt={'audio': 1, 'image': 1, 'video': 0}),
        1: dict(devices='2', tensor_parallel_size=1, gpu_memory_utilization=0.80,
                max_num_seqs=1, enforce_eager=True),
        2: dict(devices='3', tensor_parallel_size=1, gpu_memory_utilization=0.80,
                max_num_seqs=1),
    }
    for stage in config['stages']:
        stage.update(overrides[stage['stage_id']])
    with TemporaryDirectory(prefix='qwen3-omni-') as directory:
        path = Path(directory) / 'deploy.yaml'
        path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding='utf-8')
        yield path


def main():
    args = parse_args()
    source = args.audio_path.resolve(strict=True)
    output_dir = args.output_dir.resolve()
    if args.max_tokens < 1:
        raise ValueError('--max-tokens must be positive')
    os.environ.setdefault('VLLM_WORKER_MULTIPROC_METHOD', 'spawn')
    if args.nccl_safe_mode:
        os.environ['NCCL_P2P_DISABLE'] = '1'
        os.environ['NCCL_IB_DISABLE'] = '1'

    import numpy as np
    import soundfile as sf
    import torch
    import yaml
    from vllm import SamplingParams
    from vllm_omni.entrypoints.omni import Omni
    from vllm_omni.config.stage_config import resolve_deploy_yaml

    if args.deploy_config is None and torch.cuda.device_count() < 4:
        raise RuntimeError('Default deployment requires four visible GPUs. Set CUDA_VISIBLE_DEVICES or --deploy-config.')
    request = get_audio_query(source, args.question)
    output_dir.mkdir(parents=True, exist_ok=True)
    with deployment_config(args.deploy_config) as config_path:
        effective = resolve_deploy_yaml(config_path)
        (output_dir / 'resolved_deploy.yaml').write_text(
            yaml.safe_dump(effective, allow_unicode=True), encoding='utf-8')
        if args.check:
            print(f'Paths, deployment and audio preparation OK. GPUs: {torch.cuda.device_count()}')
            print(f'Configuration: {output_dir / "resolved_deploy.yaml"}')
            return
        print('Loading Thinker (TP=2), Talker and Code2Wav...' if args.deploy_config is None
              else 'Loading custom deployment...', flush=True)
        started = time.perf_counter()
        omni = Omni(model=args.model, deploy_config=str(config_path),
                    stage_init_timeout=args.stage_init_timeout)
        load_seconds = time.perf_counter() - started
        try:
            if omni.num_stages != 3:
                raise RuntimeError('Text and speech inference requires the full three-stage deployment')
            # Same three-stage sampling settings as official end2end.py;
            # only Thinker max_tokens is exposed for short dialogue responses.
            sampling_params_list = [
                SamplingParams(temperature=0.9, top_p=0.9, top_k=-1,
                               max_tokens=args.max_tokens, repetition_penalty=1.05,
                               logit_bias={}, seed=args.seed),
                SamplingParams(temperature=0.9, top_k=50, max_tokens=4096,
                               seed=args.seed, detokenize=False,
                               repetition_penalty=1.05, stop_token_ids=[2150]),
                SamplingParams(temperature=0.0, top_p=1.0, top_k=-1,
                               max_tokens=4096 * 16, seed=args.seed,
                               detokenize=True, repetition_penalty=1.1),
            ]
            print(f'Loaded in {load_seconds:.2f}s. Generating text and speech...', flush=True)
            started = time.perf_counter()
            text_output = ''
            audio_chunks = []
            # Official generator mode exposes text before audio is complete.
            for stage_outputs in omni.generate([request], sampling_params_list, py_generator=True):
                if stage_outputs.final_output_type == 'text':
                    text_output = stage_outputs.outputs[0].text
                    (output_dir / 'response.txt').write_text(text_output, encoding='utf-8')
                    print(text_output, flush=True)
                elif stage_outputs.final_output_type == 'audio':
                    audio_tensor = stage_outputs.outputs[0].multimodal_output['audio']
                    chunks = audio_tensor if isinstance(audio_tensor, list) else [audio_tensor]
                    for chunk in chunks:
                        if isinstance(chunk, torch.Tensor):
                            chunk = chunk.float().detach().cpu().numpy()
                        audio_chunks.append(np.asarray(chunk, dtype=np.float32).reshape(-1))
            generation_seconds = time.perf_counter() - started
            if not audio_chunks or not any(chunk.size for chunk in audio_chunks):
                raise RuntimeError('No audio output returned; check Talker/Code2Wav logs')
            audio = np.concatenate(audio_chunks)
            if not np.isfinite(audio).all():
                raise RuntimeError('Audio contains non-finite samples')
            wav_path = output_dir / 'response.wav'
            sf.write(str(wav_path), audio, samplerate=24000, format='WAV', subtype='PCM_16')
            result = dict(input=str(source), model=args.model, response_text=text_output,
                          audio=str(wav_path), sample_rate=24000,
                          load_seconds=load_seconds, generation_seconds=generation_seconds,
                          question=args.question, seed=args.seed, max_tokens=args.max_tokens,
                          nccl_safe_mode=args.nccl_safe_mode)
            (output_dir / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
            print(f'Generation: {generation_seconds:.2f}s; audio: {wav_path}', flush=True)
        finally:
            omni.close()


if __name__ == '__main__':
    main()
