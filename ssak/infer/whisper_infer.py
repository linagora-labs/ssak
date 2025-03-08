#!/usr/bin/env python3

import re

import numpy as np
import torch
import whisper

from ssak.utils.dataset import to_audio_batches
from ssak.utils.env import *  # handles option --gpus
from ssak.utils.misc import get_cache_dir
from ssak.utils.monitoring import tic, toc, vram_peak


def whisper_infer(
    model,
    audios,
    # batch_size = 1,
    device=None,
    language="fr",
    no_speech_threshold=0.6,
    logprob_threshold=-1.0,
    compression_ratio_threshold=2.4,
    beam_size=None,
    temperature=0.0,
    best_of=None,
    condition_on_previous_text=False,
    prompt=None,
    sort_by_len=False,
    output_ids=False,
    log_memtime=False,
    seed=1234,
):
    """
    Transcribe audio(s) with Whisper model

    Args:
        model: Whisper model or a path to the model
        audios:
            Audio file path(s), or Kaldi folder(s), or Audio waveform(s)
        device: str
            Device to use (default "cuda:0" if GPU available else "cpu").
            Can be: "cpu", "cuda:0", "cuda:1", etc.
        sort_by_len: bool
            Sort audio by length before batching (longest audio first).
        log_memtime: bool
            If True, print timing and memory usage information.
    """

    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # if batch_size == 0:
    #     batch_size = 1

    if device is None:
        device = auto_device()

    if isinstance(model, str):
        model = load_model(model, device=device, download_root=None)

    batches = to_audio_batches(
        audios,
        return_format="torch",
        sample_rate=whisper.audio.SAMPLE_RATE,
        batch_size=1,
        sort_by_len=sort_by_len,
        output_ids=output_ids,
    )

    fp16 = model.device != torch.device("cpu")

    # Compute best predictions
    tic()
    for batch in batches:
        if output_ids:
            ids = [b[1] for b in batch]
            batch = [b[0] for b in batch]

        pred = []
        for audio in batch:
            res = model.transcribe(
                audio_minimum_padding(audio),
                language=language,
                fp16=fp16,
                beam_size=beam_size,
                temperature=temperature,
                best_of=best_of,
                condition_on_previous_text=condition_on_previous_text,
                no_speech_threshold=no_speech_threshold,
                logprob_threshold=logprob_threshold,
                compression_ratio_threshold=compression_ratio_threshold,
                without_timestamps=False,
                initial_prompt=prompt if prompt else None,
            )
            # Note: other interesting keys of res are:
            #   "segments": {"start", "end", "seek", "text", "tokens", "temperature", "avg_logprob", "no_speech_prob", "compression_ratio"}
            #   - "avg_logprob" : Average log-probability of tokens
            #   - "no_speech_prob" : Probability of no speech activity
            pred.append(res["text"].strip())

        if output_ids:
            for id, p in zip(ids, pred):
                yield (id, p)
        else:
            for p in pred:
                yield p
        if log_memtime:
            vram_peak()
    if log_memtime:
        toc("apply network", log_mem_usage=True)


def audio_minimum_padding(audio):
    if audio.shape[-1] <= 200:
        return whisper.pad_or_trim(audio, 201)
    return audio


def load_model(
    name: str,
    device: str = None,
    download_root: str = None,
    in_memory: bool = False,
):
    extension = os.path.splitext(name)[-1] if os.path.isfile(name) else None

    if name in whisper.available_models() or extension == ".pt":
        if download_root is None:
            download_root = get_cache_dir("whisper")
        return whisper.load_model(name, device=device, download_root=download_root, in_memory=in_memory)

    # Otherwise, assume transformers
    if download_root is None:
        download_root = get_cache_dir("huggingface/hub")
    peft_folder = None
    if extension in [".ckpt", ".bin"]:
        model_path = name
    else:
        # Search for the cached file (download if necessary)
        if os.path.isdir(name):
            for root, _, files in os.walk(name):
                if "adapter_config.json" in files:
                    peft_folder = root
                    break
        try:
            import transformers
        except ImportError:
            raise ImportError(f"If you are trying to download a HuggingFace model with {name}, please install first the transformers library")
        from transformers.utils import cached_file

        try:
            model_path = cached_file(name, "pytorch_model.bin", cache_dir=download_root, use_auth_token=None, revision=None)
        except Exception as e:
            try:
                if isinstance(e, OSError):
                    model_path = cached_file(name, "whisper.ckpt", cache_dir=download_root, use_auth_token=None, revision=None)
                else:
                    raise e
            except:
                if peft_folder is None:
                    raise RuntimeError(f"Original error: {e}\nCould not find model {name} from HuggingFace nor local folders.")

    # Load HF Model
    if peft_folder is not None:
        import transformers
        from peft import PeftConfig, PeftModel

        peft_config = PeftConfig.from_pretrained(peft_folder)
        base_model = peft_config.base_model_name_or_path

        # quantization_config = transformers.BitsAndBytesConfig(llm_int8_enable_fp32_cpu_offload=True)
        # model = transformers.WhisperForConditionalGeneration.from_pretrained(base_model,
        #     load_in_8bit=True,
        #     device_map="auto",
        #     quantization_config=quantization_config,
        # )
        model = transformers.WhisperForConditionalGeneration.from_pretrained(base_model)

        model = PeftModel.from_pretrained(model, peft_folder)
        # model = model.float().get_base_model() # back to transformers (not needed...)
        hf_state_dict = model.state_dict()
        del model
    else:
        hf_state_dict = torch.load(model_path, map_location="cpu")

    # Rename layers
    for key in list(hf_state_dict.keys()):
        new_key = hf_to_whisper_states(key)
        if new_key is None:
            hf_state_dict.pop(key)
        elif new_key != key:
            hf_state_dict[new_key] = hf_state_dict.pop(key)

    # Init Whisper Model and replace model weights
    dims = whisper.model.ModelDimensions(**states_to_dim(hf_state_dict))
    if "proj_out.weight" in hf_state_dict:
        hf_state_dict["decoder.proj_out.weight"] = hf_state_dict.pop("proj_out.weight")
        print("WARNING: Using untied projection layer")
        whisper_model = WhisperUntied(dims)
    else:
        whisper_model = whisper.model.Whisper(dims)
    whisper_model.load_state_dict(hf_state_dict)
    del hf_state_dict
    whisper_model = whisper_model.to(device)
    return whisper_model


# Credit: https://github.com/openai/whisper/discussions/830
def hf_to_whisper_states(text):
    # From Speechbrain
    if text == "_mel_filters":
        return None

    # From PEFT
    if "default" in text:
        # print(f"WARNING: Ignoring {text}")
        return None
    if text.startswith("base_model.model."):
        text = text[len("base_model.model.") :]

    text = re.sub(".layers.", ".blocks.", text)
    text = re.sub(".self_attn.", ".attn.", text)
    text = re.sub(".q_proj.", ".query.", text)
    text = re.sub(".k_proj.", ".key.", text)
    text = re.sub(".v_proj.", ".value.", text)
    text = re.sub(".out_proj.", ".out.", text)
    text = re.sub(".fc1.", ".mlp.0.", text)
    text = re.sub(".fc2.", ".mlp.2.", text)
    text = re.sub(".fc3.", ".mlp.3.", text)
    text = re.sub(".fc3.", ".mlp.3.", text)
    text = re.sub(".encoder_attn.", ".cross_attn.", text)
    text = re.sub(".cross_attn.ln.", ".cross_attn_ln.", text)
    text = re.sub(".embed_positions.weight", ".positional_embedding", text)
    text = re.sub(".embed_tokens.", ".token_embedding.", text)
    text = re.sub("model.", "", text)
    text = re.sub("attn.layer_norm.", "attn_ln.", text)
    text = re.sub(".final_layer_norm.", ".mlp_ln.", text)
    text = re.sub("encoder.layer_norm.", "encoder.ln_post.", text)
    text = re.sub("decoder.layer_norm.", "decoder.ln.", text)
    return text


def states_to_dim(state_dict):
    n_audio_state = len(state_dict["encoder.ln_post.bias"])
    n_text_state = len(state_dict["decoder.ln.bias"])
    return {
        "n_mels": state_dict["encoder.conv1.weight"].shape[1],  # 80
        "n_vocab": state_dict["decoder.token_embedding.weight"].shape[0],  # 51864 / 51865
        "n_audio_ctx": state_dict["encoder.positional_embedding"].shape[0],  # 1500
        "n_audio_state": n_audio_state,  # 384 / 512 / 768 / 1024 / 1280
        "n_audio_head": n_audio_state // 64,  # 6 / 8 / 12 / 16 / 20
        "n_audio_layer": len(set([".".join(k.split(".")[:3]) for k in state_dict.keys() if "encoder.blocks." in k])),  # 4 / 6 / 12 / 24 / 32
        "n_text_ctx": state_dict["decoder.positional_embedding"].shape[0],  # 448
        "n_text_state": n_text_state,  # 384 / 512 / 768 / 1024 / 1280
        "n_text_head": n_text_state // 64,  # 6 / 8 / 12 / 16 / 20
        "n_text_layer": len(set([".".join(k.split(".")[:3]) for k in state_dict.keys() if "decoder.blocks." in k])),  # 4 / 6 / 12 / 24 / 32
    }


class TextDecoderUntied(whisper.model.TextDecoder):
    """
    Same as TextDecoder but with untied weights
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        n_vocab, n_state = self.token_embedding.weight.shape

        self.proj_out = torch.nn.Linear(n_state, n_vocab, bias=False)

    def forward(self, x, xa, kv_cache=None):
        offset = next(iter(kv_cache.values())).shape[1] if kv_cache else 0
        x = self.token_embedding(x) + self.positional_embedding[offset : offset + x.shape[-1]]
        x = x.to(xa.dtype)

        for block in self.blocks:
            x = block(x, xa, mask=self.mask, kv_cache=kv_cache)

        x = self.ln(x)

        # logits = self.proj_out(x).float()
        # logits = (x @ torch.transpose(self.proj_out.weight.to(x.dtype), 0, 1)).float()
        logits = self.proj_out.to(x.dtype)(x).float()

        return logits


class WhisperUntied(whisper.model.Whisper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.decoder = TextDecoderUntied(
            self.dims.n_vocab,
            self.dims.n_text_ctx,
            self.dims.n_text_state,
            self.dims.n_text_head,
            self.dims.n_text_layer,
        )


if __name__ == "__main__":
    import argparse
    import os
    import sys

    from whisper.utils import str2bool

    parser = argparse.ArgumentParser(description="Transcribe audio(s) with whisper", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("data", help="Path to data (audio file(s) or kaldi folder(s))", nargs="+")
    parser.add_argument(
        "--model",
        help=f"name of the Whisper model to use. Examples: {', '.join(whisper.available_models())}",
        default="base",
    )
    parser.add_argument(
        "--language",
        help=f"Language to use. Among : {', '.join(sorted(k+'('+v+')' for k,v in whisper.tokenizer.LANGUAGES.items()))}.",
        default="fr",
    )
    parser.add_argument("--no_speech_threshold", help="Threshold for detecting no speech activity", type=float, default=0.6)
    parser.add_argument(
        "--logprob_threshold",
        help="f the average log probability over sampled tokens is below this value, returns empty string",
        type=float,
        default=-1.0,
    )
    parser.add_argument(
        "--compression_ratio_threshold",
        help="If the gzip compression ratio is above this value, return empty string",
        type=float,
        default=2.4,
    )
    parser.add_argument("--beam_size", help="Size for beam search", type=int, default=None)
    parser.add_argument("--best_of", help="number of candidates when sampling with non-zero temperature", type=int, default=None)
    parser.add_argument("--temperature", default=0.0, help="temperature to use for sampling", type=float)
    parser.add_argument(
        "--temperature_increment_on_fallback",
        default=0.0,
        help="temperature to increase when falling back when the decoding fails to meet either of the thresholds below",
        type=float,
    )
    parser.add_argument(
        "--condition_on_previous_text",
        default=False,
        help="if True, provide the previous output of the model as a prompt for the next window; disabling may make the text inconsistent across windows, but the model becomes less prone to getting stuck in a failure loop",
        type=str2bool,
    )
    parser.add_argument("--prompt", default=None, help="Initial prompt to use", type=str)
    parser.add_argument("--output", help="Output path (will print on stdout by default)", default=None)
    parser.add_argument("--use_ids", help="Whether to print the id before result", default=False, action="store_true")
    parser.add_argument("--gpus", help="List of GPU index to use (starting from 0)", default=None)
    parser.add_argument("--max_threads", help="Maximum thread values (for CPU computation)", default=None, type=int)

    class ActionSetAccurate(argparse.Action):
        def __init__(self, option_strings, dest, nargs=None, **kwargs):
            assert nargs is None
            super().__init__(option_strings, dest, nargs=0, **kwargs)

        def __call__(self, parser, namespace, values, option_string=None):
            namespace.best_of = 5
            namespace.beam_size = 5
            namespace.temperature_increment_on_fallback = 0.2

    parser.add_argument(
        "--accurate",
        help="Shortcut to use the same default option as in Whisper (best_of=5, beam_search=5, temperature_increment_on_fallback=0.2)",
        action=ActionSetAccurate,
    )

    class ActionSetEfficient(argparse.Action):
        def __init__(self, option_strings, dest, nargs=None, **kwargs):
            assert nargs is None
            super().__init__(option_strings, dest, nargs=0, **kwargs)

        def __call__(self, parser, namespace, values, option_string=None):
            namespace.best_of = None
            namespace.beam_size = None
            namespace.temperature_increment_on_fallback = None

    parser.add_argument(
        "--efficient",
        help="Shortcut to disable beam size and options that requires to sample several times, for an efficient decoding",
        action=ActionSetEfficient,
    )

    # parser.add_argument('--batch_size', help="Maximum batch size", type=int, default=32)
    # parser.add_argument('--sort_by_len', help="Sort by (decreasing) length", default=False, action="store_true")
    parser.add_argument("--enable_logs", help="Enable logs about time", default=False, action="store_true")
    args = parser.parse_args()

    temperature = args.temperature
    temperature_increment_on_fallback = args.temperature_increment_on_fallback
    if temperature_increment_on_fallback:
        temperature = tuple(np.arange(temperature, 1.0 + 1e-6, temperature_increment_on_fallback))
    else:
        temperature = [temperature]

    if not args.output:
        args.output = sys.stdout
    elif args.output == "/dev/null":
        # output nothing
        args.output = open(os.devnull, "w")
    else:
        dname = os.path.dirname(args.output)
        if dname and not os.path.isdir(dname):
            os.makedirs(dname)
        args.output = open(args.output, "w")

    if args.max_threads:
        torch.set_num_threads(args.max_threads)

    for reco in whisper_infer(
        args.model,
        args.data,
        language=args.language,
        no_speech_threshold=args.no_speech_threshold,
        logprob_threshold=args.logprob_threshold,
        compression_ratio_threshold=args.compression_ratio_threshold,
        beam_size=args.beam_size,
        temperature=temperature,
        best_of=args.best_of,
        condition_on_previous_text=args.condition_on_previous_text,
        prompt=args.prompt,
        # batch_size = args.batch_size,
        # sort_by_len = args.sort_by_len,
        output_ids=args.use_ids,
        log_memtime=args.enable_logs,
    ):
        if isinstance(reco, str):
            print(reco, file=args.output)
        else:
            print(*reco, file=args.output)
        args.output.flush()
