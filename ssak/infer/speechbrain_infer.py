#!/usr/bin/env python3

import huggingface_hub
import speechbrain as sb

from ssak.utils.dataset import to_audio_batches
from ssak.utils.debug import plot_logits
from ssak.utils.env import auto_device  # handles option --gpus
from ssak.utils.misc import get_cache_dir, hashmd5
from ssak.utils.monitoring import logger, tic, toc, vram_peak
from ssak.utils.yaml_utils import make_yaml_overrides

if sb.__version__ >= "1.0.0":
    from speechbrain.inference.ASR import EncoderASR as SpeechBrainEncoderASR
    from speechbrain.inference.ASR import EncoderDecoderASR as SpeechBrainEncoderDecoderASR
    from speechbrain.lobes.models.huggingface_transformers.whisper import Whisper as SpeechBrainWhisper
else:
    from speechbrain.lobes.models.huggingface_whisper import HuggingFaceWhisper as SpeechBrainWhisper
    from speechbrain.pretrained.interfaces import EncoderASR as SpeechBrainEncoderASR
    from speechbrain.pretrained.interfaces import EncoderDecoderASR as SpeechBrainEncoderDecoderASR

_speechbrain_classes = (SpeechBrainEncoderASR, SpeechBrainEncoderDecoderASR)

import json
import math
import os
import tempfile

import pyctcdecode
import requests
import torch
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
import transformers


def speechbrain_infer(
    model,
    audios,
    batch_size=1,
    device=None,
    language="fr",
    arpa_path=None,
    alpha=0.5,
    beta=1.0,
    sort_by_len=False,
    output_ids=False,
    log_memtime=False,
    plot_logprobas=False,
):
    """
    Transcribe audio(s) with speechbrain model

    Args:
        model: SpeechBrain model or a path to the model
        audios:
            Audio file path(s), or Kaldi folder(s), or Audio waveform(s)
        batch_size: int
            Batch size (default 1).
        device: str
            Device to use (default "cuda:0" if GPU available else "cpu").
            Can be: "cpu", "cuda:0", "cuda:1", etc.
        arpa_path: str
            Path to arpa file for decoding with Language Model.
        alpha: float
            Language Model weight.
        beta: float
            Word insertion penalty.
        sort_by_len: bool
            Sort audio by length before batching (longest audio first).
        log_memtime: bool
            If True, print timing and memory usage information.
    """
    if batch_size == 0:
        batch_size = 1

    if isinstance(model, str):
        model = speechbrain_load_model(model, device=device)

    assert isinstance(model, _speechbrain_classes + (SpeechBrainWhisper,)), f"model must be a SpeechBrain model or a path to the model (got {type(model)})"

    sample_rate = model.audio_normalizer.sample_rate if hasattr(model, "audio_normalizer") else model.sampling_rate

    batches = to_audio_batches(
        audios,
        return_format="torch",
        sample_rate=sample_rate,
        batch_size=batch_size,
        sort_by_len=sort_by_len,
        output_ids=output_ids,
    )

    if arpa_path is None:
        # Compute best predictions
        tic()
        for batch in batches:
            if output_ids:
                ids = [b[1] for b in batch]
                batch = [b[0] for b in batch]
            pred = speechbrain_transcribe_batch(model, batch, language=language, plot_logprobas=plot_logprobas)
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

    else:
        assert os.path.isfile(arpa_path), f"Arpa file {arpa_path} not found"
        if isinstance(model, SpeechBrainEncoderDecoderASR):
            raise NotImplementedError("Language model decoding is not implemented for EncoderDecoderASR models (which do not provide an interface to access log-probabilities)")

        # Compute framewise log probas
        tic()
        logits = []
        for batch in batches:
            if output_ids:
                ids = [b[1] for b in batch]
                batch = [b[0] for b in batch]
            _, log_probas = speechbrain_compute_logits(model, batch, plot_logprobas=plot_logprobas)
            if output_ids:
                logits.append((ids, log_probas))
            else:
                logits.append(log_probas)
            if log_memtime:
                vram_peak()
        if log_memtime:
            toc("apply network", log_mem_usage=True)

        tokenizer = model.tokenizer
        processor = speechbrain_decoder_with_lm(tokenizer, arpa_path, alpha=alpha, beta=beta)

        # Apply language model
        tic()
        num_outputs = len(get_tokenizer_vocab(tokenizer, with_delimiters=True)[0])
        for l in logits:
            if output_ids:
                ids, l = l
            predictions = processor.batch_decode(conform_torch_logit(l, num_outputs).cpu().numpy()).text
            if output_ids:
                for id, p in zip(ids, predictions):
                    yield (id, p)
            else:
                for p in predictions:
                    yield p

        if log_memtime:
            toc("apply language model", log_mem_usage=True)


MAX_LEN = 2240400


def model_cannot_compute_logits(model):
    res = isinstance(model, SpeechBrainWhisper)
    if res:
        logger.warning(f"Model of type {type(model)} cannot be used to compute logits. And memory overflow might occur when processing a long audio")
    return res


def speechbrain_transcribe_batch(model, audios, max_duration=MAX_LEN, plot_logprobas=False, language="fr"):
    if (plot_logprobas or max([len(a) for a in audios]) > max_duration) and not model_cannot_compute_logits(model):
        reco, logits = speechbrain_compute_logits(model, audios, max_duration=max_duration, plot_logprobas=plot_logprobas, compute_predictions=True)
    else:
        device = speechbrain_get_device(model)
        batch, wav_lens = pack_sequences(audios, device=device)
        if isinstance(model, SpeechBrainWhisper):
            if not hasattr(model, "input_tokens") or model.task != "transcribe_" + language:
                # Note: there might be another way of doing this using
                #   model.tokenizer.set_prefix_tokens("french", "transcribe", False)
                ids = model.tokenizer.additional_special_tokens_ids
                specials = model.tokenizer.special_tokens_map["additional_special_tokens"]
                bos_index = model.tokenizer.encode("")[0]  # WTF: model.tokenizer.bos_token_id is wrong
                eos_index = model.tokenizer.encode("")[-1]  # same as model.tokenizer.eos_token_id
                language_token = f"<|{language}|>"
                if language_token not in specials:
                    raise ValueError(f"Language '{language}' not supported by the model. Please use one of {sorted([t[2:-2] for t in specials if len(t) == 6])}")
                language_index = ids[specials.index(language_token)]
                transcribe_index = ids[specials.index("<|transcribe|>")]
                dostart_index = ids[specials.index("<|notimestamps|>")]
                model.input_tokens = [bos_index, language_index, transcribe_index, dostart_index]
                model.bos_index = bos_index
                model.eos_index = eos_index
                model.task = "transcribe_" + language
            decoder_input_ids = torch.tensor([model.input_tokens] * batch.shape[0], dtype=torch.int).to(device)
            model.encoder_only = True
            hidden = model.forward(batch, decoder_input_ids=decoder_input_ids)

            decoder = sb.decoders.seq2seq.S2SWhisperGreedySearch(
                model,
                bos_index=model.bos_index,
                eos_index=model.eos_index,
                min_decode_ratio=0,
                max_decode_ratio=0.1,
            )
            decoder.set_decoder_input_tokens(model.input_tokens)
            output_ids, _ = decoder(hidden, wav_lens)

            # output_ids = torch.argmax(logprobs,dim=-1)
            # # Remove the 3 first tokens
            # print(output_ids.shape)
            # output_ids = output_ids[:, 3:]
            # print(output_ids.shape)
            reco = model.tokenizer.batch_decode(output_ids)
        else:
            reco = model.transcribe_batch(batch, wav_lens)[0]
    return reco


def speechbrain_compute_logits(model, audios, max_duration=MAX_LEN, plot_logprobas=False, compute_predictions=False):
    if isinstance(model, SpeechBrainWhisper):
        raise NotImplementedError("Computing log probability is not implemented for SpeechBrainWhisper models")
    if not isinstance(audios, list):
        audios = [audios]
        reco, log_probas = speechbrain_compute_logits(
            model,
            audios,
            max_duration=max_duration,
            plot_logprobas=plot_logprobas,
            compute_predictions=compute_predictions,
        )
        return reco[0], log_probas[0]
    assert len(audios) > 0, "audios must be a non-empty list"
    if not isinstance(audios[0], torch.Tensor):
        audios = [torch.from_numpy(a) for a in audios]
    blank_id = model.decoding_function.keywords.get("blank_id", 0)
    if max([len(a) for a in audios]) > max_duration:
        # Split audios into chunks of max_duration
        batch_size = len(audios)
        chunks = []
        i_audio = []
        for a in audios:
            chunks.extend([a[i : min(i + max_duration, len(a))] for i in range(0, len(a), max_duration)])
            i_audio.append(len(chunks))
        log_probas = [[] for i in range(len(audios))]
        for i in range(0, len(chunks), batch_size):
            chunk = chunks[i : min(i + batch_size, len(chunks))]
            _, log_probas_tmp = speechbrain_compute_logits(model, chunk)
            for j in range(i, i + len(chunk)):
                k = 0
                while j >= i_audio[k]:
                    k += 1
                log_probas[k].append(log_probas_tmp[j - i])
        log_probas = [torch.cat(p, dim=0) for p in log_probas]
        log_probas, wav_lens = pack_sequences(log_probas, device=speechbrain_get_device(model))
    else:
        batch, wav_lens = pack_sequences(audios, device=speechbrain_get_device(model))
        log_probas = model.forward(batch, wav_lens)  # Same as encode_batch for EncoderASR, but it would be same as transcribe_batch for EncoderDecoderASR (which returns strings and token indices)
    # Set log_probas outside of input signal to -inf (except from the blank)
    for i in range(len(log_probas)):
        wav_len = wav_lens[i]
        if wav_len < 1.0:
            wav_len = math.ceil(wav_len * log_probas.shape[1])
            log_probas[i, wav_len:] = -700
            log_probas[i, wav_len:, blank_id] = 0
    if plot_logprobas:
        # for debug
        plot_logits(log_probas, model)
    if compute_predictions:
        indices = sb.decoders.ctc_greedy_decode(log_probas, wav_lens, blank_id=blank_id)
        reco = model.tokenizer.decode(indices)
    else:
        reco = [""] * len(audios)
    return reco, log_probas


def speechbrain_decoder_with_lm(tokenizer, arpa_file, alpha=0.5, beta=1.0):
    """
    tokenizer: tokenizer from speechbrain
    arpa_file: path to arpa file
    alpha: language model weight
    beta: word insertion penalty

    return a processor of type Wav2Vec2ProcessorWithLM to be used as "processor.batch_decode(log_probas.numpy()).text"
    """
    labels = get_tokenizer_vocab(tokenizer, with_delimiters=True)[0]
    vocab = dict((c, i) for i, c in enumerate(labels))
    vocab_file = os.path.join(tempfile.gettempdir(), "vocab.json")
    json.dump(vocab, open(vocab_file, "w"), ensure_ascii=False)
    tokenizer_hf = transformers.Wav2Vec2CTCTokenizer(
        vocab_file,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<pad>",
        pad_token="<pad>",
        word_delimiter_token=" ",
        replace_word_delimiter_char=" ",
        do_lower_case=False,
    )
    decoder = pyctcdecode.build_ctcdecoder(
        labels=labels,
        kenlm_model_path=arpa_file,
        alpha=alpha,
        beta=beta,
    )

    processor = transformers.Wav2Vec2ProcessorWithLM(feature_extractor=transformers.Wav2Vec2FeatureExtractor(), tokenizer=tokenizer_hf, decoder=decoder)
    return processor


def get_tokenizer_vocab(tokenizer, with_delimiters=False, try_to_use="<pad>"):
    from pyctcdecode.alphabet import UNK_TOKEN  # ⁇

    _labels_trans_no_unk = {
        "": " ",
        "<unk>": try_to_use,
        f" {UNK_TOKEN} ": try_to_use,
        UNK_TOKEN: try_to_use,
        "<pad>": try_to_use,
    }
    labels00 = tokenizer.decode([[i] for i in range(tokenizer.get_piece_size())])
    labels0 = [tokenizer.id_to_piece(i) for i in range(tokenizer.get_piece_size())]
    labels = [_labels_trans_no_unk.get(i, i) for i in labels0]
    # Use lower case unless it introduces a conflict
    labels_ = [l.lower() for l in labels]
    if len(set(labels_)) == len(labels_):
        labels = labels_
    blank_id = labels.index(try_to_use)
    if len(set(labels)) != len(labels):
        print("Unexpected situation: some tokens are duplicated")
        import pdb

        pdb.set_trace()
    if with_delimiters:
        if "<s>" not in labels:
            labels.append("<s>")
        if "</s>" not in labels:
            labels.append("</s>")
    return labels, blank_id


def pack_sequences(tensors, device="cpu"):
    if len(tensors) == 1:
        return tensors[0].unsqueeze(0).to(device), torch.Tensor([1.0]).to(device)
    tensor = rnn_utils.pad_sequence(tensors, batch_first=True)
    wav_lens = [len(x) for x in tensors]
    maxwav_lens = max(wav_lens)
    wav_lens = torch.Tensor([l / maxwav_lens for l in wav_lens])
    # TODO: clarify what's going on with wav_lens
    # wav_lens = torch.Tensor([1. for l in wav_lens])
    return tensor.to(device), wav_lens.to(device)


def conform_torch_logit(x, num_outputs):
    n = x.shape[-1]
    if n < num_outputs:
        return F.pad(input=x, pad=(0, num_outputs - n), mode="constant", value=-1000)
    if n > num_outputs:
        return x[:, :, :num_outputs]
    return x


def speechbrain_cachedir(source):
    if os.path.isdir(source):
        return get_cache_dir("speechbrain/" + hashmd5(os.path.realpath(source)))
    else:
        cache_dir = get_cache_dir("speechbrain")
        cache_dir = os.path.join(cache_dir, os.path.basename(source))
        return cache_dir


def speechbrain_load_model(source, device=None):
    if device is None:
        device = auto_device()

    cache_dir = speechbrain_cachedir(source)

    if os.path.isdir(source):
        yaml_file = os.path.join(source, "hyperparams.yaml")
        assert os.path.isfile(yaml_file), f"Hyperparams file {yaml_file} not found"
    else:
        try:
            yaml_file = huggingface_hub.hf_hub_download(repo_id=source, filename="hyperparams.yaml", cache_dir=get_cache_dir("huggingface/hub"))
        except requests.exceptions.HTTPError:
            yaml_file = None

    # Using save_path=None started to fail in speechbrain 0.5.14
    save_path = None if sb.__version__ <= "0.5.13" else get_cache_dir("huggingface/hub")
    overrides = make_yaml_overrides(yaml_file, {"save_path": save_path})
    try:
        model = SpeechBrainEncoderASR.from_hparams(source=source, run_opts={"device": device}, savedir=cache_dir, overrides=overrides)
    except ValueError as err1:
        try:
            model = SpeechBrainEncoderDecoderASR.from_hparams(source=source, run_opts={"device": device}, savedir=cache_dir, overrides=overrides)
        except ValueError as err2:
            try:
                model = SpeechBrainWhisper(source, save_path=get_cache_dir("huggingface/hub"), freeze=True)
                model = model.to(device)
            except Exception as err3:
                raise RuntimeError(f"Cannot load model from {source}:\n==={err3}\n===\n{err2}\n===\n{err1}")
    model.train(False)
    model.requires_grad_(False)
    return model


def speechbrain_get_device(model):
    if hasattr(model, "device"):
        return model.device
    if hasattr(model, "model") and hasattr(model.model, "device"):
        return model.model.device
    raise NotImplementedError(f"Cannot find device for model {type(model)}")


def cli():
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Transcribe audio(s) using a model from Speechbrain (wav2vec2, Whisper...)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("data", help="Path to data (audio file(s) or kaldi folder(s))", nargs="+")
    parser.add_argument(
        "--model",
        help="Path to trained folder, or name of a pretrained model",
        default="speechbrain/asr-wav2vec2-commonvoice-fr",
    )
    parser.add_argument(
        "--language",
        help="Code of language. Relevant only for multi-lingual models such as openai/whisper-XXX",
        default="fr",
    )
    parser.add_argument("--arpa", help="Path to a n-gram language model", default=None)
    parser.add_argument("--output", help="Output path (will print on stdout by default)", default=None)
    parser.add_argument("--use_ids", help="Whether to print the id before result", default=False, action="store_true")
    parser.add_argument("--batch_size", help="Maximum batch size", type=int, default=32)
    parser.add_argument("--gpus", help="List of GPU index to use (starting from 0)", default=None)
    parser.add_argument("--sort_by_len", help="Sort by (decreasing) length", default=False, action="store_true")
    parser.add_argument("--enable_logs", help="Enable logs about time", default=False, action="store_true")
    parser.add_argument("--plot_logprobas", help="Plot logits", default=False, action="store_true")
    args = parser.parse_args()

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

    for reco in speechbrain_infer(
        args.model,
        args.data,
        batch_size=args.batch_size,
        sort_by_len=args.sort_by_len,
        output_ids=args.use_ids,
        language=args.language,
        arpa_path=args.arpa,
        log_memtime=args.enable_logs,
        plot_logprobas=args.plot_logprobas,
    ):
        if isinstance(reco, str):
            print(reco, file=args.output)
        else:
            print(*reco, file=args.output)
        args.output.flush()


if __name__ == "__main__":
    cli()
