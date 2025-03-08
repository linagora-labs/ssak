#!/usr/bin/env python3

import os

import pyctcdecode
import torch
import torch.nn.functional as F
import transformers

from ssak.utils.dataset import to_audio_batches
from ssak.utils.env import auto_device  # handles option --gpus
from ssak.utils.monitoring import tic, toc, vram_peak


def transformers_infer(
    source,
    audios,
    batch_size=1,
    device=None,
    language=None,
    arpa_path=None,
    alpha=0.5,
    beta=1.0,
    sort_by_len=False,
    output_ids=False,
    log_memtime=False,
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

    model, processor = transformers_load_model(source, device)

    if language is not None and hasattr(processor, "tokenizer") and hasattr(processor.tokenizer, "vocab"):
        languages = list(processor.tokenizer.vocab.keys())
        assert len(languages) > 0
        if isinstance(processor.tokenizer.vocab[languages[0]], dict):
            if language not in languages:
                candidate_languages = [l for l in languages if l.startswith(language)]
                if len(candidate_languages) == 0:
                    raise ValueError(f"Language {language} not in {languages}")
                elif len(candidate_languages) > 1:
                    raise ValueError(f"Language {language} not in {languages}.\nCould it be one of {candidate_languages}?")
                language = candidate_languages[0]

    sample_rate = processor.feature_extractor.sampling_rate
    device = model.device

    batches = to_audio_batches(
        audios,
        return_format="array",
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
                ids = [x[1] for x in batch]
                batch = [x[0] for x in batch]
            log_probas = transformers_compute_logits(model, processor, batch, device=device, language=language, sample_rate=sample_rate)
            need_argmax = log_probas.dtype not in [torch.int, torch.int32, torch.int64]
            if need_argmax:
                log_probas = torch.argmax(log_probas, dim=-1)
                pred = processor.batch_decode(log_probas)
            else:
                pred = processor.batch_decode(log_probas, skip_special_tokens=True)
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
        tokenizer = processor.tokenizer

        # Compute framewise log probas
        tic()
        logits = []
        for batch in batches:
            if output_ids:
                ids = [x[1] for x in batch]
                batch = [x[0] for x in batch]
            log_probas = transformers_compute_logits(model, processor, batch, device=device, language=language, sample_rate=sample_rate)
            if output_ids:
                logits.append((ids, log_probas))
            else:
                logits.append(log_probas)
            if log_memtime:
                vram_peak()
        if log_memtime:
            toc("apply network", log_mem_usage=True)

        if language:
            tokenizer = to_simple_tokenizer(tokenizer, language)
        decoder = transformers_decoder_with_lm(tokenizer, arpa_path, alpha=alpha, beta=beta)

        # Apply language model
        tic()
        num_outputs = len(tokenizer.get_vocab())
        for l in logits:
            if output_ids:
                ids, l = l
            predictions = decoder.batch_decode(conform_torch_logit(l, num_outputs).numpy()).text
            if output_ids:
                for id, p in zip(ids, predictions):
                    yield (id, p)
            else:
                for p in predictions:
                    yield p
        if log_memtime:
            toc("apply language model", log_mem_usage=True)


WAV2VEC_CLASSES = (transformers.Wav2Vec2ForCTC, transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2Model)


def transformers_load_model(source, device=None):
    if device is None:
        device = auto_device()

    if isinstance(source, str):
        peft_folder = None
        if os.path.isdir(source):
            if "adapter_config.json" in os.listdir(source):
                peft_folder = source
        if peft_folder:
            from peft import PeftConfig, PeftModel

            peft_config = PeftConfig.from_pretrained(peft_folder)
            base_model = peft_config.base_model_name_or_path
            model = auto_model(base_model)
            model = PeftModel.from_pretrained(model, peft_folder).to(device)
        else:
            model = auto_model(source, device)
        if isinstance(model, WAV2VEC_CLASSES):
            processor = transformers.Wav2Vec2Processor.from_pretrained(source)
        else:
            processor = transformers.AutoProcessor.from_pretrained(source)
    elif isinstance(source, (list, tuple)) and len(source) == 2:
        model, processor = source
        assert isinstance(model, WAV2VEC_CLASSES)
        assert isinstance(processor, transformers.Wav2Vec2Processor)
        model = model.to(device)
    else:
        raise NotImplementedError("Only Wav2Vec2ForCTC from a model name or folder is supported for now")

    return model, processor


# Wrapper of "AutoModel" which does produce unintuitive things like WhisperModel that latter fails to decode with "generate"
def auto_model(source, device=None):
    model = transformers.AutoModel.from_pretrained(source)
    if isinstance(model, transformers.models.whisper.modeling_whisper.WhisperModel):
        model = transformers.WhisperForConditionalGeneration.from_pretrained(source)
    if isinstance(model, transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2Model):
        model = transformers.Wav2Vec2ForCTC.from_pretrained(source)
    if device is not None:
        model = model.to(device)
    return model


def conform_torch_logit(x, num_outputs):
    n = x.shape[-1]
    if n < num_outputs:
        return F.pad(input=x, pad=(0, num_outputs - n), mode="constant", value=-1000)
    if n > num_outputs:
        return x[:, :, :num_outputs]
    return x


def transformers_compute_logits(model, processor, batch, device=None, language=None, sample_rate=None, max_duration=2240400):
    use_wav2vec_api = isinstance(model, WAV2VEC_CLASSES)

    if sample_rate == None:
        sample_rate = processor.feature_extractor.sampling_rate
    if device == None:
        device = model.device

    def do_infer_sub(batch, i, j):
        raise NotImplementedError

    if use_wav2vec_api:
        # Wav2Vec style

        if language is not None:
            try:
                processor.tokenizer.set_target_lang(language)
            except ValueError as err:
                if "is not a multi-lingual" in str(err):
                    language = None
                else:
                    raise err
            if language:
                model.load_adapter(language)

        processed_batch = processor(batch, sampling_rate=sample_rate, padding="longest")

        padded_batch = processor.pad(
            processed_batch,
            padding=True,
            max_length=None,
            pad_to_multiple_of=None,
            return_tensors="pt",
        )

        l = padded_batch.input_values.shape[1]

        def get_output(output):
            if hasattr(output, "logits"):
                output = output.logits
            else:
                raise NotImplementedError(f"Cannot find logits in {dir(output)}")
            return output.cpu()

        def do_infer(batch):
            return get_output(model(batch.input_values.to(device), attention_mask=batch.attention_mask.to(device)))

        def do_infer_sub(batch, i, j):
            return get_output(model(batch.input_values[:, i:j].to(device), attention_mask=batch.attention_mask[:, i:j].to(device)))

    else:
        # Whisper style

        # input_features = [{"input_features": feature} for feature in processed_batch.input_features]
        # padded_batch = processor.feature_extractor.pad(input_features, return_tensors="pt")

        padded_batch = processor(batch, return_tensors="pt", sampling_rate=sample_rate)

        l = -1

        forced_decoder_ids = processor.get_decoder_prompt_ids(language=language, task="transcribe")

        def do_infer(batch):
            input_features = batch.input_features.to(device)
            # The length of `decoder_input_ids` equal `prompt_ids` plus special start tokens is 4
            # The combined length of `decoder_input_ids` and `max_new_tokens`=444 is: 448
            # which corresponds to the `max_target_positions` of the Whisper model: 448
            return model.generate(input_features=input_features, forced_decoder_ids=forced_decoder_ids, max_new_tokens=444)

    with torch.no_grad():
        if l > max_duration:
            # Split batch in smaller chunks
            logits = []
            for i in range(0, l, max_duration):
                j = min(i + max_duration, l)
                logits.append(do_infer_sub(padded_batch, i, j))
            logits = torch.cat(logits, dim=1)
        else:
            logits = do_infer(padded_batch)

    return logits


def transformers_decoder_with_lm(tokenizer, arpa_file, alpha=0.5, beta=1.0):
    """
    tokenizer: tokenizer from speechbrain
    arpa_file: path to arpa file
    alpha: language model weight
    beta: word insertion penalty

    return a processor of type Wav2Vec2ProcessorWithLM to be used as "processor.batch_decode(log_probas.numpy()).text"
    """
    vocab_dict = tokenizer.get_vocab()
    labels = [char for char, idx in sorted(vocab_dict.items(), key=lambda x: x[-1])]

    decoder = pyctcdecode.build_ctcdecoder(
        labels=labels,
        kenlm_model_path=arpa_file,
        alpha=alpha,
        beta=beta,
    )
    processor = transformers.Wav2Vec2ProcessorWithLM(feature_extractor=transformers.Wav2Vec2FeatureExtractor(), tokenizer=tokenizer, decoder=decoder)
    return processor


def to_simple_tokenizer(tokenizer, language):
    import json
    import tempfile

    tmpfile = tempfile.mktemp(suffix=".json")
    with open(tmpfile, "w") as f:
        json.dump(tokenizer.vocab[language], f)
    return transformers.Wav2Vec2CTCTokenizer(
        vocab_file=tmpfile,
        bos_token=tokenizer.bos_token,
        eos_token=tokenizer.eos_token,
        unk_token=tokenizer.unk_token,
        pad_token=tokenizer.pad_token,
        word_delimiter_token=tokenizer.word_delimiter_token,
        replace_word_delimiter_char=tokenizer.replace_word_delimiter_char,
        do_lower_case=tokenizer.do_lower_case,
        target_lang=None,
    )


def cli():
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Transcribe audio(s) using a model from HuggingFace's transformers (wav2vec2, Whisper...)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("data", help="Path to data (audio file(s) or kaldi folder(s))", nargs="+")
    parser.add_argument(
        "--model",
        help="Path to trained folder, or name of a pretrained model",
        default="Ilyes/wav2vec2-large-xlsr-53-french",
    )
    parser.add_argument(
        "--language",
        help="Language (if needed by the ASR model, for instance Whisper model or MMS model)",
        default=None,
        type=str,
    )
    parser.add_argument("--arpa", help="Path to a n-gram language model", default=None)
    parser.add_argument("--output", help="Output path (will print on stdout by default)", default=None)
    parser.add_argument("--use_ids", help="Whether to print the id before result", default=False, action="store_true")
    parser.add_argument("--batch_size", help="Maximum batch size", type=int, default=32)
    parser.add_argument("--gpus", help="List of GPU index to use (starting from 0)", default=None)
    parser.add_argument("--sort_by_len", help="Sort by (decreasing) length", default=False, action="store_true")
    parser.add_argument("--enable_logs", help="Enable logs about time", default=False, action="store_true")
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

    for reco in transformers_infer(
        args.model,
        args.data,
        batch_size=args.batch_size,
        sort_by_len=args.sort_by_len,
        output_ids=args.use_ids,
        language=args.language,
        arpa_path=args.arpa,
        log_memtime=args.enable_logs,
    ):
        if isinstance(reco, str):
            print(reco, file=args.output)
        else:
            print(*reco, file=args.output)
        args.output.flush()


if __name__ == "__main__":
    cli()
