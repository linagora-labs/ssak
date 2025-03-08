import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

from dataclasses import dataclass
from typing import Any, Union

import torch
import whisper
from torch.utils.data import DataLoader
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from ssak.utils.dataset import kaldi_folder_to_dataset, process_dataset
from ssak.utils.env import *
from ssak.utils.text_ar import *
from ssak.utils.text_latin import *


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: list[dict[str, Union[list[int], torch.Tensor]]]) -> dict[str, torch.Tensor]:
        # collect the Features in a var
        input_features = [{"input_features": feature["input_features"]} for feature in features]

        # Extract the features from Audio[array] and add some padding and retuurned as a pytorch Tensor
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # collect the labels in a var
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        # Tokenize and add pad to each label to match the max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # check if [bos] token is appended in the previous tokenization step
        # cut the [bos] tokeb here as it's append anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


def load_model(model_path, device):
    PEFT_dir = None
    if os.path.isdir(model_path):
        for root, _, files in os.walk(model_path):
            if "adapter_config.json" in files:
                PEFT_dir = root
                break
    if PEFT_dir is not None:
        # load peft libs:
        from peft import PeftConfig, PeftModel

        peft_config = PeftConfig.from_pretrained(PEFT_dir)
        model = WhisperForConditionalGeneration.from_pretrained(peft_config.base_model_name_or_path, device_map="auto")
        model = PeftModel.from_pretrained(model, PEFT_dir)
        model.config.use_cash = True
    else:
        model = WhisperForConditionalGeneration.from_pretrained(model_path).to(device)
    return model


def whisper_infer(model, dataloder, forced_decoder_ids, device):
    import gc

    import jiwer
    import numpy as np
    import torch
    from tqdm import tqdm

    pred = []
    ref = []

    # Compile the model for optimized performance
    model = torch.compile(model, mode="reduce-overhead")
    model.eval()

    for _, batch in enumerate(tqdm(dataloder)):
        input_features = batch.get("input_features")
        if input_features is None:
            raise ValueError("input_features missing from batch")

        with torch.amp.autocast(device.type):
            with torch.no_grad():
                generated_tokens = (
                    model.generate(
                        input_features=input_features.to(device),
                        decoder_input_ids=batch["labels"][:, :4].to(device),
                        forced_decoder_ids=forced_decoder_ids,
                        max_new_tokens=255,
                    )
                    .cpu()
                    .numpy()
                )

                labels = batch["labels"].cpu().numpy()
                labels = np.where(labels != -100, labels, processor.tokenizer.pad_token_id)

                # Decoding step
                decode_pred = processor.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                decode_ref = processor.tokenizer.batch_decode(labels, skip_special_tokens=True)

                pred.extend(decode_pred)
                ref.extend(decode_ref)

        # Clear cache and memory
        del generated_tokens, batch, labels, input_features
        gc.collect()
        torch.cuda.empty_cache()

    # Compute WER using jwer
    wer_score = jiwer.wer(ref, pred) * 100  # Convert to percentage
    eval_metrics = {"eval/wer": wer_score}

    return eval_metrics, pred, ref


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Inference Whisper on Kaldi dataset or audio(s)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("data", help="Path to Kaldi dataset, or audio(s) file(s) path", default=None)
    parser.add_argument(
        "model_path_or_size",
        help="You can put a local path of fine-tuned model or a size of openai model [base, tiny, small, medium....]",
        default="small",
    )
    parser.add_argument(
        "--language",
        "-l",
        help=f"Language to use. Among : {', '.join(sorted(k+'('+v+')' for k,v in whisper.tokenizer.LANGUAGES.items()))}.",
        default="fr",
    )
    parser.add_argument("--task", "-t", help="Task to be done by the Whisper model", default="transcribe")
    parser.add_argument("--batch_size", "-bs", help="Batch size of data to infer", type=int, default=32)
    parser.add_argument("--gpus", help="List of GPUS index to use in infer (form 0 to ...)", default=None)
    parser.add_argument("--output", help="Transcription output path", default=None)

    args = parser.parse_args()

    data = args.data
    output_path = args.output
    lang = args.language
    task = args.task
    model_path = args.model_path_or_size
    base_size = [
        "tiny.en",
        "tiny",
        "base.en",
        "base",
        "small.en",
        "small",
        "medium.en",
        "medium",
        "large-v1",
        "large-v2",
        "large-v3",
        "large",
    ]
    if model_path in base_size:
        model_path = f"openai/whisper-{args.model_path_or_size}"
        if model_path.endswith("en"):
            lang = "en"

    device = auto_device()

    os.makedirs(output_path, exist_ok=True)
    peft_folder = None
    if os.path.isdir(model_path):
        for root, _, files in os.walk(data):
            if "adapter_config.json" in files:
                peft_folder = root
                break

    # Create the processor
    processor = WhisperProcessor.from_pretrained(model_path, language=lang, task=task)
    tokenizer_func = lambda x: processor.tokenizer(x).input_ids
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    # load model:
    model = load_model(model_path, device)

    metadata, kaldi_data = kaldi_folder_to_dataset(
        data,
        n_shards=2,
        shuffle=False,
        max_text_length=(tokenizer_func, 448),
    )

    processed_data = process_dataset(processor, kaldi_data, batch_size=args.batch_size)

    eval_dataloader = DataLoader(processed_data, batch_size=args.batch_size, collate_fn=data_collator)

    forced_decoder_ids = processor.get_decoder_prompt_ids(language=lang, task=task)

    wer, prediction, references = whisper_infer(model=model, dataloder=eval_dataloader, forced_decoder_ids=forced_decoder_ids, device=device)

    with open(os.path.join(output_path, "references.txt"), "w") as outref, open(os.path.join(output_path, "predictions.txt"), "w") as outpred:
        for id, r, p in zip(kaldi_data["ID"], references, prediction):
            outref.write(f"{id} {r}\n")
            outpred.write(f"{id} {p}\n")

    print(f"Eval wer: {wer}")
