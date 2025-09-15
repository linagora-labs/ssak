import os
import json
import nemo.collections.asr as nemo_asr
import logging
logging.getLogger('nemo_logger').setLevel(logging.ERROR)

def load_model(model_path, device="cuda"):
    if model_path.endswith(".nemo"):
        model = nemo_asr.models.ASRModel.restore_from(model_path, map_location=device)
    else:
        model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_path, map_location=device)
    model.change_decoding_strategy(decoder_type="ctc")
    return model


def infer(model, data, batch_size=4, num_workers=4):
    result = model.transcribe(data, batch_size=batch_size, num_workers=num_workers)
    return result


def cli():
    import argparse

    parser = argparse.ArgumentParser(
        description="Transcribe audio(s) using a model from NeMo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("data", help="Path to data (audio file(s), manifest file or kaldi folder(s))", nargs="+")
    parser.add_argument(
        "--model",
        help="Path to a .nemo model, or name of a pretrained model",
        default="linagora/linto_stt_fr_fastconformer",
    )
    parser.add_argument("--output", default=None)
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    args = parser.parse_args()
    model = load_model(args.model)
    results = infer(model, args.data, batch_size=args.batch_size, num_workers=args.num_workers)
    if os.path.isfile(args.data[0]) and args.data[0].endswith(".jsonl"):
        data = []
        with open(args.data[0], "r", encoding="utf-8") as f:
            for line in f.readlines():
                data.append(json.loads(line)["audio_filepath"])
    else:
        data = args.data
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            for i, result in enumerate(results):
                f.write(f"{data[i]}:\n{result.text}\n\n")
    else:
        for i, result in enumerate(results):
            print(data[i], ":")
            print(result.text)
            print()


if __name__ == "__main__":
    cli()
