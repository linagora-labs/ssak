import nemo.collections.asr as nemo_asr


def load_model(model_path, device="cuda"):
    if model_path.endswith(".nemo"):
        model = nemo_asr.models.ASRModel.restore_from(model_path, map_location=device)
    else:
        model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_path, map_location=device)
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
    args = parser.parse_args()
    model = load_model(args.model)
    print(infer(model, args.data))


if __name__ == "__main__":
    cli()
