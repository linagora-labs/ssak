#!/usr/bin/env python3

import os
import tempfile

from ssak.utils.audio import save_audio
from ssak.utils.dataset import to_audio_batches
from ssak.utils.linstt import linstt_transcribe
from ssak.utils.monitoring import tic, toc


def linstt_infer(
    audios,
    transcription_server,
    sample_rate=16000,
    convert_numbers=False,
    punctuation=False,
    sort_by_len=False,
    output_ids=False,
    log_memtime=False,
    verbose=False,
):
    """
    audios:
            Audio file path(s), or Kaldi folder(s), or Audio waveform(s)
    transcription_server:
            URL of the transcription server
    sample_rate:
            Sample rate of the audio
    convert_numbers:
            Whether to convert numbers in text to numbers
    punctuation:
            Whether to add punctuation
    sort_by_len:
            Whether to sort the audio by length
    output_ids:
            Whether to output the audio id before the transcription
    verbose:
            Whether to print verbose information
    """

    audios = to_audio_batches(
        audios,
        return_format="array",
        sample_rate=sample_rate,
        batch_size=0,  # No batch
        sort_by_len=sort_by_len,
        output_ids=True,
    )

    tmp_file = tempfile.mktemp(suffix=".wav")

    try:
        for audio in audios:
            assert len(audio) == 2
            audio, audio_id = audio

            save_audio(tmp_file, audio, sample_rate=sample_rate)

            tic()
            try:
                result = linstt_transcribe(
                    tmp_file,
                    transcription_server=transcription_server,
                    convert_numbers=convert_numbers,
                    punctuation=punctuation,
                    # min_vad_duration=min_vad_duration,
                    verbose=verbose,
                )
            except Exception as err:
                raise RuntimeError(f"Error while transcribing {audio_id}") from err
            if log_memtime:
                toc()

            p = result["transcription_result"]

            if output_ids:
                yield (audio_id, p)
            else:
                yield p

        if log_memtime:
            toc(total=True)

    finally:
        if os.path.isfile(tmp_file):
            os.remove(tmp_file)


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Transcribe audio(s) using a LinTO transcription server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("data", help="Path to data (audio file(s) or kaldi folder(s))", nargs="+")
    parser.add_argument("--server", help="Path to the server", default="https://api.linto.ai/stt-french-generic")
    parser.add_argument("--output", help="Output path (will print on stdout by default)", default=None)
    parser.add_argument("--use_ids", help="Whether to print the id before result", default=False, action="store_true")
    parser.add_argument("--enable_logs", help="Enable logs about time", default=False, action="store_true")
    parser.add_argument("-v", "--verbose", help="Enable verbose (curl requests)", default=False, action="store_true")
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

    for reco in linstt_infer(
        args.data,
        transcription_server=args.server,
        output_ids=args.use_ids,
        log_memtime=args.enable_logs,
        verbose=args.verbose,
    ):
        if isinstance(reco, str):
            print(reco, file=args.output)
        else:
            print(*reco, file=args.output)
        args.output.flush()
