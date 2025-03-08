#!/usr/bin/env python3

from ssak.utils.linstt import linstt_transcribe

if __name__ == "__main__":
    import argparse
    import json
    import os
    import sys

    parser = argparse.ArgumentParser(description="Transcribe audio file with LinSTT", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("audio_file", help="Audio file to transcribe", nargs="+")
    parser.add_argument("--output_file", default=None, help="Output file")
    parser.add_argument("--output_dir", default=None, help="Output folder")
    parser.add_argument(
        "--transcription_server",
        help="Transcription server",
        # default="http://biggerboi.linto.ai:8000",
        # default="https://alpha.api.linto.ai/stt-french-generic",
        default="https://api.linto.ai/stt-french-generic",
    )
    parser.add_argument("--diarization_server", help="Diarization server", default=None)
    parser.add_argument("--language", default=None, help="Target language")
    parser.add_argument("--num_speakers", type=int, help="Number of speakers", default=None)
    parser.add_argument("--speaker_names", type=str, help="Names of speakers to identify", default=None)
    parser.add_argument("--convert_numbers", default=False, action="store_true", help="Convert numbers to text")
    parser.add_argument("--min_vad_duration", default=30, type=float, help="Minimum duration of speech segments after VAD")
    parser.add_argument("--disable_punctuation", default=False, action="store_true", help="Disable punctuation")
    parser.add_argument("--diarization_service_name", default=None, help="Diarization service name (ex: stt-diarization-pyannote)")
    parser.add_argument("--output_format", default="json", help="Output format (json, text, vtt, srt)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose mode")
    # parser.add_argument('--return_raw', default = True, action='store_true', help='Convert numbers to text')
    args = parser.parse_args()

    if args.output_file:
        os.makedirs(os.path.dirname(os.path.realpath(args.output_file)), exist_ok=True)
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    if not args.num_speakers:
        args.num_speakers = None

    with open(args.output_file, "w") if args.output_file else (sys.stdout if not args.output_dir else open(os.devnull, "w")) as f:
        for audio_file in args.audio_file:
            print("Processing", audio_file)
            result = linstt_transcribe(
                audio_file,
                transcription_server=args.transcription_server,
                language=args.language,
                diarization_server=args.diarization_server,
                diarization=args.num_speakers,
                speaker_identification=args.speaker_names,
                convert_numbers=args.convert_numbers,
                punctuation=not args.disable_punctuation,
                min_vad_duration=args.min_vad_duration,
                diarization_service_name=args.diarization_service_name,
                output_format=args.output_format,
                verbose=args.verbose,
            )
            if isinstance(result, str):
                print(result, file=f)
            else:
                json.dump(result, f, indent=2, ensure_ascii=False)
            f.flush()
            if args.output_dir:
                if isinstance(result, str):
                    with open(
                        os.path.join(
                            args.output_dir,
                            os.path.basename(audio_file) + ".linstt." + args.output_format.replace("plain", "txt"),
                        ),
                        "w",
                    ) as f2:
                        print(result, file=f2)
                else:
                    with open(os.path.join(args.output_dir, os.path.basename(audio_file) + ".linstt.json"), "w") as f2:
                        json.dump(result, f2, indent=2, ensure_ascii=False)
