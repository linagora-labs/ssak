#!/usr/bin/env python3

import vosk

from ssak.utils.dataset import to_audio_batches
from ssak.utils.env import auto_device  # handles option --gpus
from ssak.utils.misc import get_cache_dir, hashmd5
from ssak.utils.monitoring import tic, toc, vram_peak

vosk.SetLogLevel(-1)

import json
import multiprocessing
import os
import shutil
import tempfile
import urllib
import zipfile

import torch


def kaldi_infer(
    modelname,
    audios,
    batch_size=1,
    device=None,
    sort_by_len=False,
    output_ids=False,
    log_memtime=False,
    cache_dir=get_cache_dir("vosk"),
    max_bytes_for_gpu=8000,
    clean_temp_file=True,
):
    """
    Compute transcription on audio(s) using a Kaldi/Vosk model.

    Args:
        model: Name of vosk model, or path to a vosk model, or paths to acoustic model and language model (separated by a comma)
        audios:
            Audio file path(s), or Kaldi folder(s), or Audio waveform(s)
        batch_size: int
            Number of audio files to process in parallel
        device: str
            Device to use for inference ("cpu", "cuda:0").
            If None, will use "cuda:x" if GPU is available ("cpu" otherwise).
        sort_by_len: bool
            Sort audio by length before batching (longest audio first).
        output_ids: bool
            Output ids in front of the transcription.
        log_memtime: bool
            If True, print timing and memory usage information.
        clean_temp_file: bool
            If True, delete temporary files after inference. It's NOT recommended to put False here, as some files may be modified.
    """
    global RECOGNIZER

    modeldir = os.path.join(cache_dir, modelname)
    files_to_move = []

    urlpath_linto = "https://dl.linto.ai/downloads/model-distribution/"

    try:
        # Linto models
        if modelname == "linSTT_fr-FR_v2.2.0":
            modelname = "linSTT_AM_fr-FR_v2.2.0,decoding_graph_fr-FR_Big_v2.2.0"
        elif modelname == "linSTT_en-US_v1.1.0":
            modelname = "linSTT_AM_en-US_v1.0.0,decoding_graph_en-US_v1.1.0"
        elif modelname == "linSTT_en-US_v1.2.0":
            modelname = "linSTT_AM_en-US_v1.0.0,decoding_graph_en-US_Big_v1.2.0"
        elif modelname == "linSTT_ar-AR_v1.1.0":
            modelname = "linSTT_AM_ar-AR_v1.0.0,decoding_graph_ar-AR_v1.1.0"
        elif modelname == "linSTT_ar-AR_v1.2.0":
            modelname = "linSTT_AM_ar-AR_v1.0.0,decoding_graph_ar-AR_v1.2.0"

        if "," in modelname:
            amdir, lmdir = modelname.split(",")

            if not os.path.isdir(amdir):
                if amdir == "linSTT_AM_fr-FR_v2.2.0":
                    amdir = download_zipped_folder(urlpath_linto + "acoustic-models/fr-FR/linSTT_AM_fr-FR_v2.2.0.zip", cache_dir)
                elif amdir == "linSTT_AM_en-US_v1.0.0":
                    amdir = download_zipped_folder(urlpath_linto + "acoustic-models/en-US/linSTT_AM_en-US_v1.0.0.zip", cache_dir)
                elif amdir == "linSTT_AM_ar-AR_v1.0.0":
                    amdir = download_zipped_folder(urlpath_linto + "acoustic-models/ar-AR/linSTT_AM_ar-AR_v1.0.0.zip", cache_dir)

            if not os.path.isdir(lmdir):
                if lmdir.endswith("fr-FR_Big_v2.2.0"):
                    lmdir = download_zipped_folder(urlpath_linto + "decoding-graphs/LVCSR/fr-FR/decoding_graph_fr-FR_Big_v2.2.0.zip", cache_dir)
                elif lmdir.endswith("en-US_v1.1.0"):
                    lmdir = download_zipped_folder(urlpath_linto + "decoding-graphs/LVCSR/en-US/decoding_graph_en-US_v1.1.0.zip", cache_dir)
                elif lmdir.endswith("en-US_v1.2.0"):
                    lmdir = download_zipped_folder(urlpath_linto + "decoding-graphs/LVCSR/en-US/decoding_graph_en-US_Big_v1.2.0.zip", cache_dir)
                elif lmdir.endswith("ar-AR_v1.1.0"):
                    lmdir = download_zipped_folder(urlpath_linto + "decoding-graphs/LVCSR/ar-AR/decoding_graph_ar-AR_v1.1.0.zip", cache_dir)
                elif lmdir.endswith("ar-AR_v1.2.0"):
                    lmdir = download_zipped_folder(urlpath_linto + "decoding-graphs/LVCSR/ar-AR/decoding_graph_ar-AR_v1.2.0.zip", cache_dir)

            modeldir = linagora2vosk(amdir, lmdir)
            files_to_move.append((modeldir, None))

        elif not os.path.isdir(modeldir):
            urlpath = "https://alphacephei.com/vosk/models/"
            modeldir = download_zipped_folder(urlpath + modelname + ".zip", cache_dir)

        conf_file = os.path.join(modeldir, "conf", "mfcc.conf")
        if not os.path.isfile(conf_file):
            conf_file = os.path.join(modeldir, "mfcc.conf")
            if not os.path.isfile(conf_file):
                raise ValueError(f"Cannot find mfcc.conf in {modeldir}")
        sample_rate = read_param_value(conf_file, "sample-frequency", int)
        if sample_rate is None:
            print("WARNING: Cannot find sample-frequency in mfcc.conf, assuming 16000")
            sample_rate = 16000

        if device is None:
            device = auto_device()

        use_gpu = device not in ["cpu", torch.device("cpu")]
        if use_gpu:
            vosk.GpuInit()
            vosk.GpuThreadInit()

        if use_gpu:
            # Link to a model sub-folder (WTF class BatchModel needs it)
            if os.path.exists("model"):
                tmp_file = "model_" + hashmd5("model")
                shutil.move("model", tmp_file)
                files_to_move.append((tmp_file, "model"))
            # Broken symlink
            elif os.path.islink("model"):
                os.remove("model")
            os.symlink(modeldir, "model")
            files_to_move.append(("model", None))
            modeldir = "model"

            # Also needs a conf/ivector.conf file
            ivector_file1 = os.path.join(modeldir, "am", "conf", "ivector_extractor.conf")
            ivector_file = os.path.join(modeldir, "conf", "ivector.conf")
            if not os.path.isfile(ivector_file):
                with open(ivector_file, "w") as f:
                    if os.path.exists(ivector_file1):
                        with open(ivector_file1) as f1:
                            for line in f1:
                                f.write(line)
                    f.write(
                        f"""
                    --lda-matrix={modeldir}/ivector/final.mat
                    --global-cmvn-stats={modeldir}/ivector/global_cmvn.stats
                    --diag-ubm={modeldir}/ivector/final.dubm
                    --ivector-extractor={modeldir}/ivector/final.ie
                    --splice-config={modeldir}/ivector/splice.conf
                    --cmvn-config={modeldir}/ivector/online_cmvn.conf
                    """
                    )
                files_to_move.append((ivector_file, None))
            conf_file = os.path.join(modeldir, "conf", "model.conf")
            if os.path.isfile(conf_file):
                tmp_file = conf_file + "_" + hashmd5(conf_file)
                shutil.move(conf_file, tmp_file)
                files_to_move.append((tmp_file, conf_file))
                with open(tmp_file) as f, open(conf_file, "w") as g:
                    for line in f:
                        if "--min-active" in line:
                            continue
                        g.write(line)
                files_to_move.append((conf_file, None))

            model = vosk.BatchModel()
        else:
            model = vosk.Model(modeldir)
            recognizer = vosk.KaldiRecognizer(model, sample_rate)
            if batch_size > 1:
                RECOGNIZER = recognizer
                pool = multiprocessing.Pool(batch_size)

    finally:
        if clean_temp_file:
            # (Re)move temporary files, starting from the most recent one
            files_to_move.reverse()
            for tmp_file, dest_file in files_to_move:
                if dest_file:
                    # Move file / symbolic link / folder
                    if os.path.exists(dest_file):
                        try:
                            os.remove(dest_file)
                        except IsADirectoryError:
                            shutil.rmtree(dest_file)
                    shutil.move(tmp_file, dest_file)
                else:
                    # Remove file / symbolic link / folder
                    try:
                        os.remove(tmp_file)
                    except IsADirectoryError:
                        shutil.rmtree(tmp_file)

                # Why a try / except?
                # /!\ To be careful with symbolic links:
                #       os.path.isdir returns True on them,
                #       and shutil.rmtree delete their content.

    batches = to_audio_batches(
        audios,
        return_format="bytes",
        sample_rate=sample_rate,
        batch_size=batch_size,
        sort_by_len=sort_by_len,
        output_ids=output_ids,
    )

    # Note: the pipeline is so different depending on whether BatchModel (for GPU) is used or not

    # Compute best predictions
    tic()
    for batch in batches:
        if output_ids:
            ids = [b[1] for b in batch]
            batch = [b[0] for b in batch]

        if use_gpu:
            recognizers = [vosk.BatchRecognizer(model, sample_rate) for _ in range(batch_size)]

            results = ["" for _ in batch]

            ended = [False for _ in batch]

            while not min(ended):
                for i, (audio, recognizer) in enumerate(zip(batch, recognizers)):
                    if len(audio) > max_bytes_for_gpu:
                        audio = audio[:max_bytes_for_gpu]
                        batch[i] = batch[i][max_bytes_for_gpu:]
                    elif len(audio) == 0:
                        if not ended[i]:
                            recognizer.FinishStream()
                            ended[i] = True
                        continue
                    else:
                        batch[i] = b""
                    recognizer.AcceptWaveform(audio)

                # Wait for results from CUDA
                model.Wait()

                for i, recognizer in enumerate(recognizers):
                    pred = recognizer.Result()
                    if len(pred):
                        pred = json.loads(pred)["text"]
                        if results[i]:
                            results[i] += " " + pred
                        else:
                            results[i] = pred

        else:
            if batch_size > 1:
                processes = [pool.apply_async(apply_recognizer_global, args=(audio,)) for audio in batch]
                results = [p.get() for p in processes]
            else:
                results = [apply_recognizer(recognizer, audio) for audio in batch]

        if output_ids:
            for id, pred in zip(ids, results):
                yield (id, pred)
        else:
            for pred in results:
                yield pred

        if log_memtime:
            vram_peak()
    if log_memtime:
        toc("apply network", log_mem_usage=True)


def download_zipped_folder(url, cache_dir, remove_prefix=None):
    dname = url.split("/")[-1]
    assert dname.endswith(".zip")
    dname = dname[:-4]
    destdir = os.path.join(cache_dir, dname)
    if not os.path.exists(destdir):
        destzip = destdir + ".zip"
        if not os.path.exists(destzip):
            print("Downloading", url, "into", destdir)
            os.makedirs(cache_dir, exist_ok=True)
            urllib.request.urlretrieve(url, destzip)
        with zipfile.ZipFile(destzip, "r") as z:
            has_folder_inside = min([f.filename.startswith(dname) for f in z.filelist])
            if has_folder_inside:
                z.extractall(cache_dir)
            else:
                os.makedirs(destdir, exist_ok=True)
                z.extractall(destdir)
            if remove_prefix:
                remove_prefix = remove_prefix.rstrip("/")
                for f in z.filelist:
                    if f.filename.startswith(remove_prefix + "/") and f.filename != remove_prefix + "/":
                        os.rename(
                            os.path.join(destdir, f.filename),
                            os.path.join(destdir, f.filename[len(remove_prefix) + 1 :]),
                        )
                if os.path.isdir(os.path.join(destdir, remove_prefix)):
                    shutil.rmtree(os.path.join(destdir, remove_prefix))
        assert os.path.isdir(destdir)
        os.remove(destzip)
    return destdir


def linagora2vosk(am_path, lm_path):
    conf_path = am_path + "/conf"
    ivector_path = am_path + "/ivector_extractor"

    vosk_path = os.path.join(tempfile.gettempdir(), hashmd5([am_path, lm_path, conf_path, ivector_path]))
    if os.path.isdir(vosk_path):
        shutil.rmtree(vosk_path)
    os.makedirs(vosk_path)
    for path_in, path_out in [
        (am_path, "am"),
        (lm_path, "graph"),
        # (conf_path, "conf"),
        # (ivector_path, "ivector"),
    ]:
        path_out = os.path.join(vosk_path, path_out)
        if os.path.exists(path_out):
            os.remove(path_out)
        os.symlink(path_in, path_out)

    new_ivector_path = os.path.join(vosk_path, "ivector")
    os.makedirs(new_ivector_path)
    for fn in os.listdir(ivector_path):
        os.symlink(os.path.join(ivector_path, fn), os.path.join(new_ivector_path, fn))
    if not os.path.exists(os.path.join(new_ivector_path, "splice.conf")):
        os.symlink(os.path.join(conf_path, "splice.conf"), os.path.join(new_ivector_path, "splice.conf"))

    phones_file = os.path.join(am_path, "phones.txt")
    with open(phones_file) as f:
        silence_indices = []
        for line in f.readlines():
            phoneme, idx = line.strip().split()
            if phoneme.startswith("SIL") or phoneme.startswith("NSN"):
                silence_indices.append(idx)

    new_conf_path = os.path.join(vosk_path, "conf")
    os.makedirs(new_conf_path)
    os.symlink(os.path.join(conf_path, "mfcc.conf"), os.path.join(new_conf_path, "mfcc.conf"))
    with open(os.path.join(new_conf_path, "model.conf"), "w") as f:
        # cf. steps/nnet3/decode.sh
        print(
            """
    --min-active=200
    --max-active=7000
    --beam=13.0
    --lattice-beam=6.0
    --frames-per-chunk=51
    --acoustic-scale=1.0
    --frame-subsampling-factor=3
    --extra-left-context-initial=1
    --endpoint.silence-phones={}
    --verbose=-1
        """.format(":".join(silence_indices)),
            file=f,
        )
    # --endpoint.silence-phones=1:2:3:4:5:6:7:8:9:10

    return vosk_path


def apply_recognizer(recognizer, audio):
    recognizer.AcceptWaveform(audio)
    pred = recognizer.FinalResult()
    if len(pred):
        pred = json.loads(pred)["text"]
    return pred


def apply_recognizer_global(audio):
    global RECOGNIZER
    return apply_recognizer(RECOGNIZER, audio)
    # RECOGNIZER.AcceptWaveform(audio)
    # pred = RECOGNIZER.FinalResult()
    # if len(pred):
    #     pred = json.loads(pred)["text"]
    # return pred


def read_param_value(filename, paramname, t=lambda x: x):
    with open(filename) as f:
        for line in f:
            if line.startswith("--" + paramname):
                return t(line.split("=", 1)[-1].strip())
    return None


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Transcribe audio data using a kaldi model (vosk or LinTO format)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("data", help="Path to data (audio file(s) or kaldi folder(s))", nargs="+")
    parser.add_argument(
        "--model",
        help="Name of vosk, Path to trained folder, or Paths to acoustic and language model (separated by a coma)",
        # default = ".../linSTT_AM_fr-FR_v2.2.0,.../decoding_graph_fr-FR_Big_v2.2.0",
        # default = "vosk-model-fr-0.6-linto-2.2.0",
        default="linSTT_fr-FR_v2.2.0",
    )
    parser.add_argument("--output", help="Output path (will print on stdout by default)", default=None)
    parser.add_argument("--use_ids", help="Whether to print the id before result", default=False, action="store_true")
    parser.add_argument("--batch_size", help="Maximum batch size", type=int, default=1)
    parser.add_argument("--sort_by_len", help="Sort by (decreasing) length", default=False, action="store_true")
    parser.add_argument("--enable_logs", help="Enable logs about time", default=False, action="store_true")
    parser.add_argument("--gpus", help="List of GPU index to use (starting from 0)", default=None)
    parser.add_argument("--cache_dir", help="Path to cache models", default=get_cache_dir("vosk"))
    parser.add_argument("--disable_clean", help="To avoid removing temporary files", action="store_true", default=False)
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

    for reco in kaldi_infer(
        args.model,
        args.data,
        batch_size=args.batch_size,
        output_ids=args.use_ids,
        sort_by_len=args.sort_by_len,
        log_memtime=args.enable_logs,
        cache_dir=args.cache_dir,
        clean_temp_file=not args.disable_clean,
    ):
        if isinstance(reco, str):
            print(reco, file=args.output)
        else:
            print(*reco, file=args.output)
        args.output.flush()
