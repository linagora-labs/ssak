import os
import tempfile
import time
import wave

import numpy as np
import pyaudio

if "disable asla messages":
    from ctypes import *

    # From alsa-lib Git 3fd4ab9be0db7c7430ebd258f2717a976381715d
    # $ grep -rn snd_lib_error_handler_t
    # include/error.h:59:typedef void (*snd_lib_error_handler_t)(const char *file, int line, const char *function, int err, const char *fmt, ...) /* __attribute__ ((format (printf, 5, 6))) */;
    # Define our error handler type
    ERROR_HANDLER_FUNC = CFUNCTYPE(None, c_char_p, c_int, c_char_p, c_int, c_char_p)

    def py_error_handler(filename, line, function, err, fmt):
        pass

    c_error_handler = ERROR_HANDLER_FUNC(py_error_handler)
    asound = cdll.LoadLibrary("libasound.so")
    asound.snd_lib_error_set_handler(c_error_handler)


class AudioPlayer:
    """
    Player implemented with PyAudio

    http://people.csail.mit.edu/hubert/pyaudio/

    Mac OS X:
        brew install portaudio
        pip install http://people.csail.mit.edu/hubert/pyaudio/packages/pyaudio-0.2.8.tar.gz

    Linux OS:
        sudo apt-get install libasound-dev portaudio19-dev libportaudio2 libportaudiocpp0
        pip install http://people.csail.mit.edu/hubert/pyaudio/packages/pyaudio-0.2.8.tar.gz
    """

    def __init__(self, wav):
        self.p = pyaudio.PyAudio()
        self.pos = 0
        self.stream = None
        self._open(wav)
        self.tmpwav = None

    def callback(self, in_data, frame_count, time_info, status):
        data = self.wf.readframes(frame_count)
        self.pos += frame_count
        return (data, pyaudio.paContinue)

    def _open(self, wav):
        assert wav and os.path.isfile(wav), f"File not found: {wav}"

        try:
            self.wf = wave.open(wav, "rb")

        except:
            # Convert wav to a wave file
            self.tmpwav = tempfile.mktemp(suffix=".wav")
            cmd = f"ffmpeg -i {wav} -acodec pcm_s16le -ac 1 -ar 16000 {self.tmpwav} > /dev/null 2> /dev/null"
            os.system(cmd)
            wav = self.tmpwav
            self.wf = wave.open(wav, "rb")

        self.getWaveForm()
        self.stream = self.p.open(
            format=self.p.get_format_from_width(self.wf.getsampwidth()),
            channels=self.wf.getnchannels(),
            rate=self.wf.getframerate(),
            output=True,
            stream_callback=self.callback,
        )
        self.pause()
        self.seek(0)

    def play(self):
        self.stream.start_stream()

    def pause(self):
        self.stream.stop_stream()

    def seek(self, seconds=0.0):
        sec = seconds * self.wf.getframerate()
        self.pos = int(sec)
        self.wf.setpos(int(sec))

    def time(self):
        return float(self.pos) / self.wf.getframerate()

    def playing(self):
        return self.stream.is_active()

    def close(self):
        self.stream.close()
        self.wf.close()
        self.p.terminate()
        if self.tmpwav:
            os.remove(self.tmpwav)

    def getWaveForm(self):
        signal = self.wf.readframes(-1)
        signal = np.frombuffer(signal, "int16")
        if len(signal) != self.wf.getnframes():
            signal = np.frombuffer(signal, "int32")
        assert len(signal) == self.wf.getnframes(), f"len(signal) {len(signal)} != self.wf.getnframes {self.wf.getnframes()}"
        fs = self.wf.getframerate()
        t = np.linspace(0, len(signal) / fs, num=len(signal))
        return t, signal

    def getDuration(self):
        return self.wf.getnframes() / self.wf.getframerate()


_player = None
_player_filename = None


def play_audiofile(filename, start=None, end=None, ask_for_replay=False, precision=0.01, can_cache=True, additional_commands={}):
    assert isinstance(additional_commands, dict)
    if additional_commands:
        ask_for_replay = True
    if additional_commands:
        assert "" not in additional_commands, "Empty string is not allowed as a key in additional_commands"
        assert "r" not in additional_commands, "Key 'r' is reserved for replay"

    msg = "(Type 'r' to replay" + (", " + ", ".join(f"'{k}' to {v}" for k, v in additional_commands.items() if v not in ["debug"]) if additional_commands else "") + ")"
    keys = list(additional_commands.keys()) + ["r", ""]
    afford_float = max([isinstance(k, float) for k in keys])
    afford_int = max([isinstance(k, int) for k in keys])

    if start is None:
        start = 0

    if can_cache:
        global _player, _player_filename
        if _player is not None and _player_filename != filename:
            _player.close()
            _player = None
        if _player is None:
            _player = AudioPlayer(filename)
            _player_filename = filename
        player = _player
    else:
        player = AudioPlayer(filename)
    try:
        x = "r"
        while x == "r":
            player.seek(start)
            player.play()
            if end is None:
                slept = 0
                while player.playing() and (end is None or slept < end - start):
                    time.sleep(precision)
                    slept += precision
            else:
                time.sleep(end - start)
            player.pause()
            if ask_for_replay:
                x = None
                while x not in keys:
                    x = input(msg)
                    if afford_int and x.isdigit():
                        x = int(x)
                        break
                    if afford_float and is_float(x):
                        x = float(x)
                        break
            else:
                x = ""
    finally:
        if not can_cache:
            player.close()

    return x


def is_float(x):
    try:
        float(x)
        return True
    except:
        return False


if __name__ == "__main__":
    import sys
    import time

    if len(sys.argv) < 2 or len(sys.argv) > 4:
        print(f"Usage: {os.path.basename(sys.executable)} {sys.argv[0]} filename [start] [end]")
        sys.exit(1)

    filename = sys.argv[1]
    start = None
    end = None
    if len(sys.argv) > 2:
        start = float(sys.argv[2])
    if len(sys.argv) > 3:
        end = float(sys.argv[3])

    play_audiofile(filename, start, end, ask_for_replay=True)
