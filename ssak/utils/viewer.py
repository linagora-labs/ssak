import matplotlib.pyplot as plt
import torchaudio
import whisper.audio

from ssak.utils.player import AudioPlayer


class PlayWav:
    """
    A viewer of segmentation
    """

    def __init__(self, wav, ax=None, draw=True, use_mel=False):
        if use_mel:
            n_plots = 2
            # Import sample rate of wav file
            info = torchaudio.info(wav)
            sample_rate = info.sample_rate
            timescale_mel = 100  # whisper.audio.HOP_LENGTH * 2 * whisper.audio.SAMPLE_RATE / sample_rate

        else:
            n_plots = 1

        if ax is None:
            fig = plt.figure(facecolor="white", tight_layout=True)
            ax = fig.add_subplot(n_plots, 1, 1)
        elif isinstance(ax, plt.Figure):
            ax = ax.add_subplot(n_plots, 1, 1)
        elif use_mel:
            raise NotImplementedError("Plotting mel spectrogram not possible when an axes is specified")
        assert isinstance(ax, plt.Axes), "Please provide None, an Axes or a Figure"
        self.ax = ax
        self.fig = ax.get_figure()
        self.axes = [self.ax]

        if use_mel:
            mel = whisper.audio.log_mel_spectrogram(wav)
            self.ax2 = self.fig.add_subplot(n_plots, 1, 2)
            plt.imshow(mel, aspect="auto", extent=[0, mel.shape[1], 0, mel.shape[0]])
            plt.yticks([])
            plt.setp(plt.gca().get_xticklabels(), visible=False)
            self.axes.append(self.ax2)

        cids = list()
        cids.append(self.fig.canvas.mpl_connect("key_press_event", self._on_keypress))
        cids.append(self.fig.canvas.mpl_connect("button_press_event", self._on_click))
        self.height = 5
        self.maxx = 0
        self.audio = AudioPlayer(wav)
        self.duration = self.audio.getDuration()
        self.timer = self.fig.canvas.new_timer(interval=10)
        self.timer.add_callback(self._update_timeline)
        self.timer.start()

        self.timelines = [ax.plot([0, 0], [0, 0], color="r")[-1] for ax in self.axes]
        self.timescales = [1] + ([timescale_mel] if use_mel else [])

        if draw:
            self.draw()
            self.t_scale = 1
            self.t_bias = 0
        else:
            minx, maxx = self.ax.get_xlim()
            self.t_scale = (maxx - minx) / self.duration
            self.t_bias = minx

        self.maxy = plt.ylim()[-1]
        self._draw_timeline(0)

    def _draw_timeline(self, t):
        """
        Draw the timeline a position t
        :param t: in second, a float

        """
        self._draw_info(t)
        t = self.t_scale * t + self.t_bias
        min, max = self.ax.get_ylim()
        for timeline, scale in zip(self.timelines, self.timescales):
            timeline.set_data([t * scale, t * scale], [min, max])
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def _update_timeline(self):
        """
        Update the timeline given the position in the audio player
        """
        if self.audio is not None and self.audio.playing():
            t = self.audio.time()
            # min, max = self.ax.get_xlim()
            self._draw_timeline(t)

    def _draw_info(self, t):
        """
        Draw information on segment and timestamp
        :param t: a float
        :return:
        """
        plt.xlabel(f"time:{self._hms(t):s}")

    def draw(self):
        plt.sca(self.ax)
        """
        Draw the segmentation
        """
        t, signal = self.audio.getWaveForm()
        plt.plot(t, signal)
        self._draw_info(0)
        plt.xlim(0, self.duration)
        plt.tight_layout()

    def _move(self, duration):
        self._move_to(self.audio.time() + duration)

    def _move_to(self, t):
        t = max(0, min(t, self.duration))
        if self.audio is not None:
            self.audio.seek(t)
        self._draw_timeline(t)

    def _on_keypress(self, event):
        """
        manage the keypress event
        :param event: a key event

        """
        hmin, hmax = self.ax.get_xlim()
        if event.key == "ctrl++" or event.key == "ctrl+=":
            plt.xlim(hmin * 1.5, hmax * 1.5)
        elif event.key == "ctrl+-":
            plt.xlim(hmin / 1.5, hmax / 1.5)
        elif event.key == "escape":
            plt.xlim(0, self.maxx)
            plt.ylim(0, self.maxy)
        elif event.key == "right":
            self._move(+1)
        elif event.key == "left":
            self._move(-1)
        elif event.key == "ctrl+right":
            self._move(+0.1)
        elif event.key == "ctrl+left":
            self._move(-0.1)
        elif event.key == "alt+right":
            self._move(+10)
        elif event.key == "alt+left":
            self._move(-10)
        elif event.key is None and self.audio is not None:
            self.audio.play()
        elif event.key == " " and self.audio is not None:
            if self.audio.playing():
                self.audio.pause()
            else:
                self.audio.pause()
                self.audio.play()

        self.fig.canvas.draw()

    def _on_click(self, event):
        """
        manage the mouse event
        :param event: a mouse event

        """
        if event.xdata is not None:
            self.audio.pause()
            xdata = event.xdata
            xdata = (xdata - self.t_bias) / self.t_scale
            self._move_to(xdata)

    @classmethod
    def _hms(cls, s):
        """
        conversion of seconds into hours, minutes and secondes
        :param s:
        :return: int, int, float
        """
        h = int(s) // 3600
        s %= 3600
        m = int(s) // 60
        s %= 60
        return f"{h:d}:{m:d}:{s:.2f}"


if __name__ == "__main__":
    import os
    import sys

    if len(sys.argv) < 2 or len(sys.argv) > 4:
        print(f"Usage: {os.path.basename(sys.executable)} {sys.argv[0]} filename [start] [end]")
        sys.exit(1)

    wavfile = sys.argv[1]
    start = float(sys.argv[2]) if len(sys.argv) > 2 else None
    end = float(sys.argv[3]) if len(sys.argv) > 3 else None
    if start or end:
        # Cut the wav in a temporary file
        import os
        import tempfile

        if start is None:
            start = 0
        with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
            os.system(" ".join(["sox", wavfile, tmp.name, "trim", str(start), str(end - start) if end is not None else ""]))
            PlayWav(tmp.name, use_mel=True)
    else:
        PlayWav(wavfile, use_mel=True)
    plt.show()
