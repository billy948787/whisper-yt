"""終端機進度條（不依賴額外套件）。"""

import sys
from collections.abc import Mapping
from typing import TextIO

BAR_WIDTH = 28


def render(percent: int, width: int = BAR_WIDTH) -> str:
    percent = max(0, min(100, int(percent)))
    filled = round(percent * width / 100)
    return "█" * filled + "░" * (width - filled)


class ProgressBar:
    """互動式終端機更新同一行；非互動環境只在每 step% 印一次。"""

    def __init__(self, label: str, stream: TextIO | None = None, step: int = 10) -> None:
        self.label = label
        self.stream = stream if stream is not None else sys.stderr
        self.step = step
        self.percent = -1
        self._bucket = -1
        self._interactive = bool(self.stream.isatty())

    def update(self, percent: int) -> None:
        percent = max(0, min(100, int(percent)))
        if percent == self.percent:
            return
        self.percent = percent
        if self._interactive:
            self.stream.write(f"\r{self.label}：[{render(percent)}] {percent:3d}%")
        else:
            bucket = percent // self.step
            if bucket <= self._bucket:
                return
            self._bucket = bucket
            self.stream.write(f"{self.label}：{percent}%\n")
        self.stream.flush()

    def finish(self) -> None:
        if self._interactive and self.percent >= 0:
            self.stream.write("\n")
            self.stream.flush()
        self.percent = -1


class PhaseProgress:
    """把 (階段, 百分比) 轉成每個階段各自一條進度條。"""

    def __init__(self, labels: Mapping[str, str], stream: TextIO | None = None) -> None:
        self._labels = labels
        self._stream = stream
        self._bars: dict[str, ProgressBar] = {}

    def __call__(self, phase: str, percent: int) -> None:
        bar = self._bars.get(phase)
        if bar is None:
            self.finish()
            bar = self._bars[phase] = ProgressBar(
                self._labels.get(phase, phase), stream=self._stream
            )
        bar.update(percent)

    def finish(self) -> None:
        for bar in self._bars.values():
            bar.finish()
        self._bars.clear()
