from io import StringIO

from whisper_yt.progress import PhaseProgress, ProgressBar, render


class FakeTty(StringIO):
    def isatty(self) -> bool:
        return True


def test_render_bar() -> None:
    assert render(0) == "░" * 28
    assert render(100) == "█" * 28
    assert render(50) == "█" * 14 + "░" * 14
    assert render(150) == "█" * 28
    assert render(-5) == "░" * 28


def test_progress_bar_non_interactive_reports_each_bucket() -> None:
    stream = StringIO()
    bar = ProgressBar("轉錄", stream=stream, step=10)
    for percent in (3, 7, 22, 100):
        bar.update(percent)
    bar.finish()
    assert stream.getvalue().splitlines() == ["轉錄：3%", "轉錄：22%", "轉錄：100%"]


def test_progress_bar_interactive_rewrites_line() -> None:
    stream = FakeTty()
    bar = ProgressBar("轉錄", stream=stream)
    bar.update(0)
    bar.update(50)
    bar.update(50)
    bar.update(100)
    bar.finish()
    text = stream.getvalue()
    assert "\r" in text
    assert text.endswith("100%\n")
    assert text.count(" 50%") == 1


def test_phase_progress_uses_one_bar_per_phase() -> None:
    stream = StringIO()
    progress = PhaseProgress({"download": "下載模型", "transcribe": "轉錄"}, stream=stream)
    progress("download", 50)
    progress("transcribe", 50)
    progress.finish()
    assert stream.getvalue().splitlines() == ["下載模型：50%", "轉錄：50%"]
