"""Exercise the desktop's input boundary without starting numerical jobs."""
from pathlib import Path
import tkinter as tk

import pytest

from softcut_solution.gui import Workbench


class FakeBackend:
    def __init__(self):
        self.calls = []

    def inventory(self, root):
        self.calls.append(("inventory", root))
        return {
            "simulation_bundles": [{"label": "angle33", "path": str(root / "fe.json")}],
            "simulation_cases": [{"label": "baseline33", "path": str(root / "case")}],
            "video_packages": [{"label": "1014n33a1", "path": str(root / "video")}],
            "reports": [],
            "projects": {},
            "settings": {"simulation_python": "python.exe", "video_python": "video_python.exe"},
        }


@pytest.fixture(scope="module")
def tk_window():
    try:
        window = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"No graphical Tk display available: {exc}")
    window.withdraw()
    yield window
    window.destroy()


@pytest.fixture
def desktop(tmp_path, tk_window):
    backend = FakeBackend()
    app = Workbench(tk_window, tmp_path, backend=backend)
    tk_window.update_idletasks()
    yield app, backend
    app.destroy()


def test_startup_only_discovers_assets(desktop):
    app, backend = desktop
    assert [name for name, _ in backend.calls] == ["inventory"]
    assert app.notebook.index("end") == 3
    assert app.fe_choice.get() == "angle33"
    assert app.video_choice.get() == "1014n33a1"
    assert app.comparison_options()["intact_status"] == "unknown"
    assert not app.busy


def test_registration_and_reference_are_explicit(desktop):
    app, _ = desktop
    options = app.comparison_options()
    assert options["registration"] == "tip_translation"
    assert options["translation_y_mm"] is None
    assert options["support_mode"] == "continuous"
    app.registration.set("显式旋转与平移")
    app.translation_y.set("2.5")
    app.translation_z.set("-1.2")
    app.reference_index.set("3")
    app.support.set("当前与参考端点支持")
    options = app.comparison_options()
    assert options["registration"] == "explicit"
    assert options["translation_y_mm"] == 2.5
    assert options["translation_z_mm"] == -1.2
    assert options["fe_reference_index"] == 3
    assert options["support_mode"] == "endpoint"


@pytest.mark.parametrize("depth", ["0", "2.01", "nan", "inf", "not a number"])
def test_comparison_cannot_pass_invalid_depth_to_worker(desktop, depth):
    app, backend = desktop
    app.max_depth.set(depth)
    with pytest.raises(ValueError):
        app.comparison_options()
    assert [name for name, _ in backend.calls] == ["inventory"]


def test_plot_retains_acquisition_order_and_missing_values(desktop, tmp_path):
    app, _ = desktop
    if app.figure is None:
        pytest.skip("matplotlib is unavailable")
    csv_path = tmp_path / "observables.csv"
    csv_path.write_text(
        "depth_mm,observed_mean_y_mm,observed_mean_z_mm,predicted_mean_y_mm,predicted_mean_z_mm\n"
        "0,0,0,0,0\n0.2,0.1,-0.1,0.08,-0.09\n0.19,,,0.07,-0.08\n",
        encoding="utf-8",
    )
    app._plot_observables(csv_path)
    line = app.axes[0].lines[0]
    assert list(line.get_xdata()) == [0, 0.2, 0.19]
    assert line.get_ydata()[-1] != line.get_ydata()[-1]  # A real plotting gap, not zero.


def test_close_keeps_active_job_under_supervision(desktop, monkeypatch):
    app, _ = desktop
    closed = []
    monkeypatch.setattr(app.master, "destroy", lambda: closed.append(True))
    app.busy = True
    app.active_task = "视频批处理"
    assert app.request_close() is False
    assert closed == []
    assert "请等待任务结束后关闭" in app.status.get()
    assert "视频批处理" in app.log_text.get("1.0", "end")
    app.busy = False
    assert app.request_close() is True
    assert closed == [True]


def test_simulation_launch_does_not_add_a_second_browser_open(desktop, monkeypatch):
    app, _ = desktop
    tasks = []
    monkeypatch.setattr(app, "_task", lambda label, function, on_success=None:
                        tasks.append((label, function, on_success)))
    app._launch_simulation()
    assert len(tasks) == 1
    assert tasks[0][2] is None


def test_comparison_controls_fit_at_default_size(desktop):
    app, _ = desktop
    window = app.master
    window.geometry("1360x900")
    window.deiconify()
    app.notebook.select(app.comparison_tab)
    window.update_idletasks()
    try:
        button = app.open_report_button
        viewport = app.control_canvas
        assert button.winfo_ismapped()
        assert button.winfo_rooty() + button.winfo_height() <= viewport.winfo_rooty() + viewport.winfo_height()
        assert app.summary_text.winfo_rootx() + app.summary_text.winfo_width() <= window.winfo_rootx() + window.winfo_width()
    finally:
        window.withdraw()
