"""Local desktop workbench for the copied simulation and video projects.

The GUI delegates all numerical and file operations to :mod:`api`.  Worker
threads never read or write Tk variables; their results cross a queue back to
the Tk event loop.  Merely opening this window never starts a computation.
"""
from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
import queue
import threading
import traceback
import webbrowser

import tkinter as tk
from tkinter import filedialog, messagebox, ttk


NAVY = "#18334b"
TEAL = "#087f8c"
BACKGROUND = "#f3f6fa"
MUTED = "#526477"


class Workbench(ttk.Frame):
    """Three-tab desktop interface; ``backend`` is injectable for smoke tests."""

    def __init__(self, master: tk.Tk, workspace_root: Path, backend=None):
        super().__init__(master, padding=18)
        if backend is None:
            from . import api as backend
        self.backend = backend
        self.workspace_root = Path(workspace_root).resolve()
        self.catalog = {}
        self.busy = False
        self.active_task = None
        self.events = queue.Queue()
        self.action_buttons = []
        self.asset_choices = {}
        self.report_path = None
        self._poll_id = None
        self._setup_styles()
        self.pack(fill="both", expand=True)
        self._build_header()
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True, pady=(14, 12))
        self.project_tab = ttk.Frame(self.notebook, padding=16)
        self.comparison_tab = ttk.Frame(self.notebook, padding=16)
        self.assets_tab = ttk.Frame(self.notebook, padding=16)
        self.notebook.add(self.project_tab, text="  项目与运行  ")
        self.notebook.add(self.comparison_tab, text="  视频 · 有限元比较  ")
        self.notebook.add(self.assets_tab, text="  研究资产  ")
        self._build_projects()
        self._build_comparison()
        self._build_assets()
        footer = ttk.Frame(self)
        footer.pack(fill="x")
        self.progress = ttk.Progressbar(footer, mode="indeterminate", length=120)
        self.progress.pack(side="right")
        self.status = tk.StringVar(value="就绪 · 启动工作台不会开始有限元或视频计算")
        ttk.Label(footer, textvariable=self.status, foreground=MUTED).pack(side="left")
        self._poll_id = self.after(100, self._poll)
        self._previous_close_handler = self.master.protocol("WM_DELETE_WINDOW")
        self.master.protocol("WM_DELETE_WINDOW", self.request_close)
        self._close_handler_id = self.master.protocol("WM_DELETE_WINDOW")
        self.refresh()

    def _setup_styles(self):
        style = ttk.Style(self)
        if "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure(".", font=("Microsoft YaHei UI", 10))
        style.configure("TFrame", background=BACKGROUND)
        style.configure("TLabel", background=BACKGROUND, foreground=NAVY)
        style.configure("TNotebook", background=BACKGROUND, borderwidth=0)
        style.configure("TNotebook.Tab", padding=(14, 9))
        style.map("TNotebook.Tab", background=[("selected", "white")])
        style.configure("TLabelframe", background=BACKGROUND, padding=12)
        style.configure("TLabelframe.Label", background=BACKGROUND, foreground=NAVY,
                        font=("Microsoft YaHei UI", 10, "bold"))
        style.configure("TButton", padding=(12, 7))
        style.configure("Primary.TButton", background=TEAL, foreground="white")
        style.map("Primary.TButton", background=[("active", "#066c77"),
                                                  ("disabled", "#a7bac0")])
        style.configure("Title.TLabel", font=("Microsoft YaHei UI", 22, "bold"))
        style.configure("Subtitle.TLabel", foreground=MUTED)
        self.master.configure(background=BACKGROUND)

    def _build_header(self):
        line = ttk.Frame(self)
        line.pack(fill="x")
        ttk.Label(line, text="SoftCutting 研究工作台", style="Title.TLabel").pack(side="left")
        self._button(line, "刷新资产", self.refresh).pack(side="right")
        ttk.Label(self, text="仿真 · 视频观测 · 共同参考态 · 可追溯比较",
                  style="Subtitle.TLabel").pack(anchor="w", pady=(3, 1))
        ttk.Label(self, text=str(self.workspace_root), style="Subtitle.TLabel").pack(anchor="w")

    def _button(self, parent, text, command, primary=False):
        button = ttk.Button(parent, text=text, command=command,
                            style="Primary.TButton" if primary else "TButton")
        self.action_buttons.append(button)
        return button

    def _build_projects(self):
        self.project_tab.columnconfigure(0, weight=1)
        self.project_tab.columnconfigure(1, weight=1)
        self.project_tab.rowconfigure(2, weight=1)
        settings = ttk.LabelFrame(self.project_tab, text="本地环境")
        settings.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 12))
        settings.columnconfigure(1, weight=1)
        self.simulation_python = tk.StringVar()
        self.video_python = tk.StringVar()
        for row, (label, variable) in enumerate((
                ("仿真 Python", self.simulation_python), ("视频 Python", self.video_python))):
            ttk.Label(settings, text=label).grid(row=row, column=0, sticky="w", padx=(0, 12), pady=4)
            ttk.Entry(settings, textvariable=variable).grid(row=row, column=1, sticky="ew", pady=4)
            self._button(settings, "选择", lambda v=variable: self._choose_python(v)).grid(
                row=row, column=2, padx=(8, 0), pady=4)
        bar = ttk.Frame(settings)
        bar.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(8, 0))
        self._button(bar, "保存本机设置", self._save_settings).pack(side="left")
        self._button(bar, "检查运行环境", self._doctor).pack(side="left", padx=8)
        ttk.Label(bar, text="设置保存在本机；运行过程写入日志。", foreground=MUTED).pack(side="left")

        fe = ttk.LabelFrame(self.project_tab, text="有限元工程")
        fe.grid(row=1, column=0, sticky="nsew", padx=(0, 6), pady=(0, 12))
        ttk.Label(fe, text="使用已有完整体有限应变模型与独立审查。\n速度 0.2 mm/s，行程上限 2 mm。",
                  wraplength=450, justify="left").pack(anchor="w", pady=(0, 10))
        self._button(fe, "打开仿真工作台", self._launch_simulation, primary=True).pack(anchor="w")
        ttk.Label(fe, text="配置和启动求解在原仿真界面中完成；本按钮仅启动界面。",
                  wraplength=430, foreground=MUTED).pack(anchor="w", pady=(8, 0))

        video = ttk.LabelFrame(self.project_tab, text="视频工程")
        video.grid(row=1, column=1, sticky="nsew", padx=(6, 0), pady=(0, 12))
        video.columnconfigure(1, weight=1)
        self.video_case = tk.StringVar(value="0930l30a1")
        ttk.Label(video, text="试次 ID").grid(row=0, column=0, sticky="w", padx=(0, 8))
        ttk.Entry(video, textvariable=self.video_case).grid(row=0, column=1, sticky="ew")
        self._button(video, "输入预检", self._preflight).grid(row=0, column=2, padx=(8, 0))
        self.video_manifest = tk.StringVar()
        ttk.Label(video, text="批处理清单").grid(row=1, column=0, sticky="w", padx=(0, 8), pady=8)
        ttk.Entry(video, textvariable=self.video_manifest).grid(row=1, column=1, sticky="ew", pady=8)
        self._button(video, "选择", self._choose_manifest).grid(row=1, column=2, padx=(8, 0), pady=8)
        actions = ttk.Frame(video)
        actions.grid(row=2, column=0, columnspan=3, sticky="w")
        self._button(actions, "清单预检", lambda: self._batch(True)).pack(side="left")
        self._button(actions, "执行视频批处理", lambda: self._batch(False)).pack(side="left", padx=8)
        ttk.Label(video, text="执行会按清单开始视频计算；预检只检查准备条件。",
                  wraplength=460, foreground=MUTED).grid(row=3, column=0, columnspan=3, sticky="w", pady=(8, 0))

        log_frame = ttk.LabelFrame(self.project_tab, text="操作日志")
        log_frame.grid(row=2, column=0, columnspan=2, sticky="nsew")
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        self.log_text = tk.Text(log_frame, height=9, wrap="word", background="white",
                                foreground=NAVY, borderwidth=0, font=("Consolas", 10), state="disabled")
        self.log_text.grid(row=0, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self.log_text.configure(yscrollcommand=scroll.set)

    def _build_comparison(self):
        self.comparison_tab.columnconfigure(0, weight=0, minsize=450)
        self.comparison_tab.columnconfigure(1, weight=1)
        self.comparison_tab.rowconfigure(0, weight=1)
        control_host = ttk.Frame(self.comparison_tab)
        control_host.grid(row=0, column=0, sticky="nsew", padx=(0, 16))
        control_host.columnconfigure(0, weight=1)
        control_host.rowconfigure(0, weight=1)
        self.control_canvas = tk.Canvas(control_host, width=446, background=BACKGROUND,
                                         highlightthickness=0, borderwidth=0)
        self.control_canvas.grid(row=0, column=0, sticky="nsew")
        control_scroll = ttk.Scrollbar(control_host, orient="vertical", command=self.control_canvas.yview)
        control_scroll.grid(row=0, column=1, sticky="ns")
        self.control_canvas.configure(yscrollcommand=control_scroll.set)
        controls = ttk.Frame(self.control_canvas)
        controls_window = self.control_canvas.create_window((0, 0), window=controls, anchor="nw")
        controls.bind("<Configure>", lambda _event: self.control_canvas.configure(
            scrollregion=self.control_canvas.bbox("all")))
        self.control_canvas.bind("<Configure>", lambda event: self.control_canvas.itemconfigure(
            controls_window, width=event.width))
        self.control_canvas.bind("<MouseWheel>", lambda event: self.control_canvas.yview_scroll(
            -int(event.delta / 120), "units"))
        sources = ttk.LabelFrame(controls, text="1  选择两侧结果")
        sources.pack(fill="x", pady=(0, 10))
        sources.columnconfigure(0, weight=1)
        ttk.Label(sources, text="有限元标准结果包").grid(row=0, column=0, sticky="w")
        self.fe_choice = tk.StringVar()
        self.fe_combo = ttk.Combobox(sources, textvariable=self.fe_choice, state="readonly", width=43)
        self.fe_combo.grid(row=1, column=0, sticky="ew", pady=(3, 7))
        self._button(sources, "文件", self._choose_fe).grid(row=1, column=1, padx=(6, 0), pady=(3, 7))
        ttk.Label(sources, text="视频固定 ID 位移包").grid(row=2, column=0, sticky="w")
        self.video_choice = tk.StringVar()
        self.video_combo = ttk.Combobox(sources, textvariable=self.video_choice, state="readonly", width=43)
        self.video_combo.grid(row=3, column=0, sticky="ew", pady=(3, 7))
        self._button(sources, "目录", self._choose_video).grid(row=3, column=1, padx=(6, 0), pady=(3, 7))
        self._button(sources, "检查双方接口与资格", self._inspect_sources).grid(row=4, column=0, columnspan=2, sticky="w")
        ttk.Label(sources, text="若尚无标准包，从已有 FE 保存态导出：", foreground=MUTED).grid(
            row=5, column=0, columnspan=2, sticky="w", pady=(10, 3))
        self.case_choice = tk.StringVar()
        self.case_combo = ttk.Combobox(sources, textvariable=self.case_choice, state="readonly", width=43)
        self.case_combo.grid(row=6, column=0, columnspan=2, sticky="ew")
        export_bar = ttk.Frame(sources)
        export_bar.grid(row=7, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        ttk.Label(export_bar, text="保存帧数").pack(side="left")
        self.export_frames = tk.StringVar(value="21")
        ttk.Spinbox(export_bar, from_=2, to=201, textvariable=self.export_frames, width=5).pack(side="left", padx=7)
        self._button(export_bar, "导出已有 FE 结果", self._export_fe).pack(side="left")

        registration = ttk.LabelFrame(controls, text="2  声明共同参考")
        registration.pack(fill="x", pady=(0, 10))
        registration.columnconfigure(1, weight=1)
        self.registration = tk.StringVar(value="刀尖平移（条件注册）")
        self.registration_combo = ttk.Combobox(registration, state="readonly", textvariable=self.registration,
                                               values=["刀尖平移（条件注册）", "显式旋转与平移"], width=26)
        ttk.Label(registration, text="空间注册").grid(row=0, column=0, sticky="w", padx=(0, 10), pady=3)
        self.registration_combo.grid(row=0, column=1, columnspan=3, sticky="ew", pady=3)
        self.registration_combo.bind("<<ComboboxSelected>>", lambda _e: self._registration_changed())
        self.rotation = tk.StringVar(value="0")
        self.translation_y = tk.StringVar(value="0")
        self.translation_z = tk.StringVar(value="0")
        self.depth_offset = tk.StringVar(value="0")
        self.max_depth = tk.StringVar(value="1.0")
        self.reference_index = tk.StringVar(value="0")
        self.support = tk.StringVar(value="全窗口连续支持")
        ttk.Label(registration, text="旋转 / °").grid(row=1, column=0, sticky="w", pady=3)
        ttk.Entry(registration, textvariable=self.rotation, width=10).grid(row=1, column=1, sticky="ew", pady=3)
        ttk.Label(registration, text="参考态索引").grid(row=1, column=2, sticky="w", padx=(10, 6), pady=3)
        ttk.Entry(registration, textvariable=self.reference_index, width=6).grid(row=1, column=3, sticky="ew", pady=3)
        ttk.Label(registration, text="平移 y / mm").grid(row=2, column=0, sticky="w", pady=3)
        self.translation_y_entry = ttk.Entry(registration, textvariable=self.translation_y, width=10)
        self.translation_y_entry.grid(row=2, column=1, sticky="ew", pady=3)
        ttk.Label(registration, text="z / mm").grid(row=2, column=2, sticky="w", padx=(10, 6), pady=3)
        self.translation_z_entry = ttk.Entry(registration, textvariable=self.translation_z, width=6)
        self.translation_z_entry.grid(row=2, column=3, sticky="ew", pady=3)
        ttk.Label(registration, text="深度偏置 / mm").grid(row=3, column=0, sticky="w", pady=3)
        ttk.Entry(registration, textvariable=self.depth_offset, width=10).grid(row=3, column=1, sticky="ew", pady=3)
        ttk.Label(registration, text="上限 / mm").grid(row=3, column=2, sticky="w", padx=(10, 6), pady=3)
        ttk.Entry(registration, textvariable=self.max_depth, width=6).grid(row=3, column=3, sticky="ew", pady=3)
        ttk.Label(registration, text="材料点支持").grid(row=4, column=0, sticky="w", pady=3)
        ttk.Combobox(registration, textvariable=self.support, state="readonly", width=26,
                     values=["全窗口连续支持", "当前与参考端点支持"]).grid(row=4, column=1, columnspan=3, sticky="ew", pady=3)
        self._registration_changed()

        ttk.Label(controls, text="比较保留条件尺度、首触和时钟的不确定性。\n位移以双方选定参考态为零；不会自动校准或判定起裂。",
                  foreground="#8b5a15", wraplength=420, justify="left").pack(anchor="w", pady=(0, 9))
        actions = ttk.Frame(controls)
        actions.pack(fill="x")
        self._button(actions, "生成共同观测比较", self._compare, primary=True).pack(side="left")
        self.open_report_button = ttk.Button(actions, text="打开完整报告", command=self._open_latest_report, state="disabled")
        self.open_report_button.pack(side="left", padx=(9, 0))

        results = ttk.Frame(self.comparison_tab)
        results.grid(row=0, column=1, sticky="nsew")
        ttk.Label(results, text="共同材料点位移", font=("Microsoft YaHei UI", 14, "bold")).pack(anchor="w")
        self.result_status = tk.StringVar(value="选择结果并检查接口。比较将在独立目录中保存来源、映射和数据。")
        ttk.Label(results, textvariable=self.result_status, wraplength=600, foreground=MUTED).pack(anchor="w", pady=(5, 10))
        self.plot_container = ttk.Frame(results)
        self.plot_container.pack(fill="both", expand=True)
        self.figure = None
        self.canvas = None
        self._build_plot()
        self.summary_text = tk.Text(results, height=8, wrap="word", background="white", foreground=NAVY,
                                     borderwidth=0, state="disabled", font=("Consolas", 9))
        self.summary_text.pack(fill="x", pady=(10, 0))

    def _build_plot(self):
        try:
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            from matplotlib.figure import Figure
            self.figure = Figure(figsize=(6.2, 4.6), dpi=100, facecolor=BACKGROUND)
            self.axes = self.figure.subplots(2, 1, sharex=True)
            for ax, component in zip(self.axes, ("y", "z")):
                ax.set_ylabel(f"{component} displacement / mm")
                ax.grid(alpha=0.18)
                ax.set_facecolor("white")
            self.axes[-1].set_xlabel("Conditional video depth / mm (acquisition order)")
            self.axes[0].text(0.5, 0.5, "Comparison preview", transform=self.axes[0].transAxes,
                              ha="center", va="center", color=MUTED)
            self.figure.tight_layout(pad=2.0)
            self.canvas = FigureCanvasTkAgg(self.figure, master=self.plot_container)
            self.canvas.get_tk_widget().pack(fill="both", expand=True)
            self.canvas.draw_idle()
        except ImportError:
            ttk.Label(self.plot_container, text="安装 matplotlib 后可在此预览曲线。完整 HTML 报告仍可打开。",
                      wraplength=500, foreground=MUTED).pack(expand=True)

    def _build_assets(self):
        ttk.Label(self.assets_tab, text="本地研究资产", font=("Microsoft YaHei UI", 15, "bold")).pack(anchor="w")
        ttk.Label(self.assets_tab, text="双击打开报告；原始数据、已有结果和派生比较分别保留。",
                  foreground=MUTED).pack(anchor="w", pady=(4, 14))
        bar = ttk.Frame(self.assets_tab)
        bar.pack(fill="x", pady=(0, 12))
        self._button(bar, "研究内容介绍", lambda: self._open_path(
            self.workspace_root / "researchtopic" / "oblique_cutting_experiment_and_research.md")).pack(side="left")
        self._button(bar, "接口与方法", self._open_interface_document).pack(side="left", padx=8)
        self._button(bar, "Git 管理规范", self._open_git_document).pack(side="left")
        self._button(bar, "打开项目目录", lambda: self._open_path(self.workspace_root)).pack(side="right")
        fracture = ttk.LabelFrame(self.assets_tab, text="起裂诊断与裂后研究算例")
        fracture.pack(fill="x", pady=(0, 12))
        fracture.columnconfigure(1, weight=1)
        self.fracture_config = tk.StringVar(value=str(
            self.workspace_root / "projects/softmodeling/configs/fracture_network.json"))
        self.fracture_seconds = tk.StringVar(value="300")
        self.fracture_length = tk.StringVar(value="0.15")
        self.fracture_threshold = tk.StringVar()
        self.fracture_case_choice = tk.StringVar()
        ttk.Label(fracture, text="裂后配置 JSON").grid(row=0, column=0, sticky="w", padx=(0, 8), pady=3)
        ttk.Entry(fracture, textvariable=self.fracture_config).grid(row=0, column=1, sticky="ew", pady=3)
        self._button(fracture, "选择配置", self._choose_fracture_config).grid(row=0, column=2, padx=6, pady=3)
        self._button(fracture, "运行裂后算例", self._run_fracture).grid(row=0, column=3, pady=3)
        ttk.Label(fracture, text="已有 FE 算例").grid(row=1, column=0, sticky="w", padx=(0, 8), pady=3)
        self.fracture_case_combo = ttk.Combobox(fracture, textvariable=self.fracture_case_choice,
                                               state="readonly", width=50)
        self.fracture_case_combo.grid(row=1, column=1, columnspan=2, sticky="ew", pady=3)
        self._button(fracture, "诊断起裂指标", self._diagnose_fracture).grid(row=1, column=3, pady=3)
        parameters = ttk.Frame(fracture)
        parameters.grid(row=2, column=0, columnspan=4, sticky="ew", pady=(5, 2))
        for label, variable, width in (("计算预算 / s", self.fracture_seconds, 6),
                                      ("邻域尺度 / mm", self.fracture_length, 6),
                                      ("阈值 / MPa（可留空）", self.fracture_threshold, 8)):
            ttk.Label(parameters, text=label).pack(side="left", padx=(0, 5))
            ttk.Entry(parameters, textvariable=variable, width=width).pack(side="left", padx=(0, 14))
        self.fracture_open_button = ttk.Button(parameters, text="打开本次报告", state="disabled",
                                               command=lambda: self._open_path(self.fracture_report_path))
        self.fracture_open_button.pack(side="right")
        ttk.Label(fracture, text="默认配置为自由裂路内聚网络，裂路在内部网格边中选择；可选择混合模式预设路径对照。整切与网格/间隙收敛尚未通过定量验证。点击运行才求解，示例参数不代表材料实测值。",
                  foreground=MUTED, wraplength=1000).grid(row=3, column=0, columnspan=4, sticky="w", pady=(6, 0))
        table = ttk.Frame(self.assets_tab)
        table.pack(fill="both", expand=True)
        table.columnconfigure(0, weight=1)
        table.rowconfigure(0, weight=1)
        self.report_tree = ttk.Treeview(table, columns=("label", "path"), show="headings", selectmode="browse")
        self.report_tree.heading("label", text="报告 / 研究资产")
        self.report_tree.heading("path", text="路径")
        self.report_tree.column("label", width=300, minwidth=180)
        self.report_tree.column("path", width=680, minwidth=220)
        self.report_tree.grid(row=0, column=0, sticky="nsew")
        self.report_tree.bind("<Double-1>", lambda _e: self._open_selected_asset())
        self.report_tree.bind("<Return>", lambda _e: self._open_selected_asset())
        scroll = ttk.Scrollbar(table, orient="vertical", command=self.report_tree.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self.report_tree.configure(yscrollcommand=scroll.set)
        self._button(self.assets_tab, "打开所选资产", self._open_selected_asset).pack(anchor="w", pady=(12, 0))

    def refresh(self):
        if self.busy:
            return
        try:
            self.catalog = self.backend.inventory(self.workspace_root)
            settings = self.catalog.get("settings", {})
            self.simulation_python.set(str(settings.get("simulation_python", "")))
            self.video_python.set(str(settings.get("video_python", "")))
            self._fill_choices("simulation_bundles", self.fe_combo, self.fe_choice)
            self._fill_choices("simulation_cases", self.case_combo, self.case_choice)
            self._fill_choices("simulation_cases", self.fracture_case_combo, self.fracture_case_choice)
            self._fill_choices("video_packages", self.video_combo, self.video_choice)
            self.report_tree.delete(*self.report_tree.get_children())
            for report in self.catalog.get("reports", []):
                self.report_tree.insert("", "end", values=(report["label"], report["path"]))
            n_fe = len(self.catalog.get("simulation_bundles", []))
            n_video = len(self.catalog.get("video_packages", []))
            self.status.set(f"资产已刷新 · {n_fe} 个 FE 标准包、{n_video} 个视频位移包")
        except Exception as exc:
            self._log(f"资产检查失败：{exc}")
            self.status.set(f"资产检查失败：{exc}")

    def _fill_choices(self, kind, widget, variable):
        previous_label = variable.get()
        previous_path = self.asset_choices.get(kind, {}).get(previous_label)
        choices = {}
        for item in self.catalog.get(kind, []):
            label = str(item["label"])
            if label in choices:
                label = f"{label} · {item['path']}"
            choices[label] = str(item["path"])
        if previous_path and previous_label.startswith("外部选择") and previous_path not in choices.values():
            choices[previous_label] = previous_path
        self.asset_choices[kind] = choices
        widget.configure(values=list(choices))
        selected = next((label for label, path in choices.items() if path == previous_path), None)
        variable.set(selected or next(iter(choices), ""))

    def _path_for(self, kind, variable):
        value = self.asset_choices.get(kind, {}).get(variable.get())
        if not value:
            raise ValueError("请先选择相应的数据包或算例。")
        return Path(value)

    def _add_external_choice(self, kind, variable, widget, path):
        path = str(Path(path).resolve())
        label = f"外部选择 · {Path(path).parent.name}/{Path(path).name}"
        self.asset_choices.setdefault(kind, {})[label] = path
        widget.configure(values=list(self.asset_choices[kind]))
        variable.set(label)

    def _choose_fe(self):
        path = filedialog.askopenfilename(parent=self, title="选择 FE 标准结果包", initialdir=self.workspace_root,
                                          filetypes=[("标准结果包", "*.json"), ("所有文件", "*.*")])
        if path:
            self._add_external_choice("simulation_bundles", self.fe_choice, self.fe_combo, path)

    def _choose_video(self):
        path = filedialog.askdirectory(parent=self, title="选择包含 material_displacements.npz 的视频目录",
                                       initialdir=self.workspace_root / "projects" / "rec_inc_soft_cut" / "results")
        if path:
            self._add_external_choice("video_packages", self.video_choice, self.video_combo, path)

    def _choose_python(self, variable):
        path = filedialog.askopenfilename(parent=self, title="选择 Python 解释器",
                                          filetypes=[("Python", "python*.exe"), ("所有文件", "*.*")])
        if path:
            variable.set(path)

    def _choose_manifest(self):
        path = filedialog.askopenfilename(parent=self, title="选择视频批处理清单",
                                          initialdir=self.workspace_root / "projects" / "rec_inc_soft_cut" / "configs",
                                          filetypes=[("JSON", "*.json")])
        if path:
            self.video_manifest.set(path)

    def _choose_fracture_config(self):
        path = filedialog.askopenfilename(parent=self, title="选择裂后研究配置",
                                          initialdir=self.workspace_root / "projects/softmodeling/configs",
                                          filetypes=[("JSON", "*.json")])
        if path:
            self.fracture_config.set(path)

    def _fracture_budget(self):
        try:
            seconds = int(self.fracture_seconds.get())
        except (ValueError, TypeError) as exc:
            raise ValueError("计算预算须为 1–3600 秒的整数。") from exc
        if not 1 <= seconds <= 3600:
            raise ValueError("计算预算须为 1–3600 秒的整数。")
        return seconds

    def _run_fracture(self):
        try:
            seconds = self._fracture_budget()
            value = self.fracture_config.get().strip()
            if not value or not Path(value).is_file():
                raise ValueError("请选择存在的裂后 JSON 配置。")
            config = Path(value)
        except ValueError as exc:
            self._input_error(str(exc))
            return
        self._task("运行裂后研究算例", lambda: self.backend.fracture_run(
            self.workspace_root, config, seconds=seconds), self._fracture_complete)

    def _diagnose_fracture(self):
        try:
            seconds = self._fracture_budget()
            case = self._path_for("simulation_cases", self.fracture_case_choice)
            length = self._number(self.fracture_length.get(), "邻域尺度")
            raw_threshold = self.fracture_threshold.get().strip()
            threshold = self._number(raw_threshold, "阈值") if raw_threshold else None
            if length <= 0 or (threshold is not None and threshold <= 0):
                raise ValueError("邻域尺度和已填写阈值必须大于零。")
        except ValueError as exc:
            self._input_error(str(exc))
            return
        self._task("诊断已有状态起裂指标", lambda: self.backend.fracture_onset(
            self.workspace_root, [case], length_mm=length, threshold_mpa=threshold,
            seconds=seconds), self._fracture_complete)

    def _fracture_complete(self, result):
        self.refresh()
        self.notebook.select(self.assets_tab)
        if result.get("report_path"):
            self.fracture_report_path = Path(result["report_path"])
            self.fracture_open_button.configure(state="normal")

    def _save_settings(self):
        settings = dict(self.catalog.get("settings", {}))
        settings.update(simulation_python=self.simulation_python.get().strip(),
                        video_python=self.video_python.get().strip())
        self._task("保存本机设置", lambda: self.backend.save_settings(self.workspace_root, settings),
                   lambda _result: self.refresh())

    def _doctor(self):
        self._task("检查运行环境", lambda: self.backend.doctor(self.workspace_root))

    def _launch_simulation(self):
        # The API owns opening the browser, including checking the server origin.
        self._task("启动仿真界面", lambda: self.backend.start_fe_workbench(self.workspace_root))

    def _preflight(self):
        case_id = self.video_case.get().strip()
        if not case_id:
            self._input_error("请填写视频试次 ID。")
            return
        self._task("视频输入预检", lambda: self.backend.video_preflight(self.workspace_root, case_id))

    def _batch(self, check_only):
        manifest = self.video_manifest.get().strip()
        if not manifest or not Path(manifest).is_file():
            self._input_error("请选择存在的 JSON 批处理清单。")
            return
        label = "视频清单预检" if check_only else "执行视频批处理"
        self._task(label, lambda: self.backend.video_batch(self.workspace_root, Path(manifest), check_only=check_only))

    def _inspect_sources(self):
        try:
            fe = self._path_for("simulation_bundles", self.fe_choice)
            video = self._path_for("video_packages", self.video_choice)
        except ValueError as exc:
            self._input_error(str(exc))
            return

        def inspect():
            return {"simulation": self.backend.inspect("simulation", fe),
                    "video": self.backend.inspect("video", video)}
        self._task("检查双方接口", inspect, lambda result: self._set_summary(result))

    def _export_fe(self):
        try:
            case = self._path_for("simulation_cases", self.case_choice)
            frames = int(self.export_frames.get())
            if not 2 <= frames <= 201:
                raise ValueError("导出帧数须在 2–201 之间。")
        except ValueError as exc:
            self._input_error(str(exc))
            return

        def completed(result):
            self.refresh()
            if result.get("bundle_path"):
                self._add_external_choice("simulation_bundles", self.fe_choice, self.fe_combo, result["bundle_path"])
            self._set_summary(result)
        self._task("从已有 FE 状态导出标准包", lambda: self.backend.export_fe(self.workspace_root, case, frames=frames), completed)

    def _registration_changed(self):
        state = "disabled" if self.registration.get().startswith("刀尖") else "normal"
        self.translation_y_entry.configure(state=state)
        self.translation_z_entry.configure(state=state)

    @staticmethod
    def _number(value, label):
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label}须为数字。") from exc
        if not math.isfinite(number):
            raise ValueError(f"{label}须为有限数值。")
        return number

    def comparison_options(self):
        """Read and validate controls on the Tk thread before starting a worker."""
        explicit = self.registration.get() == "显式旋转与平移"
        maximum = self._number(self.max_depth.get(), "最大深度")
        if not 0 < maximum <= 2:
            raise ValueError("比较深度上限须大于 0 且不超过 2 mm。")
        try:
            reference = int(self.reference_index.get())
        except ValueError as exc:
            raise ValueError("FE 参考态索引须为非负整数。") from exc
        if reference < 0:
            raise ValueError("FE 参考态索引须为非负整数。")
        return {
            "registration": "explicit" if explicit else "tip_translation",
            "rotation_deg": self._number(self.rotation.get(), "旋转角"),
            "translation_y_mm": self._number(self.translation_y.get(), "y 平移") if explicit else None,
            "translation_z_mm": self._number(self.translation_z.get(), "z 平移") if explicit else None,
            "depth_offset_mm": self._number(self.depth_offset.get(), "深度偏置"),
            "max_depth_mm": maximum,
            "fe_reference_index": reference,
            "support_mode": "continuous" if self.support.get() == "全窗口连续支持" else "endpoint",
            "intact_status": "unknown",
        }

    def _compare(self):
        try:
            fe = self._path_for("simulation_bundles", self.fe_choice)
            video = self._path_for("video_packages", self.video_choice)
            options = self.comparison_options()
        except ValueError as exc:
            self._input_error(str(exc))
            return
        self.result_status.set("正在建立固定材料点映射和共同观测；保留缺失和资格信息。")
        self._task("生成共同观测比较", lambda: self.backend.compare(self.workspace_root, fe, video, options),
                   self._comparison_complete)

    def _comparison_complete(self, result):
        self.refresh()
        self._set_summary(result)
        report = result.get("report_path")
        if report:
            self.report_path = Path(report)
            self.open_report_button.configure(state="normal")
            self._plot_observables(self.report_path.parent / "observables.csv")
        self.result_status.set("比较已保存。曲线按视频采集顺序显示；完整报告包含来源、资格和覆盖。")

    def _plot_observables(self, path):
        if self.figure is None or not path.is_file():
            return
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            return

        def values(key):
            result = []
            for row in rows:
                try:
                    value = float(row.get(key, ""))
                    result.append(value if math.isfinite(value) else math.nan)
                except (TypeError, ValueError):
                    result.append(math.nan)
            return result

        depth = values("depth_mm")
        for ax, component in zip(self.axes, ("y", "z")):
            ax.clear()
            ax.set_facecolor("white")
            ax.plot(depth, values(f"observed_mean_{component}_mm"), color=TEAL, linewidth=1.8, label="Video · conditional")
            ax.plot(depth, values(f"predicted_mean_{component}_mm"), color=NAVY, linewidth=1.8, linestyle="--", label="FE · sampled")
            ax.set_ylabel(f"{component} displacement / mm")
            ax.grid(alpha=0.18)
            ax.legend(loc="best", fontsize=8)
        self.axes[-1].set_xlabel("Conditional video depth / mm (acquisition order)")
        self.figure.tight_layout(pad=2.0)
        self.canvas.draw_idle()

    def _task(self, label, function, on_success=None):
        if self.busy:
            return
        self.busy = True
        self.active_task = label
        self.status.set(f"{label}…")
        self._log(f"\n▶ {label}")
        self.progress.start(12)
        for button in self.action_buttons:
            button.configure(state="disabled")

        def worker():
            try:
                result = function()
                self.events.put((label, result, None, on_success))
            except Exception as exc:
                self.events.put((label, None, (str(exc), traceback.format_exc()), None))
        threading.Thread(target=worker, name="softcut-workbench-job", daemon=True).start()

    def _poll(self):
        try:
            while True:
                label, result, error, callback = self.events.get_nowait()
                self.busy = False
                self.active_task = None
                self.progress.stop()
                for button in self.action_buttons:
                    button.configure(state="normal")
                if error:
                    self.status.set(f"{label}失败：{error[0]}")
                    self._log(error[1])
                    self.result_status.set(f"{label}失败：{error[0]}")
                    messagebox.showerror(label, error[0], parent=self)
                else:
                    self.status.set(f"{label}完成")
                    self._log(self._format(result))
                    if callback:
                        try:
                            callback(result)
                        except Exception as exc:
                            self._log(traceback.format_exc())
                            self.status.set(f"结果已生成，界面更新失败：{exc}")
        except queue.Empty:
            pass
        self._poll_id = self.after(100, self._poll)

    @staticmethod
    def _format(value):
        try:
            text = json.dumps(value, ensure_ascii=False, indent=2, default=str)
        except (TypeError, ValueError):
            text = str(value)
        return text if len(text) <= 20000 else text[:20000] + "\n…（更多内容见运行日志或报告）"

    def _set_summary(self, value):
        self.summary_text.configure(state="normal")
        self.summary_text.delete("1.0", "end")
        self.summary_text.insert("end", self._format(value))
        self.summary_text.configure(state="disabled")

    def _log(self, message):
        self.log_text.configure(state="normal")
        self.log_text.insert("end", str(message) + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _input_error(self, message):
        messagebox.showerror("检查输入", message, parent=self)

    def _open_path(self, path):
        path = Path(path).resolve()
        if not path.exists():
            self._input_error(f"文件或目录尚不存在：\n{path}")
            return
        if path.suffix.lower() in {".html", ".htm"}:
            webbrowser.open(path.as_uri())
        elif os.name == "nt":
            os.startfile(str(path))
        else:
            webbrowser.open(path.as_uri())

    def _open_latest_report(self):
        if self.report_path:
            self._open_path(self.report_path)

    def _open_selected_asset(self):
        selection = self.report_tree.selection()
        if selection:
            self._open_path(self.report_tree.item(selection[0], "values")[1])

    def _open_interface_document(self):
        for name in ("interface_contract.md", "interfaces.md", "integration.md", "architecture.md"):
            path = self.workspace_root / "docs" / name
            if path.is_file():
                self._open_path(path)
                return
        self._open_path(self.workspace_root / "projects" / "softmodeling" / "docs" / "video_fe_comparison.md")

    def _open_git_document(self):
        for name in ("git_workflow.md", "git_management.md", "GIT_WORKFLOW.md"):
            path = self.workspace_root / "docs" / name
            if path.is_file():
                self._open_path(path)
                return
        self._open_path(self.workspace_root / "README.md")

    def request_close(self):
        """Keep the process supervising an active job until it has finished.

        A daemon GUI worker may be waiting on a budgeted child process. Closing
        Tk must not terminate that supervisor while leaving its child running.
        This is an informational state, not another permission prompt.
        """
        if self.busy:
            label = self.active_task or "后台任务"
            message = f"{label}仍在运行。请等待任务结束后关闭，工作台正在监督进程与运行预算。"
            self.status.set(message)
            self._log(message)
            self.notebook.select(self.project_tab)
            return False
        self.master.destroy()
        return True

    def destroy(self):
        try:
            if self.master.protocol("WM_DELETE_WINDOW") == getattr(self, "_close_handler_id", None):
                self.master.protocol("WM_DELETE_WINDOW", self._previous_close_handler)
        except tk.TclError:
            pass
        if self._poll_id is not None:
            try:
                self.after_cancel(self._poll_id)
            except tk.TclError:
                pass
            self._poll_id = None
        super().destroy()


def main(root=None, *, smoke=False):
    """Launch the desktop workspace; ``smoke`` builds and closes without jobs."""
    workspace_root = Path(root) if root is not None else Path(__file__).resolve().parents[1]
    window = tk.Tk()
    window.title("SoftCutting · 仿真与视频研究工作台")
    window.geometry("1360x900")
    window.minsize(1120, 800)
    app = Workbench(window, workspace_root)
    if smoke:
        window.update_idletasks()
        app.destroy()
        window.destroy()
        return 0
    window.mainloop()
    return 0


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SoftCutting local desktop workbench")
    parser.add_argument("--root", type=Path)
    parser.add_argument("--smoke", action="store_true", help="Build and close the UI without starting any job")
    arguments = parser.parse_args()
    raise SystemExit(main(arguments.root, smoke=arguments.smoke))
