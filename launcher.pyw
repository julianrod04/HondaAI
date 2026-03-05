#!/usr/bin/env python3
"""
HondaAI-CARLA-Launcher — Tkinter GUI for managing CARLA simulator,
virtual environments, and project scripts.

Config is stored as a hidden JSON file next to the launcher.
No wizard, no breadcrumb files, no PyInstaller — just plain Python.
"""

import json
import os
import platform
import shlex
import signal
import subprocess
import sys
import threading
import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants & Platform Detection
# ---------------------------------------------------------------------------

_HERE = Path(__file__).resolve().parent
_CONFIG_FILE = _HERE / ".hondaai_launcher_config.json"

IS_WINDOWS = platform.system() == "Windows"
IS_LINUX = platform.system() == "Linux"

QUALITY_OPTIONS = ["Low", "Medium", "High", "Epic"]

_DEFAULT_CONFIG = {
    "carla_root": "",
    "venv_path": "",
    "carla_port": 2000,
    "carla_quality": "Low",
    "carla_offscreen": False,
    "carla_map": "Town10HD",
    "auto_refresh": True,
    "custom_scripts": [],
    "custom_dirs": [],
}

# ---------------------------------------------------------------------------
# Config Persistence
# ---------------------------------------------------------------------------

def _load_config() -> dict:
    if _CONFIG_FILE.exists():
        try:
            return json.loads(_CONFIG_FILE.read_text())
        except Exception:
            pass
    return {}


def _save_config(data: dict):
    try:
        _CONFIG_FILE.write_text(json.dumps(data, indent=2))
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------

def _python_bin(venv_path: Path) -> Path:
    if IS_WINDOWS:
        return venv_path / "Scripts" / "python.exe"
    return venv_path / "bin" / "python"


def _pip_bin(venv_path: Path) -> Path:
    if IS_WINDOWS:
        return venv_path / "Scripts" / "pip.exe"
    return venv_path / "bin" / "pip"


def _activate_cmd(venv_path: Path) -> str:
    if IS_WINDOWS:
        return f'call "{venv_path / "Scripts" / "activate.bat"}"'
    return f"source \"{venv_path / 'bin' / 'activate'}\""


def _carla_executable(carla_root: Path) -> Path:
    if IS_WINDOWS:
        for c in [carla_root / "CarlaUE4.exe",
                  carla_root / "WindowsNoEditor" / "CarlaUE4.exe"]:
            if c.exists():
                return c
        return carla_root / "CarlaUE4.exe"
    for c in [carla_root / "CarlaUE4.sh"]:
        if c.exists():
            return c
    return carla_root / "CarlaUE4.sh"


def _has_carla_executable(carla_root: Path) -> bool:
    return ((carla_root / "CarlaUE4.sh").exists()
            or (carla_root / "CarlaUE4.exe").exists())


def _find_carla_processes() -> list[tuple[int, str]]:
    results = []
    try:
        if IS_WINDOWS:
            out = subprocess.check_output(
                ["tasklist", "/FO", "CSV", "/NH"],
                text=True, timeout=5, stderr=subprocess.DEVNULL)
            for line in out.strip().splitlines():
                parts = line.strip('"').split('","')
                if len(parts) >= 2 and "carla" in parts[0].lower():
                    try:
                        results.append((int(parts[1]), parts[0]))
                    except ValueError:
                        pass
        else:
            out = subprocess.check_output(
                ["ps", "aux"], text=True, timeout=5, stderr=subprocess.DEVNULL)
            for line in out.strip().splitlines()[1:]:
                low = line.lower()
                if "carla" in low and "grep" not in low and "launcher" not in low:
                    parts = line.split(None, 10)
                    if len(parts) >= 11:
                        try:
                            results.append((int(parts[1]), parts[10]))
                        except ValueError:
                            pass
    except Exception:
        pass
    return results


def _kill_pid(pid: int):
    if IS_WINDOWS:
        subprocess.run(["taskkill", "/F", "/PID", str(pid)],
                       capture_output=True, timeout=10)
    else:
        os.kill(pid, signal.SIGTERM)


def _scan_scripts(directory: Path) -> list[Path]:
    try:
        return sorted(p for p in directory.iterdir()
                      if p.suffix == ".py" and p.is_file())
    except Exception:
        return []


def _pick_directory(title: str, initialdir: str = "", parent=None) -> str | None:
    start = initialdir or str(Path.home())

    if IS_LINUX:
        try:
            r = subprocess.run(
                ["zenity", "--file-selection", "--directory",
                 f"--title={title}", f"--filename={start}/"],
                capture_output=True, text=True, timeout=120,
            )
            if r.returncode == 0:
                return r.stdout.strip() or None
            if r.returncode == 1:
                return None
        except FileNotFoundError:
            pass
        try:
            r = subprocess.run(
                ["kdialog", "--getexistingdirectory", start, title],
                capture_output=True, text=True, timeout=120,
            )
            if r.returncode == 0:
                return r.stdout.strip() or None
            if r.returncode == 1:
                return None
        except FileNotFoundError:
            pass

    elif IS_WINDOWS:
        ps = (
            "Add-Type -AssemblyName System.Windows.Forms;"
            "$d = New-Object System.Windows.Forms.FolderBrowserDialog;"
            f"$d.Description = '{title}';"
            f"$d.SelectedPath = '{start}';"
            "$d.ShowNewFolderButton = $true;"
            "if ($d.ShowDialog() -eq 'OK') { $d.SelectedPath }"
        )
        try:
            r = subprocess.run(
                ["powershell", "-NoProfile", "-Command", ps],
                capture_output=True, text=True, timeout=120,
            )
            if r.returncode == 0 and r.stdout.strip():
                return r.stdout.strip()
            return None
        except Exception:
            pass

    return filedialog.askdirectory(title=title, initialdir=start, parent=parent) or None


def _venv_conflict_dialog(parent, venv_path: Path) -> str:
    """Show a modal dialog when a venv already exists.
    Returns 'use', 'override', or 'cancel'."""
    result = tk.StringVar(value="cancel")
    dlg = tk.Toplevel(parent)
    dlg.title("Venv Already Exists")
    dlg.configure(bg="#1e1e2e")
    dlg.resizable(False, False)
    dlg.transient(parent)
    dlg.grab_set()
    dlg.focus_force()
    dlg.protocol("WM_DELETE_WINDOW", dlg.destroy)

    dlg.update_idletasks()
    px, py = parent.winfo_rootx(), parent.winfo_rooty()
    dlg.geometry(f"460x210+{px + 70}+{py + 120}")

    fg = "#cdd6f4"; bg = "#1e1e2e"; surface = "#313244"
    maroon = "#7a1a2e"; muted = "#6c7086"

    tk.Label(dlg, text="Venv Already Exists", bg=bg, fg="#89b4fa",
             font=("Segoe UI", 13, "bold")).pack(anchor="w", padx=24, pady=(18, 4))
    tk.Label(dlg,
             text=f"A virtual environment already exists at:\n{venv_path}",
             bg=bg, fg=fg, font=("Segoe UI", 10), justify="left",
             wraplength=410).pack(anchor="w", padx=24, pady=(0, 4))
    tk.Label(dlg, text="What would you like to do?",
             bg=bg, fg=muted, font=("Segoe UI", 9, "italic")).pack(anchor="w", padx=24)

    sep = tk.Frame(dlg, bg=surface, height=1)
    sep.pack(fill="x", padx=0, pady=12)

    btn_row = tk.Frame(dlg, bg=bg)
    btn_row.pack(fill="x", padx=20, pady=(0, 16))

    style = ttk.Style(dlg)
    style.configure("UseVenv.TButton", font=("Segoe UI", 10), padding=(10, 6))
    style.configure("OverrideVenv.TButton", font=("Segoe UI", 10), padding=(10, 6),
                    foreground=maroon)
    style.configure("CancelVenv.TButton", font=("Segoe UI", 10), padding=(10, 6))

    def _pick(val):
        result.set(val)
        dlg.destroy()

    ttk.Button(btn_row, text="Use the existing venv", style="UseVenv.TButton",
               command=lambda: _pick("use")).pack(side="left", padx=(0, 8))
    ttk.Button(btn_row, text="Override and create new", style="OverrideVenv.TButton",
               command=lambda: _pick("override")).pack(side="left", padx=(0, 8))
    ttk.Button(btn_row, text="Cancel", style="CancelVenv.TButton",
               command=lambda: _pick("cancel")).pack(side="right")

    dlg.wait_window()
    return result.get()


def _find_python() -> str | None:
    """Return a real Python executable that can run -m venv."""
    import shutil
    candidates = [sys.executable]
    for name in ["python3", "python", "python3.12", "python3.11", "python3.10"]:
        found = shutil.which(name)
        if found:
            candidates.append(found)
    for exe in candidates:
        try:
            r = subprocess.run([exe, "-m", "venv", "--help"],
                               capture_output=True, timeout=5)
            if r.returncode == 0:
                return exe
        except Exception:
            pass
    return None


# ---------------------------------------------------------------------------
# Process Registry
# ---------------------------------------------------------------------------

class ProcessEntry:
    def __init__(self, name: str, proc_type: str,
                 proc: subprocess.Popen, cmd: str):
        self.name = name
        self.proc_type = proc_type   # "CARLA" | "Script" | "Shell"
        self.proc = proc
        self.cmd = cmd
        self.pid = proc.pid
        self.started = time.strftime("%H:%M:%S")

    @property
    def status(self) -> str:
        code = self.proc.poll()
        return "Running" if code is None else f"Exited ({code})"

    @property
    def is_running(self) -> bool:
        return self.proc.poll() is None


class ProcessRegistry:
    def __init__(self):
        self._entries: dict[int, ProcessEntry] = {}
        self._lock = threading.Lock()

    def add(self, entry: ProcessEntry):
        with self._lock:
            self._entries[entry.pid] = entry

    def remove(self, pid: int):
        with self._lock:
            self._entries.pop(pid, None)

    def all_entries(self) -> list[ProcessEntry]:
        with self._lock:
            return list(self._entries.values())

    def get(self, pid: int) -> ProcessEntry | None:
        with self._lock:
            return self._entries.get(pid)


# ---------------------------------------------------------------------------
# Main Application
# ---------------------------------------------------------------------------

class CarlaLauncher(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("HondaAI-CARLA-Launcher")
        self.geometry("980x740")
        self.minsize(820, 620)
        self.configure(bg="#1e1e2e")

        cfg = _load_config()

        self.carla_root = tk.StringVar(value=cfg.get("carla_root", _DEFAULT_CONFIG["carla_root"]))
        self.venv_path = tk.StringVar(value=cfg.get("venv_path", _DEFAULT_CONFIG["venv_path"]))
        self.carla_port = tk.IntVar(value=cfg.get("carla_port", _DEFAULT_CONFIG["carla_port"]))
        self.carla_quality = tk.StringVar(value=cfg.get("carla_quality", _DEFAULT_CONFIG["carla_quality"]))
        self.carla_offscreen = tk.BooleanVar(value=cfg.get("carla_offscreen", _DEFAULT_CONFIG["carla_offscreen"]))
        self.carla_map = tk.StringVar(value=cfg.get("carla_map", _DEFAULT_CONFIG["carla_map"]))
        self.auto_refresh = tk.BooleanVar(value=cfg.get("auto_refresh", _DEFAULT_CONFIG["auto_refresh"]))

        self._custom_scripts: list[Path] = [Path(p) for p in cfg.get("custom_scripts", [])]
        self._custom_dirs: list[Path] = [Path(p) for p in cfg.get("custom_dirs", [])]

        self.registry = ProcessRegistry()

        self._term_history: list[str] = []
        self._term_hist_idx: int = -1

        self._build_styles()
        self._build_ui()
        self._refresh_venv_status()
        self._refresh_process_table()
        self._auto_refresh()

        # Show Setup tab first if config is invalid, otherwise CARLA Server
        if self._config_is_valid():
            self.notebook.select(2)  # CARLA Server
        else:
            self.notebook.select(0)  # Setup

        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # -------------------------------------------------------------------
    # Config validity check
    # -------------------------------------------------------------------

    def _config_is_valid(self) -> bool:
        carla = self.carla_root.get()
        venv = self.venv_path.get()
        return (bool(carla) and Path(carla).is_dir() and _has_carla_executable(Path(carla))
                and bool(venv) and _python_bin(Path(venv)).exists())

    # -------------------------------------------------------------------
    # Persistence
    # -------------------------------------------------------------------

    def _on_close(self):
        data = {
            "carla_root": self.carla_root.get(),
            "venv_path": self.venv_path.get(),
            "carla_port": self.carla_port.get(),
            "carla_quality": self.carla_quality.get(),
            "carla_offscreen": self.carla_offscreen.get(),
            "carla_map": self.carla_map.get(),
            "auto_refresh": self.auto_refresh.get(),
            "custom_scripts": [str(p) for p in self._custom_scripts],
            "custom_dirs": [str(p) for p in self._custom_dirs],
        }
        _save_config(data)

        # Kill all launcher-managed processes (process groups where possible)
        for entry in self.registry.all_entries():
            if entry.is_running:
                try:
                    if IS_LINUX:
                        # Kill the entire process group (created via os.setsid)
                        os.killpg(os.getpgid(entry.pid), signal.SIGTERM)
                    else:
                        _kill_pid(entry.pid)
                except Exception:
                    # Fallback: kill just the PID
                    try:
                        _kill_pid(entry.pid)
                    except Exception:
                        pass

        # Also kill any system-level CARLA processes not in the registry
        for pid, _ in _find_carla_processes():
            try:
                _kill_pid(pid)
            except Exception:
                pass

        self.destroy()

    # -------------------------------------------------------------------
    # Styles (Catppuccin Mocha)
    # -------------------------------------------------------------------

    def _build_styles(self):
        s = ttk.Style(self)
        s.theme_use("clam")
        bg = "#1e1e2e"; fg = "#cdd6f4"; accent = "#89b4fa"
        surface = "#313244"; red = "#f38ba8"; green = "#a6e3a1"
        yellow = "#f9e2af"; cyan = "#89dceb"

        s.configure(".", background=bg, foreground=fg, fieldbackground=surface)
        s.configure("TFrame", background=bg)
        s.configure("TLabel", background=bg, foreground=fg, font=("Segoe UI", 10))
        s.configure("Status.TLabel", font=("Segoe UI", 9))
        s.configure("TButton", font=("Segoe UI", 10), padding=6)
        s.configure("Green.TButton", foreground=green)
        s.configure("Red.TButton", foreground=red)
        s.configure("TLabelframe", background=bg, foreground=accent)
        s.configure("TLabelframe.Label", background=bg, foreground=accent,
                    font=("Segoe UI", 11, "bold"))
        s.configure("TCheckbutton", background=bg, foreground=fg)
        s.configure("TCombobox", fieldbackground=surface, foreground=fg,
                    selectbackground=surface, selectforeground=fg)
        s.map("TCombobox",
              fieldbackground=[("readonly", surface)],
              foreground=[("readonly", fg)],
              selectbackground=[("readonly", surface)],
              selectforeground=[("readonly", fg)])
        s.configure("TNotebook", background=bg)
        s.configure("TNotebook.Tab", background=surface, foreground=fg,
                    padding=[12, 4], font=("Segoe UI", 10))
        s.map("TNotebook.Tab",
              background=[("selected", accent)],
              foreground=[("selected", "#1e1e2e")])
        s.configure("Treeview", background=surface, foreground=fg,
                    fieldbackground=surface, rowheight=22)
        s.configure("Treeview.Heading", background=bg, foreground=accent,
                    font=("Segoe UI", 9, "bold"))
        s.map("Treeview",
              background=[("selected", accent)],
              foreground=[("selected", "#1e1e2e")])

        muted = "#6c7086"
        self._c = dict(bg=bg, fg=fg, accent=accent, surface=surface,
                       red=red, green=green, yellow=yellow, cyan=cyan,
                       muted=muted)

    # -------------------------------------------------------------------
    # UI layout
    # -------------------------------------------------------------------

    def _build_ui(self):
        hdr = ttk.Frame(self)
        hdr.pack(fill="x", padx=16, pady=(12, 4))
        ttk.Label(hdr, text="HondaAI-CARLA-Launcher",
                  font=("Segoe UI", 18, "bold"),
                  foreground=self._c["accent"]).pack(side="left")

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True, padx=12, pady=8)

        self._build_setup_tab()
        self._build_venv_tab()
        self._build_carla_tab()
        self._build_scripts_tab()
        self._build_terminal_tab()

    # -------------------------------------------------------------------
    # Tab 1 — Setup (NEW)
    # -------------------------------------------------------------------

    def _build_setup_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="  Setup  ")

        # --- CARLA Root ---
        cf = ttk.LabelFrame(tab, text="CARLA Installation")
        cf.pack(fill="x", padx=12, pady=(12, 6))

        row = ttk.Frame(cf)
        row.pack(fill="x", padx=8, pady=6)
        ttk.Label(row, text="Path:").pack(side="left")
        ttk.Entry(row, textvariable=self.carla_root).pack(
            side="left", padx=6, fill="x", expand=True)
        ttk.Button(row, text="Browse",
                   command=self._setup_browse_carla).pack(side="left", padx=(0, 4))
        ttk.Button(row, text="Auto-detect",
                   command=self._setup_auto_detect_carla).pack(side="left")

        self._setup_carla_status = ttk.Label(cf, text="", style="Status.TLabel")
        self._setup_carla_status.pack(fill="x", padx=8, pady=(0, 6))

        # --- Venv Path ---
        vf = ttk.LabelFrame(tab, text="Virtual Environment")
        vf.pack(fill="x", padx=12, pady=6)

        row2 = ttk.Frame(vf)
        row2.pack(fill="x", padx=8, pady=6)
        ttk.Label(row2, text="Path:").pack(side="left")
        ttk.Entry(row2, textvariable=self.venv_path).pack(
            side="left", padx=6, fill="x", expand=True)
        ttk.Button(row2, text="Browse",
                   command=self._setup_browse_venv).pack(side="left", padx=(0, 4))
        ttk.Button(row2, text="Auto-detect",
                   command=self._setup_auto_detect_venv).pack(side="left", padx=(0, 4))
        self._setup_create_venv_btn = ttk.Button(
            row2, text="Create Venv", style="Green.TButton",
            command=self._setup_create_venv)
        self._setup_create_venv_btn.pack(side="left")

        self._setup_venv_status = ttk.Label(vf, text="", style="Status.TLabel")
        self._setup_venv_status.pack(fill="x", padx=8, pady=(0, 6))

        # --- Package status ---
        pf = ttk.LabelFrame(tab, text="CARLA Requirements")
        pf.pack(fill="x", padx=12, pady=6)
        pr = ttk.Frame(pf)
        pr.pack(fill="x", padx=8, pady=6)
        self._setup_pkg_status = ttk.Label(pr, text="", style="Status.TLabel")
        self._setup_pkg_status.pack(side="left", fill="x", expand=True)
        self._setup_install_btn = ttk.Button(
            pr, text="Install Missing", style="Green.TButton",
            command=self._setup_install_missing)
        self._setup_install_btn.pack(side="left")
        self._setup_install_btn.config(state="disabled")

        # --- Config file ---
        cff = ttk.LabelFrame(tab, text="Config File")
        cff.pack(fill="x", padx=12, pady=6)
        cfr = ttk.Frame(cff)
        cfr.pack(fill="x", padx=8, pady=6)
        self._setup_cfg_label = ttk.Label(cfr, text="", style="Status.TLabel")
        self._setup_cfg_label.pack(side="left", fill="x", expand=True)
        ttk.Button(cfr, text="Reset to Defaults", style="Red.TButton",
                   command=self._setup_reset_config).pack(side="left", padx=(0, 4))
        ttk.Button(cfr, text="Delete Config", style="Red.TButton",
                   command=self._setup_delete_config).pack(side="left")
        self._setup_update_cfg_label()

        # --- Status summary ---
        ttk.Separator(tab).pack(fill="x", padx=12, pady=8)

        self._setup_summary = ttk.Label(
            tab, text="", font=("Segoe UI", 11, "bold"))
        self._setup_summary.pack(padx=12, anchor="w")

        self._setup_detail = ttk.Label(tab, text="", style="Status.TLabel")
        self._setup_detail.pack(padx=12, anchor="w", pady=(2, 0))

        # --- Traces for live validation ---
        self.carla_root.trace_add("write", lambda *_: self._setup_validate())
        self.venv_path.trace_add("write", lambda *_: self._setup_validate())

        # Initial validation
        self._setup_validate()

    def _setup_validate(self):
        """Validate paths and update status icons in the Setup tab."""
        c = self._c
        issues = []

        # CARLA root
        carla = self.carla_root.get()
        if not carla:
            self._setup_carla_status.config(
                text="  \u2717  No path set", foreground=c["red"])
            issues.append("CARLA path not set")
        elif not Path(carla).is_dir():
            self._setup_carla_status.config(
                text=f"  \u2717  Directory not found: {carla}", foreground=c["red"])
            issues.append("CARLA directory not found")
        elif not _has_carla_executable(Path(carla)):
            self._setup_carla_status.config(
                text=f"  \u2717  No CarlaUE4.sh/.exe in {carla}", foreground=c["red"])
            issues.append("CARLA executable not found")
        else:
            self._setup_carla_status.config(
                text=f"  \u2713  Found: {_carla_executable(Path(carla))}",
                foreground=c["green"])

        # Venv
        venv = self.venv_path.get()
        if not venv:
            self._setup_venv_status.config(
                text="  \u2717  No path set", foreground=c["red"])
            issues.append("Venv path not set")
        elif not _python_bin(Path(venv)).exists():
            self._setup_venv_status.config(
                text=f"  \u2717  No Python found in {venv}",
                foreground=c["red"])
            issues.append("Venv not found (create one or set a valid path)")
        else:
            self._setup_venv_status.config(
                text=f"  \u2713  Found: {_python_bin(Path(venv))}",
                foreground=c["green"])

        # Create Venv button state
        carla_ok = (bool(carla) and Path(carla).is_dir()
                    and _has_carla_executable(Path(carla)))
        self._setup_create_venv_btn.config(
            state="normal" if carla_ok else "disabled")

        # Package check — only when both paths are valid
        venv_ok = bool(venv) and _python_bin(Path(venv)).exists()
        if carla_ok and venv_ok:
            self._check_requirements(Path(carla), Path(venv))
        else:
            self._setup_pkg_status.config(text="", foreground=c["fg"])
            self._setup_install_btn.config(state="disabled")

        # Summary
        if not issues:
            self._setup_summary.config(
                text="\u2713  Ready", foreground=c["green"])
            self._setup_detail.config(text="All paths are valid.")
        else:
            self._setup_summary.config(
                text=f"\u2717  {len(issues)} item(s) need attention",
                foreground=c["yellow"])
            self._setup_detail.config(text="  \u2022  " + "\n  \u2022  ".join(issues))

    def _check_requirements(self, carla_root: Path, venv_path: Path):
        """Background check: is 'carla' installed and are PythonAPI/examples
        requirements satisfied?  Updates the package status label."""
        pip = _pip_bin(venv_path)
        if not pip.exists():
            return

        req_file = carla_root / "PythonAPI" / "examples" / "requirements.txt"
        c = self._c

        def _run():
            try:
                # Get installed packages as lowercase set
                proc = subprocess.run(
                    [str(pip), "list", "--format=columns"],
                    capture_output=True, text=True, timeout=15,
                    env=self._venv_env(),
                )
                if proc.returncode != 0:
                    return
                installed = set()
                for line in proc.stdout.strip().splitlines()[2:]:
                    parts = line.split()
                    if parts:
                        installed.add(parts[0].lower().replace("-", "_"))

                # Check carla package
                missing = []
                if "carla" not in installed:
                    missing.append("carla")

                # Parse requirements.txt
                if req_file.exists():
                    for line in req_file.read_text().splitlines():
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue
                        # Handle "pkg @ ...", "pkg>=1.0", "pkg<2.0", "pkg ; ..."
                        name = line.split("@")[0].split(";")[0].split(">")[0].split("<")[0].split("=")[0].split("!")[0].strip()
                        if name and name.lower().replace("-", "_") not in installed:
                            missing.append(name)

                if missing:
                    text = f"  \u2717  Missing {len(missing)} package(s): {', '.join(missing[:5])}"
                    if len(missing) > 5:
                        text += f" (+{len(missing) - 5} more)"
                    self.after(0, lambda: (
                        self._setup_pkg_status.config(text=text, foreground=c["yellow"]),
                        self._setup_install_btn.config(state="normal"),
                    ))
                else:
                    self.after(0, lambda: (
                        self._setup_pkg_status.config(
                            text="  \u2713  All CARLA requirements installed",
                            foreground=c["green"]),
                        self._setup_install_btn.config(state="disabled"),
                    ))
            except Exception:
                pass

        threading.Thread(target=_run, daemon=True).start()

    def _setup_install_missing(self):
        """Install carla package + PythonAPI/examples/requirements.txt."""
        pip = _pip_bin(Path(self.venv_path.get()))
        if not pip.exists():
            messagebox.showerror("No venv", "Create a venv first.")
            return

        carla_root = Path(self.carla_root.get())
        req_file = carla_root / "PythonAPI" / "examples" / "requirements.txt"

        # Build pip install command: always include carla, plus requirements if file exists
        cmd = [str(pip), "install", "carla"]
        if req_file.exists():
            cmd.extend(["-r", str(req_file)])

        self._log(f"Installing CARLA requirements...")
        self.notebook.select(self._term_tab_idx)

        def _run():
            try:
                proc = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                    env=self._venv_env(),
                )
                for line in proc.stdout:
                    line = line.rstrip()
                    if line:
                        self._log(line, "shell")
                proc.wait()
                ok = proc.returncode == 0
                self._log(
                    f"--- Install requirements: {'OK' if ok else f'failed ({proc.returncode})'} ---",
                    "success" if ok else "error")
                # Re-check after install
                self.after(0, self._setup_validate)
            except Exception as e:
                self._log(f"Error: {e}", "error")

        threading.Thread(target=_run, daemon=True).start()

    def _setup_update_cfg_label(self):
        if _CONFIG_FILE.exists():
            try:
                size = _CONFIG_FILE.stat().st_size
                mtime = time.strftime("%Y-%m-%d %H:%M:%S",
                                      time.localtime(_CONFIG_FILE.stat().st_mtime))
                self._setup_cfg_label.config(
                    text=f"  {_CONFIG_FILE.name}  |  {size} bytes  |  modified {mtime}",
                    foreground=self._c["fg"])
            except Exception:
                self._setup_cfg_label.config(
                    text=f"  {_CONFIG_FILE.name}  (exists)",
                    foreground=self._c["fg"])
        else:
            self._setup_cfg_label.config(
                text="  No config file (using defaults)",
                foreground=self._c["muted"])

    def _setup_reset_config(self):
        """Reset all settings to defaults and reload the UI."""
        if not messagebox.askyesno(
            "Reset Config",
            "Reset all settings to defaults?\n\n"
            "This will clear saved paths, port, quality, map, and custom scripts.",
            parent=self):
            return
        _save_config(_DEFAULT_CONFIG)
        self.carla_root.set(_DEFAULT_CONFIG["carla_root"])
        self.venv_path.set(_DEFAULT_CONFIG["venv_path"])
        self.carla_port.set(_DEFAULT_CONFIG["carla_port"])
        self.carla_quality.set(_DEFAULT_CONFIG["carla_quality"])
        self.carla_offscreen.set(_DEFAULT_CONFIG["carla_offscreen"])
        self.carla_map.set(_DEFAULT_CONFIG["carla_map"])
        self.auto_refresh.set(_DEFAULT_CONFIG["auto_refresh"])
        self._custom_scripts.clear()
        self._custom_dirs.clear()
        self._setup_update_cfg_label()
        self._log("Config reset to defaults.", "success")

    def _setup_delete_config(self):
        """Delete the config file from disk."""
        if not _CONFIG_FILE.exists():
            messagebox.showinfo("No File", "Config file does not exist.", parent=self)
            return
        if not messagebox.askyesno(
            "Delete Config",
            f"Delete {_CONFIG_FILE.name}?\n\n"
            "The app will use defaults. A new config file is created on close.",
            parent=self):
            return
        try:
            _CONFIG_FILE.unlink()
            self._log(f"Deleted {_CONFIG_FILE}", "success")
        except Exception as e:
            self._log(f"Error deleting config: {e}", "error")
        self._setup_update_cfg_label()

    def _setup_browse_carla(self):
        d = _pick_directory("Select CARLA Installation Directory",
                            initialdir=self.carla_root.get() or str(_HERE),
                            parent=self)
        if d:
            self.carla_root.set(d)

    def _setup_browse_venv(self):
        d = _pick_directory("Select Virtual Environment Directory",
                            initialdir=self.venv_path.get() or str(_HERE),
                            parent=self)
        if d:
            self.venv_path.set(d)

    def _setup_auto_detect_venv(self):
        """BFS scan from _HERE for directories containing bin/python or Scripts/python.exe."""
        found = []
        queue = [_HERE]
        visited = set()
        max_depth = 3

        while queue:
            current = queue.pop(0)
            # Track depth relative to _HERE
            try:
                depth = len(current.relative_to(_HERE).parts)
            except ValueError:
                continue
            if depth > max_depth:
                continue
            resolved = current.resolve()
            if resolved in visited:
                continue
            visited.add(resolved)

            # Check if this directory is a venv
            if _python_bin(current).exists():
                found.append(current)
                continue  # Don't recurse into venvs

            # Enqueue children
            try:
                for child in sorted(current.iterdir()):
                    if child.is_dir() and not child.name.startswith("."):
                        queue.append(child)
            except PermissionError:
                pass

        if found:
            self.venv_path.set(str(found[0]))
            if len(found) > 1:
                self._log(f"Auto-detect found {len(found)} venvs, using: {found[0]}")
                for p in found[1:]:
                    self._log(f"  also found: {p}")
            else:
                self._log(f"Auto-detect found venv at: {found[0]}", "success")
        else:
            self._log("Auto-detect: no virtual environment found nearby.", "error")

    def _setup_auto_detect_carla(self):
        """Scan _HERE and one level deep for CarlaUE4.sh/.exe."""
        exe_names = {"CarlaUE4.sh", "CarlaUE4.exe"}
        found = []

        # Check _HERE itself
        if any((_HERE / exe).exists() for exe in exe_names):
            found.append(_HERE)

        # Check immediate children
        try:
            for child in _HERE.iterdir():
                if not child.is_dir():
                    continue
                if any((child / exe).exists() for exe in exe_names):
                    found.append(child)
                # One level deeper for dirs with "carla" in the name
                elif "carla" in child.name.lower():
                    try:
                        for grandchild in child.iterdir():
                            if grandchild.is_dir() and any(
                                (grandchild / exe).exists() for exe in exe_names
                            ):
                                found.append(grandchild)
                    except PermissionError:
                        pass
        except PermissionError:
            pass

        if found:
            self.carla_root.set(str(found[0]))
            if len(found) > 1:
                self._log(f"Auto-detect found {len(found)} CARLA installs, using: {found[0]}")
                for p in found[1:]:
                    self._log(f"  also found: {p}")
            else:
                self._log(f"Auto-detect found CARLA at: {found[0]}", "success")
        else:
            self._log("Auto-detect: no CARLA installation found nearby.", "error")

    def _setup_create_venv(self):
        """Create a venv from the Setup tab."""
        venv = self.venv_path.get()
        if not venv:
            messagebox.showerror("No Path", "Set a venv path first.", parent=self)
            return

        vp = Path(venv)
        if _python_bin(vp).exists():
            answer = _venv_conflict_dialog(self, vp)
            if answer == "use":
                self._setup_validate()
                return
            elif answer == "cancel":
                return

        python = _find_python()
        if not python:
            messagebox.showerror(
                "Python Not Found",
                "Could not find a Python interpreter to create the venv.\n"
                "Install Python 3.10+ and try again.",
                parent=self)
            return

        self._log(f"Creating venv at {vp} ...")
        self.notebook.select(self._term_tab_idx)

        def _run():
            try:
                subprocess.run(
                    [python, "-m", "venv", str(vp), "--clear"],
                    check=True, timeout=180, capture_output=True, text=True)
                self._log(f"Venv created: {vp}", "success")
                self.after(0, self._setup_validate)
                self.after(0, self._refresh_venv_status)
            except subprocess.CalledProcessError as e:
                self._log(f"Failed: {e.stderr}", "error")
            except Exception as e:
                self._log(f"Error: {e}", "error")

        threading.Thread(target=_run, daemon=True).start()

    # -------------------------------------------------------------------
    # Tab 2 — CARLA Server
    # -------------------------------------------------------------------

    def _build_carla_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="  CARLA Server  ")

        # Launch options
        of = ttk.LabelFrame(tab, text="Launch Options")
        of.pack(fill="x", padx=12, pady=(12, 6))
        oi = ttk.Frame(of)
        oi.pack(fill="x", padx=8, pady=6)
        ttk.Label(oi, text="Port:").grid(row=0, column=0, sticky="w", padx=(0, 4))
        ttk.Entry(oi, textvariable=self.carla_port, width=8).grid(row=0, column=1, sticky="w")
        ttk.Label(oi, text="Quality:").grid(row=0, column=2, sticky="w", padx=(20, 4))
        ttk.Combobox(oi, textvariable=self.carla_quality, values=QUALITY_OPTIONS,
                     width=10, state="readonly").grid(row=0, column=3, sticky="w")
        ttk.Checkbutton(oi, text="Offscreen (no window)",
                        variable=self.carla_offscreen).grid(row=0, column=4, padx=(20, 0))

        # Map selection
        mf = ttk.LabelFrame(tab, text="Map")
        mf.pack(fill="x", padx=12, pady=6)
        mi = ttk.Frame(mf)
        mi.pack(fill="x", padx=8, pady=6)
        ttk.Label(mi, text="Map:").pack(side="left")
        self._map_combo = ttk.Combobox(mi, textvariable=self.carla_map,
                                       width=20)
        self._map_combo.pack(side="left", padx=6)
        ttk.Button(mi, text="Refresh Maps",
                   command=self._refresh_maps).pack(side="left", padx=(0, 6))
        ttk.Button(mi, text="Change Map",
                   command=self._change_map).pack(side="left", padx=(0, 6))
        self._map_status = ttk.Label(mi, text="", style="Status.TLabel")
        self._map_status.pack(side="left", padx=6)

        # Buttons
        bf = ttk.Frame(tab)
        bf.pack(fill="x", padx=12, pady=6)
        ttk.Button(bf, text="Start CARLA", style="Green.TButton",
                   command=self._start_carla).pack(side="left", padx=(0, 6))
        ttk.Button(bf, text="Stop All CARLA", style="Red.TButton",
                   command=self._stop_all_carla).pack(side="left", padx=(0, 6))
        ttk.Button(bf, text="Refresh",
                   command=self._refresh_process_table).pack(side="left", padx=(0, 6))
        ttk.Checkbutton(bf, text="Auto-refresh (3s)",
                        variable=self.auto_refresh).pack(side="left", padx=(12, 0))

        if IS_LINUX:
            ttk.Label(tab, text="  chmod +x applied automatically if needed.",
                      style="Status.TLabel",
                      foreground=self._c["yellow"]).pack(fill="x", padx=12, pady=(0, 4))

        # Unified process table
        pf2 = ttk.LabelFrame(tab, text="All Active Processes")
        pf2.pack(fill="both", expand=True, padx=12, pady=(4, 12))

        cols = ("Type", "Name", "PID", "Status", "Started", "Command")
        self.proc_tree = ttk.Treeview(pf2, columns=cols, show="headings", height=10)
        for col, w in [("Type", 70), ("Name", 120), ("PID", 60),
                       ("Status", 90), ("Started", 70), ("Command", 340)]:
            self.proc_tree.heading(col, text=col)
            anchor = "center" if col in ("Type", "PID", "Status", "Started") else "w"
            self.proc_tree.column(col, width=w, anchor=anchor)

        vsb = ttk.Scrollbar(pf2, orient="vertical", command=self.proc_tree.yview)
        self.proc_tree.configure(yscrollcommand=vsb.set)
        self.proc_tree.pack(side="left", fill="both", expand=True, padx=(6, 0), pady=6)
        vsb.pack(side="left", fill="y", pady=6)

        kb = ttk.Frame(pf2)
        kb.pack(side="left", fill="y", padx=8, pady=6)
        ttk.Button(kb, text="Kill Selected", style="Red.TButton",
                   command=self._kill_selected).pack(fill="x", pady=(0, 4))
        ttk.Button(kb, text="Kill All Scripts", style="Red.TButton",
                   command=self._kill_all_scripts).pack(fill="x", pady=(0, 4))
        ttk.Button(kb, text="Remove Exited",
                   command=self._remove_exited).pack(fill="x")

    # -------------------------------------------------------------------
    # Tab 3 — Virtual Env
    # -------------------------------------------------------------------

    def _build_venv_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="  Packages  ")

        # Status (read-only, path configured in Setup)
        sf = ttk.LabelFrame(tab, text="Virtual Environment")
        sf.pack(fill="x", padx=12, pady=(12, 6))
        self.venv_status_label = ttk.Label(sf, text="", style="Status.TLabel")
        self.venv_status_label.pack(fill="x", padx=8, pady=6)
        ar = ttk.Frame(sf)
        ar.pack(fill="x", padx=8, pady=(0, 6))
        self.activate_var = tk.StringVar()
        ttk.Entry(ar, textvariable=self.activate_var, state="readonly").pack(
            side="left", fill="x", expand=True, padx=(0, 6))
        ttk.Button(ar, text="Copy",
                   command=lambda: self._copy_clip(self.activate_var.get())).pack(side="left", padx=(0, 6))
        ttk.Button(ar, text="Delete Venv", style="Red.TButton",
                   command=self._delete_venv).pack(side="left")

        # Requirements install
        rf = ttk.LabelFrame(tab, text="Requirements")
        rf.pack(fill="x", padx=12, pady=6)

        rr = ttk.Frame(rf)
        rr.pack(fill="x", padx=8, pady=6)

        carla = self.carla_root.get()
        carla_root = Path(carla) if carla else _HERE
        req_candidates = [
            carla_root / "requirements.txt",
            carla_root / "PythonAPI" / "requirements.txt",
            carla_root / "PythonAPI" / "examples" / "requirements.txt",
        ]
        req_files = [str(p) for p in req_candidates]

        self.req_file_var = tk.StringVar(value=req_files[0] if req_files else "")
        ttk.Combobox(rr, textvariable=self.req_file_var,
                     values=req_files).pack(side="left", fill="x", expand=True, padx=(0, 6))
        ttk.Button(rr, text="Browse",
                   command=self._browse_req_file).pack(side="left", padx=(0, 6))
        ttk.Button(rr, text="Install",
                   command=self._install_requirements).pack(side="left")

        # Package management
        pf = ttk.LabelFrame(tab, text="Packages")
        pf.pack(fill="both", expand=True, padx=12, pady=(6, 12))

        pr = ttk.Frame(pf)
        pr.pack(fill="x", padx=8, pady=6)
        ttk.Label(pr, text="Package:").pack(side="left")
        self.single_pkg_var = tk.StringVar()
        ttk.Entry(pr, textvariable=self.single_pkg_var, width=35).pack(side="left", padx=6)
        ttk.Button(pr, text="Install",
                   command=self._install_single_pkg).pack(side="left", padx=(0, 4))
        ttk.Button(pr, text="Uninstall", style="Red.TButton",
                   command=self._uninstall_single_pkg).pack(side="left")

        br = ttk.Frame(pf)
        br.pack(fill="x", padx=8, pady=(0, 6))
        ttk.Button(br, text="pip list",
                   command=self._pip_list).pack(side="left", padx=(0, 6))
        ttk.Button(br, text="pip freeze",
                   command=self._pip_freeze).pack(side="left")

    # -------------------------------------------------------------------
    # Tab 4 — Scripts
    # -------------------------------------------------------------------

    def _build_scripts_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="  Scripts  ")

        # Source controls
        df = ttk.LabelFrame(tab, text="Script Sources")
        df.pack(fill="x", padx=12, pady=(12, 6))

        dr = ttk.Frame(df)
        dr.pack(fill="x", padx=8, pady=6)
        ttk.Label(dr, text="PythonAPI/examples:").pack(side="left")

        carla = self.carla_root.get()
        scripts_default = str(Path(carla) / "PythonAPI" / "examples") if carla else ""
        self.scripts_dir_var = tk.StringVar(value=scripts_default)
        ttk.Entry(dr, textvariable=self.scripts_dir_var).pack(
            side="left", padx=6, fill="x", expand=True)
        ttk.Button(dr, text="Browse", command=self._browse_scripts_dir).pack(side="left", padx=(0, 4))
        ttk.Button(dr, text="Reload", command=self._reload_scripts).pack(side="left")

        ar = ttk.Frame(df)
        ar.pack(fill="x", padx=8, pady=(0, 8))
        ttk.Button(ar, text="+ Add Script File",
                   command=self._add_custom_script).pack(side="left", padx=(0, 6))
        ttk.Button(ar, text="+ Add Script Directory",
                   command=self._add_custom_dir).pack(side="left", padx=(0, 6))
        ttk.Button(ar, text="Remove Selected", style="Red.TButton",
                   command=self._remove_selected_script).pack(side="left")

        # Script list
        lf = ttk.LabelFrame(tab, text="Available Scripts  (double-click to launch)")
        lf.pack(fill="both", expand=True, padx=12, pady=6)

        cols = ("Source", "Name", "Path")
        self.scripts_tree = ttk.Treeview(lf, columns=cols, show="headings", height=14)
        self.scripts_tree.heading("Source", text="Source")
        self.scripts_tree.heading("Name", text="Name")
        self.scripts_tree.heading("Path", text="Path")
        self.scripts_tree.column("Source", width=90, anchor="center")
        self.scripts_tree.column("Name", width=200)
        self.scripts_tree.column("Path", width=500)
        self.scripts_tree.bind("<Double-1>", lambda _: self._launch_selected_script())

        vsb = ttk.Scrollbar(lf, orient="vertical", command=self.scripts_tree.yview)
        self.scripts_tree.configure(yscrollcommand=vsb.set)
        self.scripts_tree.pack(side="left", fill="both", expand=True, padx=(6, 0), pady=6)
        vsb.pack(side="right", fill="y", pady=6)

        # Launch bar
        bf = ttk.Frame(tab)
        bf.pack(fill="x", padx=12, pady=(4, 10))
        ttk.Button(bf, text="Launch Selected", style="Green.TButton",
                   command=self._launch_selected_script).pack(side="left", padx=(0, 8))
        ttk.Label(bf, text="Args:").pack(side="left")
        self.script_args_var = tk.StringVar()
        ttk.Entry(bf, textvariable=self.script_args_var).pack(
            side="left", padx=6, fill="x", expand=True)

        self._reload_scripts()

    # -------------------------------------------------------------------
    # Tab 5 — Terminal
    # -------------------------------------------------------------------

    def _build_terminal_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="  Terminal  ")
        self._term_tab_idx = self.notebook.index("end") - 1

        # Toolbar
        tb = ttk.Frame(tab)
        tb.pack(fill="x", padx=8, pady=(8, 2))
        ttk.Label(tb, text="Output & Shell",
                  font=("Segoe UI", 10, "bold"),
                  foreground=self._c["accent"]).pack(side="left")
        ttk.Button(tb, text="Save Log", command=self._save_log).pack(side="right")
        ttk.Button(tb, text="Clear", command=self._clear_terminal).pack(side="right", padx=(0, 4))

        # Output area
        out_frame = ttk.Frame(tab)
        out_frame.pack(fill="both", expand=True, padx=8, pady=(2, 0))

        self.term_text = tk.Text(
            out_frame, wrap="word",
            bg=self._c["surface"], fg=self._c["fg"],
            font=("Consolas", 10),
            insertbackground=self._c["fg"],
            relief="flat", borderwidth=0,
            state="disabled",
        )
        vsb = ttk.Scrollbar(out_frame, orient="vertical", command=self.term_text.yview)
        self.term_text.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        self.term_text.pack(side="left", fill="both", expand=True)

        # Color tags
        self.term_text.tag_configure("ts", foreground="#6c7086")
        self.term_text.tag_configure("info", foreground=self._c["fg"])
        self.term_text.tag_configure("carla", foreground=self._c["cyan"])
        self.term_text.tag_configure("script", foreground=self._c["green"])
        self.term_text.tag_configure("shell", foreground=self._c["yellow"])
        self.term_text.tag_configure("error", foreground=self._c["red"])
        self.term_text.tag_configure("success", foreground=self._c["green"])
        self.term_text.tag_configure("cmd_in", foreground=self._c["accent"],
                                     font=("Consolas", 10, "bold"))

        # Input row
        inp = ttk.Frame(tab)
        inp.pack(fill="x", padx=8, pady=(4, 8))

        vp = self.venv_path.get()
        vp_name = Path(vp).name if vp else ""
        self._prompt_label = ttk.Label(
            inp, text=f"({vp_name}) $ " if vp_name else "$ ",
            foreground=self._c["green"],
            font=("Consolas", 11, "bold"),
            background=self._c["bg"])
        self._prompt_label.pack(side="left")

        self.term_input = ttk.Entry(inp, font=("Consolas", 10))
        self.term_input.pack(side="left", fill="x", expand=True, padx=(0, 6))
        self.term_input.bind("<Return>", self._run_terminal_cmd)
        self.term_input.bind("<Up>", self._hist_up)
        self.term_input.bind("<Down>", self._hist_down)
        self.term_input.bind("<Tab>", self._tab_complete)

        ttk.Button(inp, text="Run", style="Green.TButton",
                   command=self._run_terminal_cmd).pack(side="left")

        self.term_input.focus()

    # -------------------------------------------------------------------
    # Terminal helpers
    # -------------------------------------------------------------------

    def _log(self, message: str, tag: str = "info"):
        def _write():
            self.term_text.config(state="normal")
            ts = time.strftime("%H:%M:%S")
            self.term_text.insert("end", f"[{ts}] ", "ts")
            self.term_text.insert("end", message + "\n", tag)
            self.term_text.see("end")
            self.term_text.config(state="disabled")
        if threading.current_thread() is threading.main_thread():
            _write()
        else:
            self.after(0, _write)

    def _clear_terminal(self):
        self.term_text.config(state="normal")
        self.term_text.delete("1.0", "end")
        self.term_text.config(state="disabled")

    def _save_log(self):
        f = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All", "*.*")],
            initialfile=f"launcher_log_{time.strftime('%Y%m%d_%H%M%S')}.txt")
        if f:
            Path(f).write_text(self.term_text.get("1.0", "end"))
            self._log(f"Log saved to {f}", "success")

    def _venv_env(self) -> dict:
        vp = Path(self.venv_path.get())
        env = os.environ.copy()
        bin_dir = str(vp / ("Scripts" if IS_WINDOWS else "bin"))
        env["PATH"] = bin_dir + os.pathsep + env.get("PATH", "")
        env["VIRTUAL_ENV"] = str(vp)
        env.pop("PYTHONHOME", None)
        return env

    def _run_terminal_cmd(self, event=None):
        cmd = self.term_input.get().strip()
        if not cmd:
            return
        self.term_input.delete(0, "end")

        if not self._term_history or self._term_history[-1] != cmd:
            self._term_history.append(cmd)
        self._term_hist_idx = -1

        self._log(f"$ {cmd}", "cmd_in")
        self.notebook.select(self._term_tab_idx)

        def _run():
            try:
                proc = subprocess.Popen(
                    cmd, shell=True,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, env=self._venv_env(), cwd=str(_HERE),
                )
                self.registry.add(ProcessEntry("shell", "Shell", proc, cmd))
                self.after(0, self._refresh_process_table)
                for line in proc.stdout:
                    line = line.rstrip()
                    if line:
                        self._log(line, "shell")
                proc.wait()
                self.after(0, self._refresh_process_table)
            except Exception as e:
                self._log(f"Error: {e}", "error")

        threading.Thread(target=_run, daemon=True).start()

    def _hist_up(self, event=None):
        if not self._term_history:
            return
        if self._term_hist_idx == -1:
            self._term_hist_idx = len(self._term_history) - 1
        else:
            self._term_hist_idx = max(0, self._term_hist_idx - 1)
        self.term_input.delete(0, "end")
        self.term_input.insert(0, self._term_history[self._term_hist_idx])

    def _hist_down(self, event=None):
        if self._term_hist_idx == -1:
            return
        self._term_hist_idx += 1
        self.term_input.delete(0, "end")
        if self._term_hist_idx >= len(self._term_history):
            self._term_hist_idx = -1
        else:
            self.term_input.insert(0, self._term_history[self._term_hist_idx])

    def _tab_complete(self, event=None):
        text = self.term_input.get()
        words = text.split()
        if not words:
            return "break"
        last = words[-1]
        try:
            base = Path(last).parent
            prefix = Path(last).name
            matches = [str(p) for p in (base if str(base) != "." else Path(".")).iterdir()
                       if p.name.startswith(prefix)]
            if len(matches) == 1:
                words[-1] = matches[0]
                self.term_input.delete(0, "end")
                self.term_input.insert(0, " ".join(words))
        except Exception:
            pass
        return "break"

    def _copy_clip(self, text: str):
        self.clipboard_clear()
        self.clipboard_append(text)
        self._log(f"Copied: {text}")

    # -------------------------------------------------------------------
    # CARLA actions
    # -------------------------------------------------------------------

    def _refresh_maps(self):
        """Query the running CARLA server for available maps."""
        python = _python_bin(Path(self.venv_path.get()))
        if not python.exists():
            messagebox.showerror("No venv", "Create a venv first.")
            return

        port = self.carla_port.get()
        self._map_status.config(text="Querying...", foreground=self._c["yellow"])

        script = (
            "import carla, json;"
            f"c = carla.Client('localhost', {port});"
            "c.set_timeout(5.0);"
            "maps = sorted(c.get_available_maps());"
            "print(json.dumps(maps))"
        )

        def _run():
            try:
                proc = subprocess.run(
                    [str(python), "-c", script],
                    capture_output=True, text=True, timeout=15,
                    env=self._venv_env(),
                )
                if proc.returncode == 0:
                    import json as _json
                    raw_maps = _json.loads(proc.stdout.strip())
                    # Strip /Game/Carla/Maps/ prefix if present
                    clean = [m.rsplit("/", 1)[-1] for m in raw_maps]
                    self.after(0, lambda: self._apply_maps(clean))
                else:
                    err = proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else "Unknown error"
                    self._log(f"Map query failed: {err}", "error")
                    self.after(0, lambda: self._map_status.config(
                        text="Failed — is CARLA running?", foreground=self._c["red"]))
            except subprocess.TimeoutExpired:
                self._log("Map query timed out — is CARLA running?", "error")
                self.after(0, lambda: self._map_status.config(
                    text="Timeout", foreground=self._c["red"]))
            except Exception as e:
                self._log(f"Map query error: {e}", "error")
                self.after(0, lambda: self._map_status.config(
                    text="Error", foreground=self._c["red"]))

        threading.Thread(target=_run, daemon=True).start()

    def _apply_maps(self, maps: list[str]):
        """Update the map combobox with queried maps."""
        self._map_combo.config(values=maps)
        current = self.carla_map.get()
        if current not in maps and maps:
            self.carla_map.set(maps[0])
        self._map_status.config(
            text=f"{len(maps)} maps available", foreground=self._c["green"])
        self._log(f"Found {len(maps)} maps: {', '.join(maps)}", "carla")

    def _change_map(self):
        """Connect to a running CARLA server and load the selected map."""
        python = _python_bin(Path(self.venv_path.get()))
        if not python.exists():
            messagebox.showerror("No venv", "Create a venv first.")
            return

        map_name = self.carla_map.get()
        if not map_name:
            messagebox.showinfo("No Map", "Select or type a map name first.")
            return
        port = self.carla_port.get()

        self._map_status.config(text="Loading...", foreground=self._c["yellow"])
        self._log(f"Changing map to {map_name} on port {port}...", "carla")
        self.notebook.select(self._term_tab_idx)

        script = (
            "import carla, sys;"
            f"c = carla.Client('localhost', {port});"
            "c.set_timeout(10.0);"
            f"c.load_world('{map_name}');"
            f"print('Map changed to {map_name}')"
        )

        def _run():
            try:
                proc = subprocess.run(
                    [str(python), "-c", script],
                    capture_output=True, text=True, timeout=30,
                    env=self._venv_env(),
                )
                if proc.returncode == 0:
                    self._log(f"Map changed to {map_name}", "success")
                    self.after(0, lambda: self._map_status.config(
                        text=f"{map_name}", foreground=self._c["green"]))
                else:
                    err = proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else "Unknown error"
                    self._log(f"Map change failed: {err}", "error")
                    self.after(0, lambda: self._map_status.config(
                        text="Failed", foreground=self._c["red"]))
            except subprocess.TimeoutExpired:
                self._log("Map change timed out — is CARLA running?", "error")
                self.after(0, lambda: self._map_status.config(
                    text="Timeout", foreground=self._c["red"]))
            except Exception as e:
                self._log(f"Map change error: {e}", "error")
                self.after(0, lambda: self._map_status.config(
                    text="Error", foreground=self._c["red"]))

        threading.Thread(target=_run, daemon=True).start()

    def _start_carla(self):
        carla_root = Path(self.carla_root.get())
        exe = _carla_executable(carla_root)

        if not exe.exists():
            messagebox.showerror("Not Found",
                                 f"CARLA executable not found:\n{exe}\n\n"
                                 "Check the CARLA installation path.")
            return

        if not IS_WINDOWS and not os.access(exe, os.X_OK):
            self._log(f"Setting +x on {exe.name}")
            try:
                os.chmod(exe, os.stat(exe).st_mode | 0o755)
            except PermissionError:
                try:
                    subprocess.run(["sudo", "chmod", "+x", str(exe)],
                                   check=True, timeout=30)
                    self._log("chmod +x OK via sudo")
                except Exception as e:
                    messagebox.showerror("Permission Error",
                                         f"Cannot set +x:\n{e}\n\n"
                                         f"Run manually:\n  sudo chmod +x {exe}")
                    return

        port = self.carla_port.get()
        quality = self.carla_quality.get()
        cmd = [str(exe), "-carla-port", str(port),
               "-quality-level", quality, "-nosound"]
        if self.carla_offscreen.get():
            cmd.append("-RenderOffScreen")

        self._log(f"Starting CARLA: {' '.join(cmd)}", "carla")
        self.notebook.select(self._term_tab_idx)

        def _run():
            try:
                env = os.environ.copy()
                kwargs = dict(env=env,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                if IS_WINDOWS:
                    kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
                elif IS_LINUX:
                    kwargs["preexec_fn"] = os.setsid
                proc = subprocess.Popen(cmd, **kwargs)
                self.registry.add(ProcessEntry("CARLA", "CARLA", proc, " ".join(cmd)))
                self._log(f"CARLA started PID={proc.pid} port={port}", "carla")
                self.after(0, self._refresh_process_table)

                for raw in proc.stdout:
                    line = raw.decode("utf-8", errors="replace").rstrip()
                    if line:
                        self._log(f"[CARLA] {line}", "carla")

                proc.wait()
                self._log(f"CARLA PID={proc.pid} exited ({proc.returncode})", "carla")
                self.after(0, self._refresh_process_table)

            except Exception as e:
                self._log(f"Error starting CARLA: {e}", "error")

        threading.Thread(target=_run, daemon=True).start()

    def _stop_all_carla(self):
        sys_procs = _find_carla_processes()
        reg_procs = [e for e in self.registry.all_entries()
                     if e.proc_type == "CARLA" and e.is_running]
        all_pids = {p[0] for p in sys_procs} | {e.pid for e in reg_procs}

        if not all_pids:
            self._log("No CARLA processes found.")
            return
        if not messagebox.askyesno("Confirm", f"Stop {len(all_pids)} CARLA process(es)?"):
            return

        for pid in all_pids:
            try:
                _kill_pid(pid)
                self._log(f"Killed CARLA PID={pid}", "carla")
            except Exception as ex:
                self._log(f"Failed PID={pid}: {ex}", "error")

        self.after(1500, self._refresh_process_table)

    def _refresh_process_table(self):
        for item in self.proc_tree.get_children():
            self.proc_tree.delete(item)

        reg_pids = {e.pid for e in self.registry.all_entries()}

        for pid, cmd in _find_carla_processes():
            if pid not in reg_pids:
                self.proc_tree.insert("", "end", iid=f"sys_{pid}",
                    values=("CARLA", "CarlaUE4", pid, "Running", "\u2014", cmd[:80]))

        for e in self.registry.all_entries():
            self.proc_tree.insert("", "end", iid=str(e.pid),
                values=(e.proc_type, e.name, e.pid,
                        e.status, e.started, e.cmd[:80]))

        if not self.proc_tree.get_children():
            self.proc_tree.insert("", "end",
                values=("\u2014", "\u2014", "\u2014", "No processes", "\u2014", ""))

    def _auto_refresh(self):
        if self.auto_refresh.get():
            self._refresh_process_table()
        self.after(3000, self._auto_refresh)

    def _kill_selected(self):
        sel = self.proc_tree.selection()
        if not sel:
            messagebox.showinfo("Select", "Select a process row first.")
            return
        vals = self.proc_tree.item(sel[0], "values")
        try:
            pid = int(vals[2])
        except (ValueError, IndexError):
            return
        if not messagebox.askyesno("Confirm", f"Kill PID {pid}?"):
            return
        try:
            _kill_pid(pid)
            self._log(f"Killed PID={pid}", "error")
        except Exception as e:
            self._log(f"Failed: {e}", "error")
        self.after(1000, self._refresh_process_table)

    def _kill_all_scripts(self):
        entries = [e for e in self.registry.all_entries()
                   if e.proc_type == "Script" and e.is_running]
        if not entries:
            self._log("No script processes running.")
            return
        if not messagebox.askyesno("Confirm", f"Kill {len(entries)} script process(es)?"):
            return
        for e in entries:
            try:
                _kill_pid(e.pid)
                self._log(f"Killed {e.name} PID={e.pid}", "error")
            except Exception as ex:
                self._log(f"Failed {e.pid}: {ex}", "error")
        self.after(1000, self._refresh_process_table)

    def _remove_exited(self):
        for e in self.registry.all_entries():
            if not e.is_running:
                self.registry.remove(e.pid)
        self._refresh_process_table()

    # -------------------------------------------------------------------
    # Venv actions
    # -------------------------------------------------------------------

    def _browse_req_file(self):
        f = filedialog.askopenfilename(
            title="Select requirements.txt",
            filetypes=[("Text files", "*.txt"), ("All", "*.*")])
        if f:
            self.req_file_var.set(f)

    def _refresh_venv_status(self):
        vp = Path(self.venv_path.get())
        python = _python_bin(vp)
        self.activate_var.set(_activate_cmd(vp))

        if hasattr(self, "_prompt_label"):
            self._prompt_label.config(
                text=f"({vp.name}) $ " if python.exists() else "$ ")

        if python.exists():
            try:
                ver = subprocess.check_output(
                    [str(python), "--version"], text=True, timeout=5,
                    stderr=subprocess.STDOUT).strip()
            except Exception:
                ver = "unknown"
            self.venv_status_label.config(
                text=f"Active  |  {ver}  |  {python}",
                foreground=self._c["green"])
        else:
            self.venv_status_label.config(
                text=f"Not found: {vp}",
                foreground=self._c["red"])

    def _delete_venv(self):
        vp = Path(self.venv_path.get())
        if not vp.exists():
            messagebox.showinfo("Not found", f"No venv at {vp}")
            return
        if not messagebox.askyesno("Confirm", f"Delete {vp}? Cannot be undone."):
            return

        def _run():
            import shutil
            try:
                shutil.rmtree(vp)
                self._log("Venv deleted.", "error")
                self.after(0, self._refresh_venv_status)
                self.after(0, self._setup_validate)
            except Exception as e:
                self._log(f"Error: {e}", "error")

        threading.Thread(target=_run, daemon=True).start()

    def _install_requirements(self):
        pip = _pip_bin(Path(self.venv_path.get()))
        if not pip.exists():
            messagebox.showerror("No venv", "Create a venv first.")
            return
        req = Path(self.req_file_var.get())
        if not req.exists():
            messagebox.showerror("Not found", f"File not found:\n{req}")
            return
        self._run_pip([str(pip), "install", "-r", str(req)],
                      f"Install from {req.name}")

    def _pip_freeze(self):
        pip = _pip_bin(Path(self.venv_path.get()))
        if not pip.exists():
            messagebox.showerror("No venv", "Create a venv first."); return
        self._run_pip([str(pip), "freeze"], "pip freeze")

    def _pip_list(self):
        pip = _pip_bin(Path(self.venv_path.get()))
        if not pip.exists():
            messagebox.showerror("No venv", "Create a venv first."); return
        self._run_pip([str(pip), "list"], "pip list")

    def _install_single_pkg(self):
        pkg = self.single_pkg_var.get().strip()
        if not pkg: return
        pip = _pip_bin(Path(self.venv_path.get()))
        if not pip.exists():
            messagebox.showerror("No venv", "Create a venv first."); return
        self._run_pip([str(pip), "install", pkg], f"Install {pkg}")

    def _uninstall_single_pkg(self):
        pkg = self.single_pkg_var.get().strip()
        if not pkg: return
        pip = _pip_bin(Path(self.venv_path.get()))
        if not pip.exists():
            messagebox.showerror("No venv", "Create a venv first."); return
        if not messagebox.askyesno("Confirm", f"Uninstall {pkg}?"): return
        self._run_pip([str(pip), "uninstall", "-y", pkg], f"Uninstall {pkg}")

    def _run_pip(self, cmd: list[str], label: str):
        self._log(f"--- {label} ---")
        self.notebook.select(self._term_tab_idx)

        def _run():
            try:
                proc = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                for line in proc.stdout:
                    line = line.rstrip()
                    if line:
                        self._log(line, "shell")
                proc.wait()
                ok = proc.returncode == 0
                self._log(f"--- {label}: {'OK' if ok else f'failed ({proc.returncode})'} ---",
                          "success" if ok else "error")
            except Exception as e:
                self._log(f"Error: {e}", "error")

        threading.Thread(target=_run, daemon=True).start()

    # -------------------------------------------------------------------
    # Scripts tab actions
    # -------------------------------------------------------------------

    def _browse_scripts_dir(self):
        d = _pick_directory("Select scripts directory",
                            initialdir=self.scripts_dir_var.get() or str(_HERE),
                            parent=self)
        if d:
            self.scripts_dir_var.set(d)
            self._reload_scripts()

    def _reload_scripts(self):
        for item in self.scripts_tree.get_children():
            self.scripts_tree.delete(item)

        # PythonAPI/examples
        for p in _scan_scripts(Path(self.scripts_dir_var.get())):
            self.scripts_tree.insert("", "end", iid=f"ex_{p}",
                values=("examples", p.name, str(p)))

        # Custom directories
        for d in self._custom_dirs:
            for p in _scan_scripts(d):
                self.scripts_tree.insert("", "end", iid=f"dir_{p}",
                    values=(d.name[:12], p.name, str(p)))

        # Custom individual files
        for p in self._custom_scripts:
            self.scripts_tree.insert("", "end", iid=f"cust_{p}",
                values=("custom", p.name, str(p)))

    def _add_custom_script(self):
        f = filedialog.askopenfilename(
            title="Add Script",
            filetypes=[("Python files", "*.py"), ("All", "*.*")])
        if f:
            p = Path(f)
            if p not in self._custom_scripts:
                self._custom_scripts.append(p)
            self._reload_scripts()

    def _add_custom_dir(self):
        d = _pick_directory("Add Script Directory",
                            initialdir=str(_HERE), parent=self)
        if d:
            p = Path(d)
            if p not in self._custom_dirs:
                self._custom_dirs.append(p)
            self._reload_scripts()

    def _remove_selected_script(self):
        sel = self.scripts_tree.selection()
        if not sel:
            return
        vals = self.scripts_tree.item(sel[0], "values")
        if not vals:
            return
        source, _, path_str = vals
        p = Path(path_str)
        if source == "custom":
            self._custom_scripts = [x for x in self._custom_scripts if x != p]
        elif source != "examples":
            self._custom_dirs = [x for x in self._custom_dirs if x != p.parent]
        else:
            messagebox.showinfo("Info",
                "Cannot remove built-in example scripts.\n"
                "Change the examples directory path instead.")
            return
        self._reload_scripts()

    def _launch_selected_script(self):
        sel = self.scripts_tree.selection()
        if not sel:
            messagebox.showinfo("Select", "Select a script first.")
            return
        vals = self.scripts_tree.item(sel[0], "values")
        if not vals:
            return
        _, name, path_str = vals
        self._launch_script(Path(path_str), name, self.script_args_var.get().strip())

    def _launch_script(self, script_path: Path, name: str, args: str = ""):
        python = _python_bin(Path(self.venv_path.get()))
        if not python.exists():
            messagebox.showerror("No venv", "Create a venv first.")
            return
        if not script_path.exists():
            messagebox.showerror("Not found", f"Script not found:\n{script_path}")
            return

        cmd_list = [str(python), str(script_path)]
        if args:
            try:
                cmd_list.extend(shlex.split(args))
            except ValueError:
                cmd_list.extend(args.split())

        self._log(f"Launching {name}: {' '.join(cmd_list)}", "script")
        self.notebook.select(self._term_tab_idx)

        def _run():
            try:
                env = self._venv_env()
                kwargs = dict(stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, cwd=str(script_path.parent), env=env)
                if IS_LINUX:
                    kwargs["preexec_fn"] = os.setsid
                elif IS_WINDOWS:
                    kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP

                proc = subprocess.Popen(cmd_list, **kwargs)
                self.registry.add(ProcessEntry(name, "Script", proc, " ".join(cmd_list)))
                self._log(f"[{name}] PID={proc.pid}", "script")
                self.after(0, self._refresh_process_table)

                for line in proc.stdout:
                    line = line.rstrip()
                    if line:
                        self._log(f"[{name}] {line}", "script")

                proc.wait()
                self._log(f"[{name}] exited ({proc.returncode})", "script")
                self.after(0, self._refresh_process_table)

            except Exception as e:
                self._log(f"[{name}] Error: {e}", "error")

        threading.Thread(target=_run, daemon=True).start()


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

def main():
    # Suppress stdout/stderr when launched without a terminal (e.g. from
    # a .desktop file or .pyw on Windows) so stray prints don't cause errors.
    if not sys.stdout or not hasattr(sys.stdout, "fileno"):
        sys.stdout = open(os.devnull, "w")
        sys.stderr = open(os.devnull, "w")
    else:
        try:
            if not os.isatty(sys.stdout.fileno()):
                sys.stdout = open(os.devnull, "w")
                sys.stderr = open(os.devnull, "w")
        except (OSError, ValueError):
            sys.stdout = open(os.devnull, "w")
            sys.stderr = open(os.devnull, "w")

    app = CarlaLauncher()
    app.mainloop()


if __name__ == "__main__":
    main()
