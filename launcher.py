#!/usr/bin/env python3
"""
HondaAI-CARLA-Launcher — Tkinter GUI for managing CARLA simulator,
virtual environments, and project scripts.

On first launch a setup wizard runs to configure paths and create a venv.
Config is saved next to the executable as launcher_config.json.
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
# Path detection — works whether launcher is inside CARLA_x.x.x or not
# ---------------------------------------------------------------------------

# When frozen by PyInstaller, __file__ lives in a temp extraction dir that is
# wiped on exit. Use the actual executable's directory so config persists.
_HERE = (Path(sys.executable).resolve().parent
         if getattr(sys, "frozen", False)
         else Path(__file__).resolve().parent)


def _detect_carla_root() -> Path:
    """Walk up from launcher location to find CARLA install root."""
    for candidate in [_HERE, _HERE.parent]:
        if (candidate / "CarlaUE4.sh").exists() or (candidate / "CarlaUE4.exe").exists():
            return candidate
    return _HERE


CARLA_ROOT_DEFAULT = _detect_carla_root()
SCRIPTS_DEFAULT    = CARLA_ROOT_DEFAULT / "PythonAPI" / "examples"
VENV_DEFAULT       = CARLA_ROOT_DEFAULT / "venv"
CONFIG_FILE        = _HERE / "launcher_config.json"

_req_candidates = [
    CARLA_ROOT_DEFAULT / "requirements.txt",
    CARLA_ROOT_DEFAULT / "PythonAPI" / "requirements.txt",
    CARLA_ROOT_DEFAULT / "PythonAPI" / "examples" / "requirements.txt",
]
REQUIREMENTS_FILES_DEFAULT = [str(p) for p in _req_candidates]

IS_WINDOWS = platform.system() == "Windows"
IS_LINUX   = platform.system() == "Linux"
QUALITY_OPTIONS = ["Low", "Medium", "High", "Epic"]


# ---------------------------------------------------------------------------
# Config persistence
# ---------------------------------------------------------------------------

def _load_config() -> dict:
    if CONFIG_FILE.exists():
        try:
            return json.loads(CONFIG_FILE.read_text())
        except Exception:
            pass
    return {}


def _save_config(data: dict):
    try:
        CONFIG_FILE.write_text(json.dumps(data, indent=2))
    except Exception:
        pass


def _needs_setup() -> bool:
    """Return True if first-run setup should be shown."""
    cfg = _load_config()
    if cfg.get("setup_complete", False):
        return False
    # Backward compat: config written before the wizard existed already has
    # carla_root and venv_path — treat as complete and migrate the file.
    if cfg.get("carla_root") and cfg.get("venv_path"):
        cfg["setup_complete"] = True
        _save_config(cfg)
        return False
    return True


def _pick_directory(title: str, initialdir: str = "", parent=None) -> str | None:
    """Open a directory picker using the OS native dialog where possible,
    falling back to tkinter's dialog if unavailable."""
    start = initialdir or str(Path.home())
    system = platform.system()

    if system == "Linux":
        # Try zenity (GNOME/generic) then kdialog (KDE)
        try:
            r = subprocess.run(
                ["zenity", "--file-selection", "--directory",
                 f"--title={title}", f"--filename={start}/"],
                capture_output=True, text=True, timeout=120,
            )
            if r.returncode == 0:
                return r.stdout.strip() or None
            if r.returncode == 1:   # user cancelled
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

    elif system == "Windows":
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

    elif system == "Darwin":
        script = (
            f'POSIX path of (choose folder with prompt "{title}" '
            f'default location POSIX file "{start}")'
        )
        try:
            r = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True, text=True, timeout=120,
            )
            if r.returncode == 0:
                return r.stdout.strip().rstrip("/") or None
        except Exception:
            pass

    # Fallback: tkinter dialog
    return filedialog.askdirectory(title=title, initialdir=start, parent=parent) or None


def _venv_conflict_dialog(parent, venv_path: Path) -> str:
    """Show a modal dialog when a venv already exists.
    Returns 'use', 'override', or 'cancel'."""
    result = tk.StringVar(value="cancel")
    dlg = tk.Toplevel(parent)
    dlg.title("Venv Already Exists")
    dlg.configure(bg="#1e1e2e")
    dlg.resizable(False, False)
    dlg.transient(parent)          # stays above parent, minimizes with it
    dlg.grab_set()
    dlg.focus_force()
    dlg.protocol("WM_DELETE_WINDOW", dlg.destroy)

    # Center over parent
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
    style.configure("UseVenv.TButton",      font=("Segoe UI", 10), padding=(10, 6))
    style.configure("OverrideVenv.TButton", font=("Segoe UI", 10), padding=(10, 6),
                    foreground=maroon)
    style.configure("CancelVenv.TButton",   font=("Segoe UI", 10), padding=(10, 6))

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


def _find_carla_dirs(root: Path) -> list[Path]:
    """Scan root (and one level deep) for CARLA installation directories."""
    found = []
    exe_names = {"CarlaUE4.sh", "CarlaUE4.exe"}
    try:
        for child in root.iterdir():
            if not child.is_dir():
                continue
            if any((child / exe).exists() for exe in exe_names):
                found.append(child)
            elif "carla" in child.name.lower():
                # Check one level deeper
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
    return sorted(found)


# ---------------------------------------------------------------------------
# Setup Wizard
# ---------------------------------------------------------------------------

class SetupWizard(tk.Toplevel):
    """
    First-run wizard: collects project root, CARLA path, creates venv.
    Calls on_complete(config_dict) when finished.
    """

    STEPS = ["Welcome", "Project Root", "CARLA Install", "Venv Setup", "Done"]

    def __init__(self, parent: tk.Tk, on_complete):
        super().__init__(parent)
        self.on_complete = on_complete
        self.title("HondaAI-CARLA-Launcher — Setup")
        self.geometry("600x480")
        self.minsize(600, 480)
        self.resizable(True, True)
        self.configure(bg="#1e1e2e")
        self.grab_set()          # modal
        self.focus_force()
        self.protocol("WM_DELETE_WINDOW", self._cancel)

        # Wizard state
        self._step       = 0
        self._proj_root  = tk.StringVar(value=str(Path.home()))
        self._carla_root = tk.StringVar()
        self._venv_path  = tk.StringVar()
        self._carla_choices: list[Path] = []

        # Colors (match main app)
        self._c = dict(
            bg="#1e1e2e", fg="#cdd6f4", accent="#89b4fa",
            surface="#313244", green="#a6e3a1", red="#f38ba8",
            yellow="#f9e2af", muted="#6c7086",
        )

        self._build_styles()
        self._build_chrome()
        self._show_step()

    # -----------------------------------------------------------------------
    def _build_styles(self):
        s = ttk.Style(self)
        c = self._c
        s.configure("W.TFrame",  background=c["bg"])
        s.configure("W.TLabel",  background=c["bg"], foreground=c["fg"],
                    font=("Segoe UI", 10))
        s.configure("WH.TLabel", background=c["bg"], foreground=c["accent"],
                    font=("Segoe UI", 16, "bold"))
        s.configure("WS.TLabel", background=c["bg"], foreground=c["muted"],
                    font=("Segoe UI", 9))
        s.configure("WB.TLabel", background=c["bg"], foreground=c["fg"],
                    font=("Segoe UI", 10, "bold"))
        s.configure("W.TButton", font=("Segoe UI", 10), padding=(10, 6))
        s.configure("WG.TButton", font=("Segoe UI", 10, "bold"), padding=(14, 6),
                    foreground=c["green"])
        s.configure("WR.TButton", font=("Segoe UI", 10), padding=(10, 6),
                    foreground=c["red"])
        s.configure("W.TEntry",  fieldbackground=c["surface"], foreground=c["fg"],
                    font=("Segoe UI", 10))
        s.configure("W.TCombobox", fieldbackground=c["surface"], foreground=c["fg"])
        s.configure("W.Horizontal.TProgressbar",
                    troughcolor=c["surface"], background=c["accent"])

    def _build_chrome(self):
        c = self._c
        # Step indicator strip
        self._step_bar = tk.Frame(self, bg=c["surface"], height=4)
        self._step_bar.pack(fill="x", side="top")
        self._step_fill = tk.Frame(self._step_bar, bg=c["accent"], height=4)
        self._step_fill.place(x=0, y=0, relheight=1)

        # Step label row
        label_row = tk.Frame(self, bg=c["bg"])
        label_row.pack(fill="x", padx=24, pady=(14, 0))
        self._step_label = tk.Label(label_row, text="", bg=c["bg"],
                                    fg=c["muted"], font=("Segoe UI", 9))
        self._step_label.pack(side="right")

        # Content area
        self._content = tk.Frame(self, bg=c["bg"])
        self._content.pack(fill="both", expand=True, padx=32, pady=(8, 0))

        # Bottom nav
        nav = tk.Frame(self, bg=c["surface"])
        nav.pack(fill="x", side="bottom")
        inner = tk.Frame(nav, bg=c["surface"])
        inner.pack(fill="x", padx=20, pady=12)

        self._btn_back   = ttk.Button(inner, text="← Back",   style="W.TButton",
                                      command=self._back)
        self._btn_cancel = ttk.Button(inner, text="Cancel",    style="WR.TButton",
                                      command=self._cancel)
        self._btn_next   = ttk.Button(inner, text="Next →",    style="W.TButton",
                                      command=self._next)
        self._btn_finish = ttk.Button(inner, text="Finish  ✓", style="WG.TButton",
                                      command=self._finish)

        self._btn_cancel.pack(side="left")
        self._btn_back.pack(side="left", padx=(8, 0))
        self._btn_finish.pack(side="right")
        self._btn_next.pack(side="right", padx=(0, 8))

    # -----------------------------------------------------------------------
    def _update_chrome(self):
        n     = len(self.STEPS)
        frac  = (self._step + 1) / n
        self._step_fill.place(relwidth=frac)
        self._step_label.config(
            text=f"Step {self._step + 1} of {n}  —  {self.STEPS[self._step]}")
        # Button visibility
        self._btn_back.config(state="normal" if self._step > 0 else "disabled")
        last = self._step == n - 1
        if last:
            self._btn_next.pack_forget()
            self._btn_finish.pack(side="right")
        else:
            self._btn_finish.pack_forget()
            self._btn_next.pack(side="right", padx=(0, 8))

    def _clear_content(self):
        for w in self._content.winfo_children():
            w.destroy()

    def _show_step(self):
        self._clear_content()
        self._update_chrome()
        steps = [
            self._step_welcome,
            self._step_project_root,
            self._step_carla,
            self._step_venv,
            self._step_done,
        ]
        steps[self._step]()

    def _back(self):
        if self._step > 0:
            self._step -= 1
            self._show_step()

    def _next(self):
        if not self._validate_step():
            return
        self._step += 1
        self._show_step()

    def _cancel(self):
        if messagebox.askyesno("Cancel Setup",
                               "Cancel setup? The launcher will close.",
                               parent=self):
            self.master.destroy()

    # -----------------------------------------------------------------------
    def _validate_step(self) -> bool:
        if self._step == 1:   # Project root
            p = Path(self._proj_root.get())
            if not p.exists():
                messagebox.showerror("Invalid Path",
                                     f"Directory does not exist:\n{p}", parent=self)
                return False
        if self._step == 2:   # CARLA
            if not self._carla_root.get():
                messagebox.showerror("Required",
                                     "Please select or enter a CARLA directory.", parent=self)
                return False
            exe = _carla_executable(Path(self._carla_root.get()))
            if not exe.exists():
                if not messagebox.askyesno(
                    "Not Found",
                    f"CARLA executable not found in:\n{self._carla_root.get()}\n\n"
                    "Continue anyway?", parent=self):
                    return False
        return True

    # -----------------------------------------------------------------------
    # Step pages
    # -----------------------------------------------------------------------

    def _lbl(self, parent, text, style="W.TLabel", **kw):
        return ttk.Label(parent, text=text, style=style, wraplength=520, **kw)

    def _step_welcome(self):
        c  = self._c
        f  = self._content
        tk.Label(f, text="HondaAI-CARLA-Launcher",
                 bg=c["bg"], fg=c["accent"],
                 font=("Segoe UI", 20, "bold")).pack(anchor="w", pady=(16, 4))
        tk.Label(f, text="First-run setup",
                 bg=c["bg"], fg=c["muted"],
                 font=("Segoe UI", 11)).pack(anchor="w")

        ttk.Separator(f).pack(fill="x", pady=16)

        for line in [
            "This wizard will configure the launcher in three quick steps:",
            "  1.  Choose a project root folder. This folder will contain\n"
            "       CARLA and any subdirectories for scripts and configs.",
            "  2.  Point to your CARLA installation directory.\n"
            "       (Auto-detected if found inside the project root.)",
            "  3.  We will initialize a Python virtual environment so\n"
            "       scripts run with the correct dependencies.",
        ]:
            tk.Label(f, text=line, bg=c["bg"], fg=c["fg"],
                     font=("Segoe UI", 10), justify="left",
                     wraplength=520).pack(anchor="w", pady=2)

        ttk.Separator(f).pack(fill="x", pady=16)
        tk.Label(f, text="Click Next → to begin.",
                 bg=c["bg"], fg=c["muted"],
                 font=("Segoe UI", 10, "italic")).pack(anchor="w")

    def _step_project_root(self):
        c = self._c
        f = self._content

        tk.Label(f, text="Project Root",
                 bg=c["bg"], fg=c["accent"],
                 font=("Segoe UI", 16, "bold")).pack(anchor="w", pady=(16, 4))
        tk.Label(f, text="Choose the folder where HondaAI-CARLA-Launcher will keep\n"
                         "its configuration, logs, scripts, and virtual environment.",
                 bg=c["bg"], fg=c["fg"], font=("Segoe UI", 10),
                 justify="left").pack(anchor="w", pady=(0, 16))

        row = tk.Frame(f, bg=c["bg"])
        row.pack(fill="x")
        entry = ttk.Entry(row, textvariable=self._proj_root, style="W.TEntry")
        entry.pack(side="left", fill="x", expand=True, padx=(0, 8))
        ttk.Button(row, text="Browse…", style="W.TButton",
                   command=self._browse_proj_root).pack(side="left")

        # Tip
        tk.Label(f,
                 text="Tip: pick a short, permanent path — e.g. ~/HondaAI or ~/projects/carla.",
                 bg=c["bg"], fg=c["muted"], font=("Segoe UI", 9),
                 wraplength=520).pack(anchor="w", pady=(12, 0))

    def _browse_proj_root(self):
        d = _pick_directory("Choose Project Root",
                            initialdir=str(_HERE),
                            parent=self)
        if d:
            self._proj_root.set(d)
            # Trigger CARLA scan in background so it's ready on next step
            threading.Thread(target=self._scan_carla, daemon=True).start()

    def _scan_carla(self):
        root = Path(self._proj_root.get())
        self._carla_choices = _find_carla_dirs(root)
        if self._carla_choices:
            self.after(0, lambda: self._carla_root.set(str(self._carla_choices[0])))

    def _step_carla(self):
        c = self._c
        f = self._content

        tk.Label(f, text="CARLA Installation",
                 bg=c["bg"], fg=c["accent"],
                 font=("Segoe UI", 16, "bold")).pack(anchor="w", pady=(16, 4))

        # Scan now (synchronous, fast)
        root = Path(self._proj_root.get())
        self._carla_choices = _find_carla_dirs(root)

        if self._carla_choices:
            tk.Label(f,
                     text=f"Found {len(self._carla_choices)} CARLA installation(s) "
                          f"inside your project root:",
                     bg=c["bg"], fg=c["fg"],
                     font=("Segoe UI", 10)).pack(anchor="w", pady=(0, 8))

            choice_strs = [str(p) for p in self._carla_choices]
            if not self._carla_root.get() or self._carla_root.get() not in choice_strs:
                self._carla_root.set(choice_strs[0])

            cb = ttk.Combobox(f, textvariable=self._carla_root,
                              values=choice_strs, state="readonly",
                              style="W.TCombobox")
            cb.pack(fill="x", pady=(0, 8))
        else:
            tk.Label(f,
                     text="No CARLA installation was found inside the project root.\n"
                          "Please browse to your CARLA_x.x.x directory manually.",
                     bg=c["bg"], fg=c["yellow"],
                     font=("Segoe UI", 10), wraplength=520,
                     justify="left").pack(anchor="w", pady=(0, 8))

        tk.Label(f, text="CARLA directory:", bg=c["bg"], fg=c["fg"],
                 font=("Segoe UI", 10)).pack(anchor="w")
        row = tk.Frame(f, bg=c["bg"])
        row.pack(fill="x", pady=(4, 0))
        ttk.Entry(row, textvariable=self._carla_root,
                  style="W.TEntry").pack(side="left", fill="x", expand=True, padx=(0, 8))
        ttk.Button(row, text="Browse…", style="W.TButton",
                   command=self._browse_carla).pack(side="left")

        tk.Label(f,
                 text="Expected contents: CarlaUE4.sh (Linux) or CarlaUE4.exe (Windows),\n"
                      "PythonAPI/, Import/, etc.",
                 bg=c["bg"], fg=c["muted"],
                 font=("Segoe UI", 9), justify="left").pack(anchor="w", pady=(12, 0))

    def _browse_carla(self):
        d = _pick_directory("Select CARLA Installation Directory",
                            initialdir=self._proj_root.get(),
                            parent=self)
        if d:
            self._carla_root.set(d)

    def _step_venv(self):
        c = self._c
        f = self._content

        proj   = Path(self._proj_root.get())
        venv   = proj / "venv"
        self._venv_path.set(str(venv))

        tk.Label(f, text="Virtual Environment",
                 bg=c["bg"], fg=c["accent"],
                 font=("Segoe UI", 16, "bold")).pack(anchor="w", pady=(16, 4))

        tk.Label(f,
                 text="A Python virtual environment will be created so all project\n"
                      "scripts run with the correct isolated dependencies.",
                 bg=c["bg"], fg=c["fg"], font=("Segoe UI", 10),
                 justify="left").pack(anchor="w", pady=(0, 14))

        # Path display (editable)
        tk.Label(f, text="Venv location:", bg=c["bg"], fg=c["fg"],
                 font=("Segoe UI", 10, "bold")).pack(anchor="w")
        row = tk.Frame(f, bg=c["bg"])
        row.pack(fill="x", pady=(4, 0))
        ttk.Entry(row, textvariable=self._venv_path,
                  style="W.TEntry").pack(side="left", fill="x", expand=True, padx=(0, 8))
        ttk.Button(row, text="Change…", style="W.TButton",
                   command=self._browse_venv).pack(side="left")

        ttk.Separator(f).pack(fill="x", pady=16)

        tk.Label(f,
                 text="The venv will be created when you click Finish on the next page.",
                 bg=c["bg"], fg=c["muted"],
                 font=("Segoe UI", 9, "italic")).pack(anchor="w")
        tk.Label(f,
                 text="You can install packages from the Virtual Env tab after setup.",
                 bg=c["bg"], fg=c["muted"],
                 font=("Segoe UI", 9)).pack(anchor="w", pady=(6, 0))

    def _browse_venv(self):
        d = _pick_directory("Choose Venv Location",
                            initialdir=self._proj_root.get(),
                            parent=self)
        if d:
            self._venv_path.set(d)

    def _step_done(self):
        c = self._c
        f = self._content

        tk.Label(f, text="All set!",
                 bg=c["bg"], fg=c["green"],
                 font=("Segoe UI", 22, "bold")).pack(anchor="w", pady=(20, 6))
        tk.Label(f, text="HondaAI-CARLA-Launcher is configured and ready.",
                 bg=c["bg"], fg=c["fg"],
                 font=("Segoe UI", 11)).pack(anchor="w", pady=(0, 20))

        for label, val in [
            ("Project root",  self._proj_root.get()),
            ("CARLA install", self._carla_root.get()),
            ("Virtual env",   self._venv_path.get()),
        ]:
            row = tk.Frame(f, bg=c["surface"])
            row.pack(fill="x", pady=2, padx=0)
            tk.Label(row, text=f"  {label}:", bg=c["surface"], fg=c["muted"],
                     font=("Segoe UI", 9), width=16, anchor="w").pack(side="left")
            tk.Label(row, text=val, bg=c["surface"], fg=c["fg"],
                     font=("Consolas", 9)).pack(side="left", padx=(4, 8))

        ttk.Separator(f).pack(fill="x", pady=12)

        self._done_status = tk.Label(f, text="Click  Finish  to create the venv and launch.",
                                     bg=c["bg"], fg=c["muted"],
                                     font=("Segoe UI", 10, "italic"))
        self._done_status.pack(anchor="w")

        self._done_progress = ttk.Progressbar(f, style="W.Horizontal.TProgressbar",
                                              mode="indeterminate")
        self._done_progress.pack(fill="x", pady=(8, 0))

    # -----------------------------------------------------------------------
    # Venv creation (triggered by Finish on the Done step)
    # -----------------------------------------------------------------------

    @staticmethod
    def _find_python() -> str | None:
        """Return a real Python executable that can run -m venv.
        Needed because sys.executable points to the PyInstaller bundle when frozen."""
        import shutil
        candidates = []
        if not getattr(sys, "frozen", False):
            candidates.append(sys.executable)
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

    def _finish(self):
        """Create venv then call on_complete."""
        venv_path = Path(self._venv_path.get())

        if _python_bin(venv_path).exists():
            answer = _venv_conflict_dialog(self, venv_path)
            if answer == "use":
                self._complete()
                return
            elif answer == "cancel":
                return
            # "override": fall through to recreate with --clear

        python = self._find_python()
        if not python:
            messagebox.showerror("Python Not Found",
                                 "Could not find a Python interpreter to create the venv.\n"
                                 "Install Python 3.10+ and try again.",
                                 parent=self)
            return

        self._btn_finish.config(state="disabled")
        self._btn_back.config(state="disabled")
        self._done_progress.start(12)
        self._done_status.config(text="Creating virtual environment...",
                                 fg=self._c["accent"])

        def _run():
            try:
                subprocess.run(
                    [python, "-m", "venv", str(venv_path), "--clear"],
                    check=True, timeout=180, capture_output=True, text=True,
                )
                self.after(0, self._complete)
            except subprocess.CalledProcessError as e:
                self.after(0, lambda: self._venv_failed(e.stderr))
            except Exception as e:
                self.after(0, lambda: self._venv_failed(str(e)))

        threading.Thread(target=_run, daemon=True).start()

    def _complete(self):
        self._done_progress.stop()
        self._done_status.config(text="Done!", fg=self._c["green"])
        self.on_complete({
            "project_root":   self._proj_root.get(),
            "carla_root":     self._carla_root.get(),
            "venv_path":      self._venv_path.get(),
            "setup_complete": True,
        })
        # on_complete calls root.quit() which exits mainloop;
        # root.destroy() in main() cleans up this window.

    def _venv_failed(self, err: str):
        self._done_progress.stop()
        self._done_status.config(text=f"Failed: {err[:120]}", fg=self._c["red"])
        self._btn_finish.config(state="normal")
        self._btn_back.config(state="normal")
        messagebox.showerror("Venv Error",
                             f"Could not create virtual environment:\n\n{err}",
                             parent=self)


# ---------------------------------------------------------------------------
# Helper utilities
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
        return str(venv_path / "Scripts" / "activate.bat")
    return f"source {venv_path / 'bin' / 'activate'}"


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


def _find_carla_processes() -> list[tuple[int, str]]:
    """Return (pid, cmdline) for running CARLA processes found via OS."""
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
    """Return sorted .py files in directory (non-recursive)."""
    try:
        return sorted(p for p in directory.iterdir()
                      if p.suffix == ".py" and p.is_file())
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Process Registry
# ---------------------------------------------------------------------------

class ProcessEntry:
    def __init__(self, name: str, proc_type: str,
                 proc: subprocess.Popen, cmd: str):
        self.name      = name
        self.proc_type = proc_type   # "CARLA" | "Script" | "Shell"
        self.proc      = proc
        self.cmd       = cmd
        self.pid       = proc.pid
        self.started   = time.strftime("%H:%M:%S")

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

        self.carla_root      = tk.StringVar(value=cfg.get("carla_root",      str(CARLA_ROOT_DEFAULT)))
        self.venv_path       = tk.StringVar(value=cfg.get("venv_path",       str(VENV_DEFAULT)))
        self.carla_port      = tk.IntVar(   value=cfg.get("carla_port",      2000))
        self.carla_quality   = tk.StringVar(value=cfg.get("carla_quality",   "Low"))
        self.carla_offscreen = tk.BooleanVar(value=cfg.get("carla_offscreen", False))

        self._custom_scripts: list[Path] = [Path(p) for p in cfg.get("custom_scripts", [])]
        self._custom_dirs:    list[Path] = [Path(p) for p in cfg.get("custom_dirs", [])]

        self.registry = ProcessRegistry()

        self._term_history:  list[str] = []
        self._term_hist_idx: int       = -1

        self._build_styles()
        self._build_ui()
        self._refresh_venv_status()
        self._refresh_process_table()
        self._auto_refresh()

        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # -----------------------------------------------------------------------
    # Persistence
    # -----------------------------------------------------------------------

    def _on_close(self):
        _save_config({
            "carla_root":      self.carla_root.get(),
            "venv_path":       self.venv_path.get(),
            "carla_port":      self.carla_port.get(),
            "carla_quality":   self.carla_quality.get(),
            "carla_offscreen": self.carla_offscreen.get(),
            "custom_scripts":  [str(p) for p in self._custom_scripts],
            "custom_dirs":     [str(p) for p in self._custom_dirs],
        })
        self.destroy()

    # -----------------------------------------------------------------------
    # Styles
    # -----------------------------------------------------------------------

    def _build_styles(self):
        s = ttk.Style(self)
        s.theme_use("clam")
        bg = "#1e1e2e"; fg = "#cdd6f4"; accent = "#89b4fa"
        surface = "#313244"; red = "#f38ba8"; green = "#a6e3a1"
        yellow = "#f9e2af"; cyan = "#89dceb"

        s.configure(".",               background=bg, foreground=fg, fieldbackground=surface)
        s.configure("TFrame",          background=bg)
        s.configure("TLabel",          background=bg, foreground=fg, font=("Segoe UI", 10))
        s.configure("Status.TLabel",   font=("Segoe UI", 9))
        s.configure("TButton",         font=("Segoe UI", 10), padding=6)
        s.configure("Green.TButton",   foreground=green)
        s.configure("Red.TButton",     foreground=red)
        s.configure("TLabelframe",     background=bg, foreground=accent)
        s.configure("TLabelframe.Label", background=bg, foreground=accent,
                    font=("Segoe UI", 11, "bold"))
        s.configure("TCheckbutton",    background=bg, foreground=fg)
        s.configure("TCombobox",       fieldbackground=surface, foreground=fg)
        s.configure("TNotebook",       background=bg)
        s.configure("TNotebook.Tab",   background=surface, foreground=fg,
                    padding=[12, 4], font=("Segoe UI", 10))
        s.map("TNotebook.Tab",
              background=[("selected", accent)],
              foreground=[("selected", "#1e1e2e")])
        s.configure("Treeview",        background=surface, foreground=fg,
                    fieldbackground=surface, rowheight=22)
        s.configure("Treeview.Heading", background=bg, foreground=accent,
                    font=("Segoe UI", 9, "bold"))
        s.map("Treeview",
              background=[("selected", accent)],
              foreground=[("selected", "#1e1e2e")])

        self._c = dict(bg=bg, fg=fg, accent=accent, surface=surface,
                       red=red, green=green, yellow=yellow, cyan=cyan)

    # -----------------------------------------------------------------------
    # UI layout
    # -----------------------------------------------------------------------

    def _build_ui(self):
        hdr = ttk.Frame(self)
        hdr.pack(fill="x", padx=16, pady=(12, 4))
        ttk.Label(hdr, text="HondaAI-CARLA-Launcher",
                  font=("Segoe UI", 18, "bold"),
                  foreground=self._c["accent"]).pack(side="left")

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True, padx=12, pady=8)

        self._build_carla_tab()
        self._build_venv_tab()
        self._build_scripts_tab()
        self._build_terminal_tab()

    # -----------------------------------------------------------------------
    # Tab 1 — CARLA Server
    # -----------------------------------------------------------------------

    def _build_carla_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="  CARLA Server  ")

        # Path
        pf = ttk.LabelFrame(tab, text="CARLA Installation")
        pf.pack(fill="x", padx=12, pady=(12, 6))
        row = ttk.Frame(pf)
        row.pack(fill="x", padx=8, pady=6)
        ttk.Label(row, text="Path:").pack(side="left")
        ttk.Entry(row, textvariable=self.carla_root).pack(
            side="left", padx=6, fill="x", expand=True)
        ttk.Button(row, text="Browse", command=self._browse_carla).pack(side="left")

        # Options
        of = ttk.LabelFrame(tab, text="Launch Options")
        of.pack(fill="x", padx=12, pady=6)
        oi = ttk.Frame(of)
        oi.pack(fill="x", padx=8, pady=6)
        ttk.Label(oi, text="Port:").grid(row=0, column=0, sticky="w", padx=(0, 4))
        ttk.Entry(oi, textvariable=self.carla_port, width=8).grid(row=0, column=1, sticky="w")
        ttk.Label(oi, text="Quality:").grid(row=0, column=2, sticky="w", padx=(20, 4))
        ttk.Combobox(oi, textvariable=self.carla_quality, values=QUALITY_OPTIONS,
                     width=10, state="readonly").grid(row=0, column=3, sticky="w")
        ttk.Checkbutton(oi, text="Offscreen (no window)",
                        variable=self.carla_offscreen).grid(row=0, column=4, padx=(20, 0))

        # Buttons
        bf = ttk.Frame(tab)
        bf.pack(fill="x", padx=12, pady=6)
        ttk.Button(bf, text="Start CARLA", style="Green.TButton",
                   command=self._start_carla).pack(side="left", padx=(0, 6))
        ttk.Button(bf, text="Stop All CARLA", style="Red.TButton",
                   command=self._stop_all_carla).pack(side="left", padx=(0, 6))
        ttk.Button(bf, text="Refresh",
                   command=self._refresh_process_table).pack(side="left")

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

    # -----------------------------------------------------------------------
    # Tab 2 — Virtual Env
    # -----------------------------------------------------------------------

    def _build_venv_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="  Virtual Env  ")

        pf = ttk.LabelFrame(tab, text="Virtual Environment")
        pf.pack(fill="x", padx=12, pady=(12, 6))
        row = ttk.Frame(pf)
        row.pack(fill="x", padx=8, pady=6)
        ttk.Label(row, text="Path:").pack(side="left")
        ttk.Entry(row, textvariable=self.venv_path).pack(
            side="left", padx=6, fill="x", expand=True)
        ttk.Button(row, text="Browse", command=self._browse_venv).pack(side="left")
        self.venv_status_label = ttk.Label(pf, text="", style="Status.TLabel")
        self.venv_status_label.pack(fill="x", padx=8, pady=(0, 6))

        af = ttk.LabelFrame(tab, text="Activation Command")
        af.pack(fill="x", padx=12, pady=6)
        self.activate_var = tk.StringVar()
        ttk.Entry(af, textvariable=self.activate_var, state="readonly").pack(
            fill="x", padx=8, pady=6)
        ttk.Button(af, text="Copy to Clipboard",
                   command=lambda: self._copy_clip(self.activate_var.get())).pack(pady=(0, 6))

        xf = ttk.LabelFrame(tab, text="Actions")
        xf.pack(fill="x", padx=12, pady=6)
        br = ttk.Frame(xf)
        br.pack(fill="x", padx=8, pady=6)
        ttk.Button(br, text="Create Venv", style="Green.TButton",
                   command=self._create_venv).pack(side="left", padx=(0, 6))
        ttk.Button(br, text="Delete Venv", style="Red.TButton",
                   command=self._delete_venv).pack(side="left", padx=(0, 6))
        ttk.Button(br, text="Refresh Status",
                   command=self._refresh_venv_status).pack(side="left")

        rf = ttk.LabelFrame(tab, text="Package Management")
        rf.pack(fill="both", expand=True, padx=12, pady=(6, 12))

        ttk.Label(rf, text="Requirements file:").pack(anchor="w", padx=8, pady=(6, 0))
        self.req_file_var = tk.StringVar(
            value=REQUIREMENTS_FILES_DEFAULT[0] if REQUIREMENTS_FILES_DEFAULT else "")
        ttk.Combobox(rf, textvariable=self.req_file_var,
                     values=REQUIREMENTS_FILES_DEFAULT).pack(
            fill="x", padx=8, pady=4)
        ttk.Button(rf, text="Browse for requirements.txt",
                   command=self._browse_req_file).pack(anchor="w", padx=8, pady=(0, 4))

        pb = ttk.Frame(rf)
        pb.pack(fill="x", padx=8, pady=4)
        ttk.Button(pb, text="Install Requirements",
                   command=self._install_requirements).pack(side="left", padx=(0, 6))
        ttk.Button(pb, text="pip freeze",
                   command=self._pip_freeze).pack(side="left", padx=(0, 6))
        ttk.Button(pb, text="pip list",
                   command=self._pip_list).pack(side="left")

        sr = ttk.Frame(rf)
        sr.pack(fill="x", padx=8, pady=(4, 8))
        ttk.Label(sr, text="Package:").pack(side="left")
        self.single_pkg_var = tk.StringVar()
        ttk.Entry(sr, textvariable=self.single_pkg_var, width=35).pack(side="left", padx=6)
        ttk.Button(sr, text="Install",
                   command=self._install_single_pkg).pack(side="left", padx=(0, 4))
        ttk.Button(sr, text="Uninstall", style="Red.TButton",
                   command=self._uninstall_single_pkg).pack(side="left")

    # -----------------------------------------------------------------------
    # Tab 3 — Scripts
    # -----------------------------------------------------------------------

    def _build_scripts_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="  Scripts  ")

        # Source controls
        df = ttk.LabelFrame(tab, text="Script Sources")
        df.pack(fill="x", padx=12, pady=(12, 6))

        dr = ttk.Frame(df)
        dr.pack(fill="x", padx=8, pady=6)
        ttk.Label(dr, text="PythonAPI/examples:").pack(side="left")
        self.scripts_dir_var = tk.StringVar(value=str(SCRIPTS_DEFAULT))
        ttk.Entry(dr, textvariable=self.scripts_dir_var).pack(
            side="left", padx=6, fill="x", expand=True)
        ttk.Button(dr, text="Browse", command=self._browse_scripts_dir).pack(side="left", padx=(0, 4))
        ttk.Button(dr, text="Reload",  command=self._reload_scripts).pack(side="left")

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
        self.scripts_tree.heading("Name",   text="Name")
        self.scripts_tree.heading("Path",   text="Path")
        self.scripts_tree.column("Source", width=90,  anchor="center")
        self.scripts_tree.column("Name",   width=200)
        self.scripts_tree.column("Path",   width=500)
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

    # -----------------------------------------------------------------------
    # Tab 4 — Terminal
    # -----------------------------------------------------------------------

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
        ttk.Button(tb, text="Clear",    command=self._clear_terminal).pack(side="right", padx=(0, 4))

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
        self.term_text.tag_configure("ts",      foreground="#6c7086")
        self.term_text.tag_configure("info",    foreground=self._c["fg"])
        self.term_text.tag_configure("carla",   foreground=self._c["cyan"])
        self.term_text.tag_configure("script",  foreground=self._c["green"])
        self.term_text.tag_configure("shell",   foreground=self._c["yellow"])
        self.term_text.tag_configure("error",   foreground=self._c["red"])
        self.term_text.tag_configure("success", foreground=self._c["green"])
        self.term_text.tag_configure("cmd_in",  foreground=self._c["accent"],
                                     font=("Consolas", 10, "bold"))

        # Input row
        inp = ttk.Frame(tab)
        inp.pack(fill="x", padx=8, pady=(4, 8))

        vp_name = Path(self.venv_path.get()).name
        self._prompt_label = ttk.Label(
            inp, text=f"({vp_name}) $ ",
            foreground=self._c["green"],
            font=("Consolas", 11, "bold"),
            background=self._c["bg"])
        self._prompt_label.pack(side="left")

        self.term_input = ttk.Entry(inp, font=("Consolas", 10))
        self.term_input.pack(side="left", fill="x", expand=True, padx=(0, 6))
        self.term_input.bind("<Return>", self._run_terminal_cmd)
        self.term_input.bind("<Up>",     self._hist_up)
        self.term_input.bind("<Down>",   self._hist_down)
        self.term_input.bind("<Tab>",    self._tab_complete)

        ttk.Button(inp, text="Run", style="Green.TButton",
                   command=self._run_terminal_cmd).pack(side="left")

        self.term_input.focus()

    # -----------------------------------------------------------------------
    # Terminal helpers
    # -----------------------------------------------------------------------

    def _log(self, message: str, tag: str = "info"):
        """Write a timestamped line to the terminal output area."""
        def _write():
            self.term_text.config(state="normal")
            ts = time.strftime("%H:%M:%S")
            self.term_text.insert("end", f"[{ts}] ", "ts")
            self.term_text.insert("end", message + "\n", tag)
            self.term_text.see("end")
            self.term_text.config(state="disabled")
        # Thread-safe: call on main thread
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
        """Env dict with venv's bin prepended to PATH."""
        vp  = Path(self.venv_path.get())
        env = os.environ.copy()
        bin_dir = str(vp / ("Scripts" if IS_WINDOWS else "bin"))
        env["PATH"]       = bin_dir + os.pathsep + env.get("PATH", "")
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
        text  = self.term_input.get()
        words = text.split()
        if not words:
            return "break"
        last = words[-1]
        try:
            base    = Path(last).parent
            prefix  = Path(last).name
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

    # -----------------------------------------------------------------------
    # CARLA actions
    # -----------------------------------------------------------------------

    def _browse_carla(self):
        d = filedialog.askdirectory(title="Select CARLA Folder",
                                    initialdir=self.carla_root.get())
        if d:
            self.carla_root.set(d)

    def _start_carla(self):
        carla_root = Path(self.carla_root.get())
        exe        = _carla_executable(carla_root)

        if not exe.exists():
            messagebox.showerror("Not Found",
                                 f"CARLA executable not found:\n{exe}\n\n"
                                 "Check the CARLA installation path.")
            return

        if IS_LINUX and not os.access(exe, os.X_OK):
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

        port    = self.carla_port.get()
        quality = self.carla_quality.get()
        cmd = [str(exe), "-carla-port", str(port),
               "-quality-level", quality, "-nosound"]
        if self.carla_offscreen.get():
            cmd.append("-RenderOffScreen")

        self._log(f"Starting CARLA: {' '.join(cmd)}", "carla")
        self.notebook.select(self._term_tab_idx)

        def _run():
            try:
                env    = os.environ.copy()
                kwargs = dict(env=env,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                if IS_WINDOWS:
                    kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
                else:
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
        all_pids  = {p[0] for p in sys_procs} | {e.pid for e in reg_procs}

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

        # System CARLA processes not in registry
        for pid, cmd in _find_carla_processes():
            if pid not in reg_pids:
                self.proc_tree.insert("", "end", iid=f"sys_{pid}",
                    values=("CARLA", "CarlaUE4", pid, "Running", "—", cmd[:80]))

        # All registry entries
        for e in self.registry.all_entries():
            self.proc_tree.insert("", "end", iid=str(e.pid),
                values=(e.proc_type, e.name, e.pid,
                        e.status, e.started, e.cmd[:80]))

        if not self.proc_tree.get_children():
            self.proc_tree.insert("", "end",
                values=("—", "—", "—", "No processes", "—", ""))

    def _auto_refresh(self):
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

    # -----------------------------------------------------------------------
    # Venv actions
    # -----------------------------------------------------------------------

    def _browse_venv(self):
        d = filedialog.askdirectory(title="Select venv folder",
                                    initialdir=str(_HERE))
        if d:
            self.venv_path.set(d)
            self._refresh_venv_status()

    def _browse_req_file(self):
        f = filedialog.askopenfilename(
            title="Select requirements.txt",
            filetypes=[("Text files", "*.txt"), ("All", "*.*")])
        if f:
            self.req_file_var.set(f)

    def _refresh_venv_status(self):
        vp     = Path(self.venv_path.get())
        python = _python_bin(vp)
        self.activate_var.set(_activate_cmd(vp))

        # Update terminal prompt label if it exists
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

    def _create_venv(self):
        vp = Path(self.venv_path.get())
        if _python_bin(vp).exists():
            if not messagebox.askyesno("Exists", f"Venv at {vp} exists. Recreate?"):
                return
        self._log(f"Creating venv at {vp} ...")
        self.notebook.select(self._term_tab_idx)

        def _run():
            try:
                subprocess.run(
                    [sys.executable, "-m", "venv", str(vp), "--clear"],
                    check=True, timeout=120, capture_output=True, text=True)
                self._log(f"Venv created: {vp}", "success")
                self.after(0, self._refresh_venv_status)
            except subprocess.CalledProcessError as e:
                self._log(f"Failed: {e.stderr}", "error")

        threading.Thread(target=_run, daemon=True).start()

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

    # -----------------------------------------------------------------------
    # Scripts tab actions
    # -----------------------------------------------------------------------

    def _browse_scripts_dir(self):
        d = filedialog.askdirectory(title="Select scripts directory",
                                    initialdir=self.scripts_dir_var.get())
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
                _save_config({
                    "custom_scripts": [str(x) for x in self._custom_scripts],
                    "custom_dirs":    [str(x) for x in self._custom_dirs],
                })
            self._reload_scripts()

    def _add_custom_dir(self):
        d = filedialog.askdirectory(title="Add Script Directory")
        if d:
            p = Path(d)
            if p not in self._custom_dirs:
                self._custom_dirs.append(p)
                _save_config({
                    "custom_scripts": [str(x) for x in self._custom_scripts],
                    "custom_dirs":    [str(x) for x in self._custom_dirs],
                })
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
        _save_config({
            "custom_scripts": [str(x) for x in self._custom_scripts],
            "custom_dirs":    [str(x) for x in self._custom_dirs],
        })
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
                env    = self._venv_env()
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
# Entry point
# ---------------------------------------------------------------------------

def main():
    # Hidden root owns the wizard as a Toplevel (required by Tk)
    root = tk.Tk()
    root.withdraw()

    if _needs_setup():
        result: dict = {}

        def _on_setup_complete(cfg: dict):
            existing = _load_config()
            existing.update(cfg)
            _save_config(existing)
            result.update(existing)
            root.quit()   # exit mainloop cleanly; root.destroy() follows

        SetupWizard(root, _on_setup_complete)
        root.mainloop()   # blocks until root.quit()
        root.destroy()    # now safe to destroy

        if not result.get("setup_complete"):
            return        # wizard was cancelled
    else:
        root.destroy()

    _launch_main()


def _launch_main():
    app = CarlaLauncher()
    app.mainloop()


if __name__ == "__main__":
    main()
