# HondaAI-CARLA-Launcher — Features & Design

## Target Platforms

- **Linux** and **Windows** only — no macOS support
- Python 3.10+ is a documented prerequisite

### Distribution

| Platform | How to run |
|----------|-----------|
| Windows | Double-click `launcher.pyw` (no console window if Python file associations are set) |
| Linux | Double-click `HondaAI-CARLA-Launcher.desktop`, or `python3 launcher.py` from a terminal |

Three files shipped:
- `launcher.py` — main application source
- `launcher.pyw` — identical content; `.pyw` extension suppresses the console window on Windows
- `HondaAI-CARLA-Launcher.desktop` — freedesktop entry with `Terminal=false`; lets Linux users launch without a terminal window

---

## Design Philosophy

- The app **always launches** — no blocking wizard, no modal setup flow
- A **Setup tab** is the single source of truth for paths; shown first when paths are not yet valid
- CARLA path and venv path are configured **only in the Setup tab** — other tabs use them but don't duplicate the controls
- Status indicators tell the user exactly what is missing or broken
- Every other tab works independently — buttons show errors inline when prerequisites (CARLA path, venv) are missing
- **Pure stdlib** — `tkinter`, `json`, `subprocess`, `threading`, `pathlib` only; zero third-party deps
- **Single source file** — all logic in one `launcher.py`
- **Config saves on close** — simple flat JSON next to the launcher

---

## Config

**Location:** hidden file next to `launcher.py`

```python
_HERE = Path(__file__).resolve().parent
_CONFIG_FILE = _HERE / ".hondaai_launcher_config.json"
```

**Schema (flat JSON):**

```json
{
  "carla_root":      "",
  "venv_path":       "",
  "carla_port":      2000,
  "carla_quality":   "Low",
  "carla_offscreen": false,
  "carla_map":       "Town10HD",
  "auto_refresh":    true,
  "custom_scripts":  [],
  "custom_dirs":     []
}
```

- Corrupt or missing config falls back to defaults — app never crashes on bad config
- No breadcrumb files, no nested config directories

---

## Tab 1 — Setup

**Purpose:** One-stop configuration for all paths. Replaces the old multi-step wizard so users are never blocked from the app. This is the *only* place CARLA root and venv path are edited.

### Features

| Feature | How it works | Why |
|---------|-------------|-----|
| **CARLA Root path** | Entry + Browse + Auto-detect + status label (✓/✗) | User needs to point at their CARLA install exactly once |
| **Venv path** | Entry + Browse + Auto-detect + Create Venv + status label (✓/✗) | Separate from CARLA root so users can place venvs anywhere |
| **Auto-detect CARLA** | Button scans `_HERE` and one level deep for `CarlaUE4.sh`/`.exe` | Saves browsing on typical project layouts where CARLA sits in a subdirectory |
| **Auto-detect venv** | BFS scan from `_HERE` up to 3 levels deep for dirs containing `bin/python` or `Scripts/python.exe` | Finds existing venvs without manual browsing; breadth-first so nearest match wins |
| **Create Venv** | Button, disabled until CARLA root is valid | Convenient shortcut so users don't have to switch to the Venv tab |
| **CARLA Requirements check** | Background check runs when both paths are valid; compares installed packages against `carla` + `PythonAPI/examples/requirements.txt` | Users immediately see if their venv is missing packages |
| **Install Missing** | Button enabled when packages are missing; runs `pip install carla -r requirements.txt` | One click to get a venv CARLA-ready without switching tabs |
| **Live validation** | `StringVar` traces fire on every keystroke; status labels update immediately | Instant feedback — no "Save" or "Apply" button needed |
| **Status summary** | Bottom of tab: "Ready" (green) or "X items need attention" (yellow) with bullet list | At-a-glance health check |
| **Tab selection on launch** | If config is valid → open to CARLA Server; otherwise → open to Setup | First-time users land on Setup; returning users skip it |

### Validation rules

- **CARLA root:** directory exists AND contains `CarlaUE4.sh` or `CarlaUE4.exe`
- **Venv:** `bin/python` (Linux) or `Scripts/python.exe` (Windows) exists inside the path
- **Packages:** `carla` pip package installed AND all entries in `{carla_root}/PythonAPI/examples/requirements.txt` present

---

## Tab 2 — Packages

**Purpose:** Package management for the venv. Path and creation are handled in Setup — this tab focuses on inspecting and modifying what's installed.

### Features

| Feature | How it works | Why |
|---------|-------------|-----|
| **Venv status** | Read-only label showing Python version and path (or "Not found") | Quick confirmation of which venv is active without switching to Setup |
| **Activation command** | Read-only entry + Copy button | Users need this for their own external terminal sessions |
| **Delete Venv** | `shutil.rmtree` after confirmation | Nuclear option for broken envs; create a new one from the Setup tab |
| **Requirements file** | Combobox pre-populated with common paths under CARLA root + Browse + Install | One click to install from any requirements.txt |
| **Single package install/uninstall** | Text entry + Install/Uninstall buttons | Ad-hoc package management without typing pip commands |
| **pip list / pip freeze** | Buttons that output to the Terminal tab | Quick package inspection |

---

## Tab 3 — CARLA Server

**Purpose:** Launch, stop, and monitor CARLA simulator instances. Change maps on a running server.

No CARLA path controls here — the path comes from the Setup tab. This keeps a single source of truth and avoids the confusion of having the same field in two places.

### Features

| Feature | How it works | Why |
|---------|-------------|-----|
| **Launch options** | Port (int entry), Quality (Low/Med/High/Epic combobox, default Low), Offscreen (checkbox) | Covers the flags users actually change; Low default keeps CARLA responsive on modest hardware |
| **Map selector** | Editable combobox + "Refresh Maps" + "Change Map" buttons + status label | Maps are queried from the running server, not hardcoded — different CARLA versions ship different maps |
| **Start CARLA** | Spawns `CarlaUE4.sh/.exe` with flags in a background thread; streams stdout to Terminal | Non-blocking — UI stays responsive while CARLA boots |
| **Auto chmod +x** | Linux only: sets executable bit if missing, falls back to `sudo chmod +x` with guidance | Common gotcha on fresh CARLA extracts |
| **Stop All CARLA** | Kills registry entries + system CARLA processes found via `ps aux`/`tasklist` | Catches CARLA instances launched outside the GUI too |
| **Process table** | Treeview: Type, Name, PID, Status, Started, Command | Unified view of everything the launcher is managing |
| **Kill Selected / Kill All Scripts / Remove Exited** | Buttons next to the table | Granular process control without leaving the GUI |
| **Refresh** | Manual button to refresh the process table on demand | For when auto-refresh is off or the user wants an immediate update |
| **Auto-refresh toggle** | Checkbox next to Refresh; when on, table refreshes every 3 seconds; persisted in config | Some users find the 3s poll distracting or want to reduce CPU use |

### Map management

- **Refresh Maps** queries `client.get_available_maps()` from the running CARLA server and populates the combobox dynamically
- **Change Map** runs `client.load_world(name)` to switch the active map without restarting CARLA
- Both require the `carla` pip package in the venv
- The combobox is editable so users can type a map name directly (useful for custom maps)
- Status label shows "Querying..." / "N maps available" / "Loading..." / map name / error
- Default value in config: **Town10HD** (CARLA's default map), but the dropdown only fills with real maps once Refresh Maps is clicked

### Process management details

- CARLA launched with `preexec_fn=os.setsid` (Linux) / `CREATE_NEW_PROCESS_GROUP` (Windows) so the whole process group can be killed
- System-level CARLA processes (not launched by the GUI) appear in the table with a dash for Started time
- Registry is in-memory only — no persistence across launcher restarts

---

## Tab 4 — Scripts

**Purpose:** Browse, manage, and launch Python scripts using the configured venv.

### Features

| Feature | How it works | Why |
|---------|-------------|-----|
| **Default source** | `{carla_root}/PythonAPI/examples/` — all `.py` files scanned | CARLA ships example scripts here; makes them immediately discoverable |
| **Custom script files** | "Add Script File" button → file picker → added to list | For one-off scripts outside the examples directory |
| **Custom directories** | "Add Script Directory" button → all `.py` files inside are listed | For project-specific script folders |
| **Treeview** | Columns: Source, Name, Path | Source column tells you where each script came from |
| **Double-click to launch** | Runs script with venv Python in a background thread | Fastest path from browsing to running |
| **Args entry** | Text field next to Launch button | Many CARLA scripts accept `--host`, `--port`, etc. |
| **Remove Selected** | Removes custom entries only; built-in examples can't be removed | Prevents accidentally hiding the examples directory |
| **Persisted in config** | `custom_scripts` and `custom_dirs` arrays saved on close | Custom sources survive launcher restarts |

### Script execution

- Scripts run with the venv's Python, with venv `bin/` prepended to PATH
- Working directory is set to the script's parent directory (so relative imports work)
- `preexec_fn=os.setsid` (Linux) / `CREATE_NEW_PROCESS_GROUP` (Windows) for clean process group kills
- All output streams to the Terminal tab in real time

---

## Tab 5 — Terminal


**Purpose:** Unified output area for all launcher activity, plus an interactive shell.

### Features

| Feature | How it works | Why |
|---------|-------------|-----|
| **Output area** | Read-only `tk.Text` widget with color tags | All process output (CARLA, scripts, pip, shell commands) flows here |
| **Color coding** | Cyan = CARLA, Green = scripts, Yellow = shell, Red = errors, Blue/bold = typed commands, Gray = timestamps | Visually separates interleaved output from multiple sources |
| **Shell input** | Entry at the bottom with `(venv-name) $` prompt | Quick commands without leaving the GUI |
| **Venv-activated commands** | Shell commands run with venv `bin/` on PATH + `VIRTUAL_ENV` set | `python`, `pip`, etc. resolve to the venv without manual activation |
| **Command history** | Up/Down arrow keys cycle through previous commands | Standard shell UX |
| **Tab completion** | Completes file paths in the current directory | Basic convenience for `python script.py` style commands |
| **Save Log** | File dialog → writes terminal content to a `.txt` file | For debugging sessions or sharing output |
| **Clear** | Empties the terminal output | Keeps the view manageable during long sessions |

### Threading model

- Every command/process runs in a daemon thread
- Output lines are posted to the main thread via `self.after(0, ...)` for thread-safe Tk updates
- The `_log()` method auto-detects which thread it's on and routes accordingly

---

## Cross-Platform Details

| Concern | Linux | Windows |
|---------|-------|---------|
| Launch without terminal | `HondaAI-CARLA-Launcher.desktop` (`Terminal=false`) | `launcher.pyw` (`.pyw` suppresses console) |
| Python in venv | `venv/bin/python` | `venv/Scripts/python.exe` |
| Pip in venv | `venv/bin/pip` | `venv/Scripts/pip.exe` |
| Activate command | `source ".../activate"` | `call ".../activate.bat"` |
| CARLA executable | `CarlaUE4.sh` | `CarlaUE4.exe` (also checks `WindowsNoEditor/`) |
| Process isolation | `preexec_fn=os.setsid` | `CREATE_NEW_PROCESS_GROUP` |
| Kill process | `os.kill(pid, SIGTERM)` | `taskkill /F /PID` |
| Find CARLA procs | `ps aux` | `tasklist /FO CSV /NH` |
| Directory picker | zenity → kdialog → tkinter fallback | PowerShell FolderBrowserDialog → tkinter fallback |
| chmod +x | Applied automatically if missing | N/A |

---

## Behavioral Design Choices

- **Config saves on close** — no partial saves mid-session; avoids file I/O on every keystroke
- **All processes killed on exit** — `_on_close` kills all launcher-managed process groups (`os.killpg` on Linux, `taskkill` on Windows) plus any system-level CARLA processes found via `ps`/`tasklist`
- **CARLA 0.9.x layout assumed** — `CarlaUE4.sh`/`.exe` at root, `PythonAPI/examples/` for scripts
- **Default quality: Low** — keeps CARLA responsive on modest hardware; other options available in the combobox
- **Default map: Town10HD** — CARLA's default map
- **No persistent log** — terminal output is in-memory only; "Save Log" is manual
- **No multi-CARLA-instance management** — one port / one instance at a time (process table shows all, but launch options target one)

---

## Theme

Catppuccin Mocha palette via ttk `clam` theme:

| Role | Hex | Usage |
|------|-----|-------|
| Background | `#1e1e2e` | Window and frame backgrounds |
| Foreground | `#cdd6f4` | Default text |
| Accent | `#89b4fa` | Tab highlights, headings, typed commands |
| Surface | `#313244` | Entry fields, treeview background, separators |
| Green | `#a6e3a1` | Success messages, "Start" buttons, valid status |
| Red | `#f38ba8` | Error messages, "Stop/Delete" buttons, invalid status |
| Yellow | `#f9e2af` | Warnings, shell output |
| Cyan | `#89dceb` | CARLA output |
| Muted | `#6c7086` | Timestamps, hints |
