#!/usr/bin/env python3
"""Detect a live team trainer or mid-eval.

Matches any /home/ubuntu/ippo_runs/exp_TM*/indococo-train train process
and official uniquely-named bins (indococo-train-tm12-*, tm11-*, …).
Does not match speed-bench bins (indococo-train-speed-*), TM12 wait/launch
wrappers, or ssh/inspect lines that only mention those paths in arguments.

TEAM_PROC_ROOT defaults to /proc (override with a stub tree in tests).
TEAM_GPU_USED_MIB, if set, overrides nvidia-smi for state().
"""
from __future__ import annotations

import os
import pathlib
import re
import subprocess
import sys

PROC_ROOT = os.environ.get("TEAM_PROC_ROOT", "/proc")

# /home/ubuntu/ippo_runs/exp_TM11/indococo-train
# /home/ubuntu/ippo_runs/exp_TM12/indococo-train
# /home/ubuntu/ippo_runs/exp_TM5r/indococo-train (historical suffix)
EXP_TRAIN_RE = re.compile(
    r"^/home/ubuntu/ippo_runs/exp_TM\d+[A-Za-z]*/indococo-train$"
)

# Official uniquely-named trainer bins, not indococo-train-speed-*.
OFFICIAL_BIN_RE = re.compile(r"(?:^|/)indococo-train-tm\d+[A-Za-z0-9_-]*$")

EXP_TM_IN_CMD = re.compile(r"/exp_TM\d+[A-Za-z]*/")

# cmdline is joined with spaces and wrapped in spaces before matching.
H2H_MARKERS = (
    " galactus-h2h ",
    " megabyte-h2h ",
    " nexto-h2h ",
    " necto-h2h ",
    " h2h ",
)


def argv_of(proc_dir: pathlib.Path) -> list[str] | None:
    try:
        raw = (proc_dir / "cmdline").read_bytes()
    except OSError:
        return None
    if not raw:
        return None
    parts = [p.decode("utf-8", "replace") for p in raw.split(b"\0") if p]
    return parts or None


def is_team_trainer(argv: list[str]) -> bool:
    """True iff argv is a live team *train* process, not an eval or wrapper."""
    if len(argv) < 2:
        return False
    if argv[1] != "train":
        return False
    argv0 = argv[0]
    if EXP_TRAIN_RE.match(argv0):
        return True
    if OFFICIAL_BIN_RE.search(argv0):
        return True
    return False


def is_mid_eval(argv: list[str]) -> bool:
    if is_team_trainer(argv):
        return False
    cmd = " " + " ".join(argv) + " "
    if not EXP_TM_IN_CMD.search(cmd):
        return False
    return any(m in cmd for m in H2H_MARKERS)


def classify(argv: list[str]) -> str:
    if is_team_trainer(argv):
        return "TRAINER"
    if is_mid_eval(argv):
        return "MID"
    return "OTHER"


def _skip_pids() -> set[int]:
    pids = {os.getpid(), os.getppid()}
    # Also skip an explicit list (tests, or a guard that is scanning itself).
    extra = os.environ.get("TEAM_DETECT_SKIP_PIDS", "")
    for tok in extra.replace(",", " ").split():
        if tok.isdigit():
            pids.add(int(tok))
    return pids


def iter_procs() -> list[tuple[int, list[str]]]:
    root = pathlib.Path(PROC_ROOT)
    skip = _skip_pids()
    out: list[tuple[int, list[str]]] = []
    try:
        entries = list(root.iterdir())
    except OSError:
        return out
    for p in entries:
        if not p.name.isdigit():
            continue
        pid = int(p.name)
        if pid in skip:
            continue
        argv = argv_of(p)
        if argv:
            out.append((pid, argv))
    out.sort(key=lambda x: x[0])
    return out


def trainer_pids() -> list[int]:
    return [pid for pid, argv in iter_procs() if is_team_trainer(argv)]


def mid_eval_pids() -> list[int]:
    return [pid for pid, argv in iter_procs() if is_mid_eval(argv)]


def gpu_used_mib() -> int:
    if "TEAM_GPU_USED_MIB" in os.environ and os.environ["TEAM_GPU_USED_MIB"] != "":
        return int(os.environ["TEAM_GPU_USED_MIB"].strip())
    try:
        raw = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return int(raw.strip().splitlines()[0].split()[0])
    except (OSError, subprocess.CalledProcessError, ValueError, IndexError):
        return 99999


def state_line() -> str:
    trainers = trainer_pids()
    used = gpu_used_mib()
    if trainers:
        pids = ",".join(str(p) for p in trainers)
        return f"TRAIN pid={pids} used={used}MiB"
    mids = mid_eval_pids()
    if mids:
        ev = ",".join(str(p) for p in mids)
        return f"MID used={used}MiB ev={ev}"
    return f"GAP used={used}MiB"


def main(argv: list[str]) -> int:
    if not argv or argv[0] in ("-h", "--help"):
        sys.stderr.write(
            "usage: team_detect.py pid|pids|mid-pids|state|classify [argv...]\n"
        )
        return 2
    cmd = argv[0]
    if cmd == "pid":
        pids = trainer_pids()
        if pids:
            print(pids[0])
        return 0
    if cmd == "pids":
        for p in trainer_pids():
            print(p)
        return 0
    if cmd == "mid-pids":
        for p in mid_eval_pids():
            print(p)
        return 0
    if cmd == "state":
        print(state_line())
        return 0
    if cmd == "classify":
        if len(argv) < 2:
            sys.stderr.write("usage: team_detect.py classify <argv0> [args...]\n")
            return 2
        print(classify(argv[1:]))
        return 0
    sys.stderr.write(f"unknown command: {cmd}\n")
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
