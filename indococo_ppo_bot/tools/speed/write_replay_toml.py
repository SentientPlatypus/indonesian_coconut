#!/usr/bin/env python3
"""Write a learn-replay train.toml for one learner-bench arm. Defaults stay off."""
import argparse
import pathlib
import re
import sys


def set_top(text, key, val):
    pat = re.compile(rf"(?m)^{re.escape(key)}\s*=.*$")
    line = f"{key} = {val}"
    if pat.search(text):
        return pat.sub(line, text, count=1)
    m = re.search(r"(?m)^\[", text)
    if m:
        return text[: m.start()] + line + "\n" + text[m.start() :]
    return text.rstrip() + "\n" + line + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="template train.toml")
    ap.add_argument("--arm", required=True, help="base | l1 | l1l2 | l3 | l1l2l3")
    ap.add_argument("--mb", type=int, default=50000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    text = pathlib.Path(args.src).read_text()
    text = set_top(text, "mini_batch_size", str(args.mb))
    # Default path (base): i64 masks, pinned off, no critic_layers key.
    text = set_top(text, "mask_upload", '"i64"')
    text = set_top(text, "pinned_upload", "false")
    arm = args.arm
    if "l1" in arm:
        text = set_top(text, "mask_upload", '"auto"')
    if "l2" in arm:
        text = set_top(text, "pinned_upload", "true")
    if "l3" in arm:
        # critic_layers is added later with the L3 lever; keep the key out until then.
        pass
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.out).write_text(text)
    print("wrote", args.out, "arm", arm)


if __name__ == "__main__":
    sys.exit(main())
