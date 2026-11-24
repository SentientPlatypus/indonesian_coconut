#!/usr/bin/env python3
import sys
import pathlib
import re

def set_top(text, key, val):
    pat = re.compile(rf"(?m)^{re.escape(key)}\s*=.*$")
    line = f"{key} = {val}"
    if pat.search(text):
        return pat.sub(line, text, count=1)
    m = re.search(r"(?m)^\[", text)
    if m:
        return text[:m.start()] + line + "\n" + text[m.start():]
    return text.rstrip() + "\n" + line + "\n"

def main():
    src, dest, lever, out, ck, nexto, iters, mb, tpi, replicas, chunk, intra = sys.argv[1:]
    text = pathlib.Path(src).read_text()
    text = set_top(text, "checkpoints", f'"{out}/{lever}_ckpt"')
    text = set_top(text, "iterations", iters)
    text = set_top(text, "mini_batch_size", mb)
    text = set_top(text, "timesteps_per_iteration", tpi)
    text = set_top(text, "async_collect", "false")
    text = set_top(text, "critic_warmup_steps", "0")
    text = re.sub(r"(?m)^chance = .*$", "chance = 1.0", text, count=1)
    text = re.sub(r"(?m)^past_versions_weight = .*$", "past_versions_weight = 0.0", text, count=1)
    text = re.sub(
        r"\n\[\[opponents\.(?:external|fixed)\]\][\s\S]*?(?=\n\[|\Z)",
        "\n",
        text,
    )
    # Drop PFSP: a single nexto external + chance=1 forces Nexto-only.
    # Leaving clamp/floor around a 1.0 league share is infeasible.
    text = re.sub(r"(?ms)^\[pfsp(?:\.clamp)?\]\n.*?(?=\n\[|\Z)", "\n", text)
    text = set_top(text, "async_collect", "false")
    if lever == "new":
        text = set_top(text, "nf_intra_threads", intra)
    block = (
        "\n\n[[opponents.external]]\n"
        'name = "nexto"\n'
        'kind = "nexto"\n'
        f'path = "{nexto}"\n'
        "weight = 1.0\n"
        "stochastic = 0.2\n"
    )
    if lever == "new":
        block += f"replicas = {replicas}\nchunk_rows = {chunk}\n"
    pathlib.Path(dest).write_text(text.rstrip() + "\n" + block)
    print("wrote", dest, "chars", len(text) + len(block))

if __name__ == "__main__":
    main()
