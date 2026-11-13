#!/usr/bin/env python3
"""3D / 2D positioning heatmaps from full-game traces written by tools/viz/pos_dump.py.

  python tools/viz/make_positions.py --label ST25 \
      --in galactus=~/ippo_runs/viz/positions/st25_vs_galactus.jsonl \
      --in fr4=~/ippo_runs/viz/positions/st25_vs_fr4.jsonl --out docs/figures/positions

Everything is rotated so OUR goal is at -y (we attack +y). Demoed frames are dropped.
Writes positions_3d.html (plotly.js from CDN, data inline), PNGs and positions_stats.json.
No plotly Python package needed; numpy 1.26.4 + matplotlib only.
"""
import argparse, json, math, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, LinearSegmentedColormap
from matplotlib.patches import Rectangle

BG, PANEL, FG, GRID, MUTED = "#0f1117", "#161b22", "#e6edf3", "#30363d", "#8b949e"
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": PANEL, "savefig.facecolor": BG, "axes.edgecolor": GRID,
                     "axes.labelcolor": FG, "axes.titlecolor": FG, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": FG,
                     "grid.color": GRID, "legend.facecolor": PANEL, "legend.edgecolor": GRID, "font.size": 9,
                     "axes.titlesize": 10, "axes.titleweight": "bold"})
L, W, H, GW, GH, GD = 5120.0, 4096.0, 2044.0, 893.0, 642.0, 880.0
NX, NY, NZ = 16, 20, 8
def dark_cmap(c): return LinearSegmentedColormap.from_list("d" + c, ["#1c2633", c, "#ffffff"])
CM = {"Blues": dark_cmap("#3b8eea"), "Reds": dark_cmap("#e5534b"), "Purples": dark_cmap("#a371f7")}
EX = (-W, W); EY = (-L, L); EZ = (0, H)
ENT = {"u": "us", "o": "opponent", "b": "ball"}
OPPNAME = {"galactus": "Galactus", "fr4": "FR4"}

def log(*a): print("[pos]", *a, flush=True)

def load(path, skip_ticks=0):
    """-> dict of arrays: u,o,b (N,3), u_ground, o_ground, bv (N,3), game idx, result per frame."""
    U, O, B, BV, UG, OG, GI, RES, UB = [], [], [], [], [], [], [], [], []
    games = 0; ticks = 0; results = []
    for gi, l in enumerate(open(path)):
        g = json.loads(l); s = 1.0 if g["cand_team"] == 0 else -1.0
        games += 1; ticks += g["ticks"]; results.append(g["result"])
        for f in g["frames"]:
            u, o = f.get("u"), f.get("o")
            if u is None or o is None or u[5] or o[5] or f["t"] < skip_ticks: continue
            U.append([s * u[0], s * u[1], u[2]]); O.append([s * o[0], s * o[1], o[2]])
            B.append([s * f["b"][0], s * f["b"][1], f["b"][2]]); BV.append([s * f["bv"][0], s * f["bv"][1], f["bv"][2]])
            UG.append(u[4]); OG.append(o[4]); UB.append(u[3]); GI.append(gi); RES.append(g["result"])
    d = {k: np.array(v, float) for k, v in dict(u=U, o=O, b=B, bv=BV, ug=UG, og=OG, ub=UB, gi=GI, res=RES).items()}
    d["games"], d["ticks"], d["results"] = games, ticks, results
    return d

def stats(d):
    u, o, b, bv = d["u"], d["o"], d["b"], d["bv"]
    n = len(u)
    pct = lambda m: round(100.0 * float(np.mean(m)), 1) if len(m) else None
    du = np.linalg.norm(u[:, :2] - b[:, :2], axis=1); do = np.linalg.norm(o[:, :2] - b[:, :2], axis=1)
    attack = (b[:, 1] < 0) & (bv[:, 1] < -300)            # ball in our half, moving at our goal
    opp_pos = do < du                                    # opponent closer to the ball
    threat = attack & opp_pos
    s = {"frames": int(n), "games": d["games"], "game_ticks": int(d["ticks"]),
         "win_pct": pct(np.array(d["results"]) == 1),
         "us_goal_side_pct": pct(u[:, 1] < b[:, 1]),
         "opp_goal_side_pct": pct(o[:, 1] > b[:, 1]),
         "us_air_pct": pct(d["ug"] == 0), "opp_air_pct": pct(d["og"] == 0),
         "us_high_pct_z_gt_300": pct(u[:, 2] > 300), "opp_high_pct_z_gt_300": pct(o[:, 2] > 300),
         "us_own_half_pct": pct(u[:, 1] < 0), "opp_own_half_pct": pct(o[:, 1] > 0),
         "ball_our_half_pct": pct(b[:, 1] < 0),
         "us_closer_to_ball_pct": pct(du < do),
         "us_mean_boost": round(float(np.mean(d["ub"])), 1),
         "threat_frames_pct": pct(threat),
         "threat_us_goal_side_pct": pct(u[threat, 1] < b[threat, 1]) if threat.any() else None,
         "threat_us_mean_dist_own_goal": round(float(np.mean(np.linalg.norm(u[threat, :2] - [0, -L], axis=1))), 0) if threat.any() else None,
         "threat_us_in_box_pct": pct((np.abs(u[threat, 0]) < 1800) & (u[threat, 1] < -L + 1800)) if threat.any() else None,
         "threat_us_air_pct": pct(d["ug"][threat] == 0) if threat.any() else None,
         "us_mean_dist_ball": round(float(np.mean(du)), 0), "opp_mean_dist_ball": round(float(np.mean(do)), 0)}
    return s, threat

def vox(p):
    h, _ = np.histogramdd(np.clip(p, [EX[0], EY[0], EZ[0]], [EX[1] - 1e-3, EY[1] - 1e-3, EZ[1] - 1e-3]),
                          bins=(NX, NY, NZ), range=(EX, EY, EZ))
    return 100.0 * h / max(1, len(p))     # % of frames per voxel

def centers():
    cx = np.linspace(EX[0], EX[1], NX + 1); cy = np.linspace(EY[0], EY[1], NY + 1); cz = np.linspace(EZ[0], EZ[1], NZ + 1)
    cx, cy, cz = (cx[:-1] + cx[1:]) / 2, (cy[:-1] + cy[1:]) / 2, (cz[:-1] + cz[1:]) / 2
    return np.meshgrid(cx, cy, cz, indexing="ij")

def field_lines():
    """Wireframe segments: floor, ceiling, corners, goals (as lists of polylines)."""
    P = []
    for z in (0, H):
        P.append([(-W, -L, z), (W, -L, z), (W, L, z), (-W, L, z), (-W, -L, z)])
    for x in (-W, W):
        for y in (-L, L): P.append([(x, y, 0), (x, y, H)])
    P.append([(-W, 0, 0), (W, 0, 0)])
    for s in (-1, 1):
        y0, y1 = s * L, s * (L + GD)
        P.append([(-GW, y0, 0), (-GW, y0, GH), (GW, y0, GH), (GW, y0, 0)])
        P.append([(-GW, y1, 0), (-GW, y1, GH), (GW, y1, GH), (GW, y1, 0), (-GW, y1, 0)])
        for x in (-GW, GW):
            P.append([(x, y0, 0), (x, y1, 0)]); P.append([(x, y0, GH), (x, y1, GH)])
    th = np.linspace(0, 2 * np.pi, 48)
    P.append([(1000 * math.cos(t), 1000 * math.sin(t), 0) for t in th])
    return P

# ------------------------------------------------------------------ plotly html
def write_html(path, data, label, st):
    X, Y, Z = centers()
    traces, names = [], []
    cmaps = {"u": "Blues", "o": "Reds", "b": "YlOrBr"}
    for m, d in data.items():
        for e in ("u", "o", "b"):
            v = vox(d[e]); vmax = float(np.percentile(v[v > 0], 99)) if (v > 0).any() else 1.0
            traces.append({"type": "volume", "x": X.ravel().round(0).tolist(), "y": Y.ravel().round(0).tolist(),
                           "z": Z.ravel().round(0).tolist(), "value": v.ravel().round(4).tolist(),
                           "isomin": vmax * 0.06, "isomax": vmax, "opacity": 0.14, "surface": {"count": 14},
                           "colorscale": cmaps[e], "reversescale": e != "u" and False, "showscale": True,
                           "colorbar": {"title": {"text": "% of frames"}, "len": 0.6, "x": 1.0},
                           "name": f"{ENT[e]} vs {OPPNAME.get(m, m)}", "visible": (m == list(data)[0] and e == "u"),
                           "hovertemplate": "x %{x}<br>y %{y}<br>z %{z}<br>%{value:.3f}% of frames<extra></extra>"})
            names.append((m, e))
    nvol = len(traces)
    xs, ys, zs = [], [], []
    for pl in field_lines():
        for p in pl: xs.append(p[0]); ys.append(p[1]); zs.append(p[2])
        xs.append(None); ys.append(None); zs.append(None)
    traces.append({"type": "scatter3d", "mode": "lines", "x": xs, "y": ys, "z": zs, "hoverinfo": "skip",
                   "line": {"color": "#8b949e", "width": 3}, "name": "field", "showlegend": False})
    traces.append({"type": "scatter3d", "mode": "text", "x": [0, 0], "y": [-L - 500, L + 500], "z": [800, 800],
                   "text": ["OUR GOAL", "THEIR GOAL"], "textfont": {"color": ["#f78166", "#58a6ff"], "size": 13},
                   "hoverinfo": "skip", "showlegend": False})
    buttons = []
    for i, (m, e) in enumerate(names):
        vis = [j == i for j in range(nvol)] + [True, True]
        s = st[m]
        buttons.append({"label": f"{ENT[e]} · vs {OPPNAME.get(m, m)}", "method": "update",
                        "args": [{"visible": vis}, {"title.text": f"{label} positioning density: {ENT[e]} vs {OPPNAME.get(m, m)} "
                                 f"({s['games']} games, {s['frames']:,} frames)"}]})
    m0 = list(data)[0]
    layout = {"title": {"text": f"{label} positioning density: us vs {OPPNAME.get(m0, m0)} ({st[m0]['games']} games, {st[m0]['frames']:,} frames)",
                        "font": {"color": "#e6edf3"}},
              "paper_bgcolor": "#0f1117", "font": {"color": "#e6edf3"},
              "scene": {"xaxis": {"title": "x (side)", "range": [-W - 200, W + 200], "backgroundcolor": "#0f1117", "gridcolor": "#30363d"},
                        "yaxis": {"title": "y (our goal → their goal)", "range": [-L - GD - 200, L + GD + 200], "backgroundcolor": "#0f1117", "gridcolor": "#30363d"},
                        "zaxis": {"title": "z (height)", "range": [0, H + 50], "backgroundcolor": "#0f1117", "gridcolor": "#30363d"},
                        "aspectmode": "manual", "aspectratio": {"x": 0.8, "y": 2.0, "z": 0.4},
                        "camera": {"eye": {"x": 1.6, "y": -1.6, "z": 0.9}}},
              "updatemenus": [{"type": "dropdown", "buttons": buttons, "x": 0.01, "y": 1.0, "xanchor": "left",
                               "bgcolor": "#161b22", "bordercolor": "#30363d", "font": {"color": "#e6edf3"}}],
              "margin": {"l": 0, "r": 0, "t": 60, "b": 0}}
    rows = "".join(f"<tr><td>{OPPNAME.get(m, m)}</td><td>{s['games']}</td><td>{s['win_pct']}</td><td>{s['us_goal_side_pct']}</td>"
                   f"<td>{s['us_air_pct']}</td><td>{s['us_high_pct_z_gt_300']}</td><td>{s['opp_air_pct']}</td>"
                   f"<td>{s['threat_us_goal_side_pct']}</td><td>{s['threat_us_in_box_pct']}</td></tr>" for m, s in st.items())
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>{label} 3D positioning</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>body{{background:#0f1117;color:#e6edf3;font-family:system-ui,sans-serif;margin:0}}#p{{width:100vw;height:82vh}}
table{{border-collapse:collapse;margin:8px 16px;font-size:13px}}td,th{{border:1px solid #30363d;padding:4px 8px;text-align:right}}
th{{color:#8b949e}}p{{margin:4px 16px;color:#8b949e;font-size:12px}}</style></head><body>
<div id="p"></div>
<p>Rotated so our goal is at −y. First 3 s of each game (kickoff) excluded. Voxels 512×512×256 uu; colour/opacity = % of non-demoed frames in that voxel. Use the dropdown to switch us / opponent / ball and vs Galactus / vs FR4. Games: poolkit kickoff, first goal wins, both sides per seed.</p>
<table><tr><th>vs</th><th>games</th><th>win %</th><th>we're goal-side %</th><th>we're airborne %</th><th>we're z&gt;300 %</th><th>opp airborne %</th><th>goal-side while they attack %</th><th>in our box while they attack %</th></tr>{rows}</table>
<script>Plotly.newPlot("p",{json.dumps(traces, separators=(",", ":"))},{json.dumps(layout, separators=(",", ":"))},{{responsive:true}});</script>
</body></html>"""
    open(path, "w").write(html)

# ------------------------------------------------------------------ matplotlib
def draw_wire3d(ax):
    for pl in field_lines():
        a = np.array(pl); ax.plot(a[:, 0], a[:, 1], a[:, 2], color=MUTED, lw=0.6, alpha=0.8)

def style3d(ax):
    ax.set_facecolor(BG)
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.set_pane_color((0.06, 0.07, 0.09, 1.0)); a._axinfo["grid"]["color"] = (0.19, 0.21, 0.24, 0.6)
    ax.set_xlim(-W, W); ax.set_ylim(-L - GD, L + GD); ax.set_zlim(0, H)
    ax.set_box_aspect((2 * W, 2 * (L + GD), H * 1.4))
    ax.set_xticks([-4000, 0, 4000]); ax.set_yticks([-5000, 0, 5000]); ax.set_zticks([0, 1000, 2000])
    ax.tick_params(labelsize=6, colors=MUTED)

def scatter_vox(ax, v, cmap, vmax=None):
    X, Y, Z = centers()
    vmax = vmax or np.percentile(v[v > 0], 99)
    m = v > 0.10 * vmax
    a = np.clip(v[m] / vmax, 0, 1) ** 0.6
    cols = plt.get_cmap("turbo")(0.15 + 0.85 * a); cols[:, 3] = 0.08 + 0.8 * a
    o = np.argsort(a)
    ax.scatter(X[m][o], Y[m][o], Z[m][o], s=4 + 90 * a[o], c=cols[o], edgecolors="none", depthshade=False)

def fig_3d(out, data, label, made):
    views = [(22, -60, "corner"), (12, -90, "behind our goal"), (55, -20, "high side")]
    ms = list(data)
    fig = plt.figure(figsize=(15, 4.6 * len(ms)))
    for r, m in enumerate(ms):
        v = vox(data[m]["u"])
        for c, (el, az, nm) in enumerate(views):
            ax = fig.add_subplot(len(ms), 3, r * 3 + c + 1, projection="3d")
            style3d(ax); draw_wire3d(ax); scatter_vox(ax, v, "Blues")
            ax.view_init(elev=el, azim=az)
            ax.set_title(f"us vs {OPPNAME.get(m, m)} · {nm}", fontsize=9, pad=0)
            ax.text(0, -L - GD, 0, "our goal", color="#f78166", fontsize=7); ax.text(0, L + GD, 0, "their goal", color="#58a6ff", fontsize=7)
    fig.suptitle(f"{label}: where our car spends its time (3D voxels 512×512×256 uu; size, colour, opacity = share of frames; kickoffs excluded)", fontsize=12, fontweight="bold")
    fig.subplots_adjust(left=0, right=1, top=0.92, bottom=0.02, wspace=0.0, hspace=0.08)
    p = os.path.join(out, "positions_3d.png"); fig.savefig(p, dpi=90); plt.close(fig); made.append(p)

def field2d(ax):
    ax.add_patch(Rectangle((-W, -L), 2 * W, 2 * L, fill=False, ec=MUTED, lw=1))
    ax.plot([-W, W], [0, 0], color=MUTED, lw=0.6); ax.add_patch(plt.Circle((0, 0), 1000, fill=False, ec=MUTED, lw=0.6))
    ax.add_patch(Rectangle((-GW, -L - GD), 2 * GW, GD, fill=False, ec="#f78166", lw=1.4))
    ax.add_patch(Rectangle((-GW, L), 2 * GW, GD, fill=False, ec="#58a6ff", lw=1.4))
    ax.set_xlim(-W - 150, W + 150); ax.set_ylim(-L - GD - 150, L + GD + 150); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.grid(False)

def fig_topdown(out, data, threats, st, label, made):
    ms = list(data)
    panels = []
    for m in ms:
        panels += [(m, "u", None, f"us vs {OPPNAME.get(m, m)}", "Blues"), (m, "o", None, f"{OPPNAME.get(m, m)}", "Reds"),
                   (m, "u", "threat", f"us while {OPPNAME.get(m, m)} attacks", "Purples")]
    fig, axs = plt.subplots(len(ms), 3, figsize=(10.5, 6.8 * len(ms)))
    axs = np.atleast_2d(axs)
    for ax, (m, e, cond, t, cm) in zip(axs.flat, panels):
        p = data[m][e]
        if cond == "threat": p = p[threats[m]]
        field2d(ax)
        h, xe, ye = np.histogram2d(p[:, 0], p[:, 1], bins=(32, 40), range=(EX, EY))
        h = 100 * h / max(1, len(p))
        ax.imshow(h.T, origin="lower", extent=(EX[0], EX[1], EY[0], EY[1]), cmap=CM[cm],
                  norm=LogNorm(vmin=max(h[h > 0].min(), 1e-3), vmax=h.max()), interpolation="bilinear", alpha=0.95)
        field2d(ax)
        sub = ""
        if cond == "threat": sub = f"\n{len(p):,} frames · goal-side {st[m]['threat_us_goal_side_pct']}% · in box {st[m]['threat_us_in_box_pct']}%"
        elif e == "u": sub = f"\nown half {st[m]['us_own_half_pct']}% · goal-side {st[m]['us_goal_side_pct']}%"
        else: sub = f"\nits own half {st[m]['opp_own_half_pct']}% · goal-side {st[m]['opp_goal_side_pct']}%"
        ax.set_title(t + sub, fontsize=9)
        ax.text(0, -L - GD - 120, "our goal", ha="center", va="top", color="#f78166", fontsize=7)
    fig.suptitle(f"{label}: top-down position density (log colour; our goal at the bottom; first 3 s of each game excluded)\n"
                 "'attacks' = ball in our half moving at our goal (vy < −300) and the opponent closer to it", fontsize=11, fontweight="bold")
    fig.tight_layout(); p = os.path.join(out, "positions_topdown.png"); fig.savefig(p, dpi=90); plt.close(fig); made.append(p)

def fig_height(out, data, st, label, made):
    ms = list(data)
    fig = plt.figure(figsize=(14, 7.4))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.4, 1.4, 1])
    for r, m in enumerate(ms[:2]):
        ax = fig.add_subplot(gs[r, 0:2])
        u = data[m]["u"]
        h, _, _ = np.histogram2d(u[:, 1], u[:, 2], bins=(80, 32), range=(EY, EZ)); h = 100 * h / len(u)
        ax.imshow(h.T, origin="lower", extent=(EY[0], EY[1], EZ[0], EZ[1]), aspect="auto", cmap=CM["Blues"],
                  norm=LogNorm(vmin=max(h[h > 0].min(), 1e-4), vmax=h.max()))
        ax.axvline(0, color=MUTED, lw=0.6); ax.axhline(GH, color=MUTED, ls=":", lw=0.6)
        ax.add_patch(Rectangle((-L - 300, 0), 300, GH, color="#f78166", alpha=0.6)); ax.add_patch(Rectangle((L, 0), 300, GH, color="#58a6ff", alpha=0.6))
        ax.set_xlim(-L - 300, L + 300); ax.set_ylim(0, H); ax.grid(False)
        ax.set_ylabel("height z (uu)"); ax.set_title(f"Side view, us vs {OPPNAME.get(m, m)} (y = our goal → their goal; dotted = crossbar 642)")
        if r == len(ms[:2]) - 1: ax.set_xlabel("y (uu)")
    ax = fig.add_subplot(gs[:, 2])
    bins = np.array([0, 50, 150, 300, 500, 800, 1200, 1600, 2100])
    labs = ["<50", "50–150", "150–300", "300–500", "500–800", "800–1200", "1200–1600", ">1600"]
    cols = {"u": ["#58a6ff", "#a5d6ff"], "o": ["#f78166", "#ffa198"]}
    yy = np.arange(len(labs)); w = 0.2; k = 0
    for m in ms:
        for e in ("u", "o"):
            c, _ = np.histogram(data[m][e][:, 2], bins=bins)
            ax.barh(yy + (k - 1.5) * w, 100 * c / c.sum(), w, color=cols[e][ms.index(m) % 2],
                    label=f"{'us' if e == 'u' else OPPNAME.get(m, m)} (vs {OPPNAME.get(m, m)})" if e == "u" else OPPNAME.get(m, m))
            k += 1
    ax.set_yticks(yy, labs); ax.set_xscale("log"); ax.set_xlabel("% of frames (log)"); ax.set_ylabel("car height z (uu)")
    ax.set_title("Height profile"); ax.legend(fontsize=7); ax.grid(True, axis="x", alpha=0.4)
    fig.suptitle(f"{label}: height profile ({', '.join(f'airborne vs {OPPNAME.get(m, m)}: us {st[m]['us_air_pct']}% / them {st[m]['opp_air_pct']}%' for m in ms)})",
                 fontsize=11, fontweight="bold")
    fig.tight_layout(); p = os.path.join(out, "positions_height.png"); fig.savefig(p, dpi=90); plt.close(fig); made.append(p)

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inputs", action="append", required=True, help="name=path.jsonl (name: galactus, fr4, ...)")
    ap.add_argument("--label", default="ST25")
    ap.add_argument("--skip-kickoff", type=float, default=3.0, help="drop the first N seconds of each game (kickoff paths)"); ap.add_argument("--out", default="docs/figures/positions")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    try: os.nice(19)
    except Exception: pass
    data, st, threats = {}, {}, {}
    for s in a.inputs:
        k, p = s.split("=", 1); d = load(os.path.expanduser(p), int(a.skip_kickoff * 120)); data[k] = d
        st[k], threats[k] = stats(d); log(k, json.dumps(st[k]))
    made = []
    write_html(os.path.join(a.out, "positions_3d.html"), data, a.label, st); made.append(os.path.join(a.out, "positions_3d.html"))
    fig_3d(a.out, data, a.label, made); fig_topdown(a.out, data, threats, st, a.label, made); fig_height(a.out, data, st, a.label, made)
    json.dump({"label": a.label, "inputs": a.inputs, "skip_kickoff_s": a.skip_kickoff, "stats": st}, open(os.path.join(a.out, "positions_stats.json"), "w"), indent=1)
    for p in made: log("wrote", p, "%.0f KB" % (os.path.getsize(p) / 1024))

if __name__ == "__main__":
    main()
