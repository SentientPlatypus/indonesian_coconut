#!/usr/bin/env python3
"""Rocket League standard Soccar arena model (uu) + 3D positioning heatmaps rendered inside it.

Geometry (RocketSim / RLBot field constants):
  floor x +-4096, y +-5120, ceiling 2044; 45 deg corner walls on |x|+|y| = 8064 (1152 uu cut),
  rounded vertical corners, ~256 uu floor/wall and wall/ceiling transitions,
  goals 1786 wide (x +-893), 642.775 tall, 880 deep behind y = +-5120, 6 big + 28 small boost pads.
Our goal (blue) is at -y, theirs (orange) at +y, matching make_positions.py.

Library use (needs plotly; kaleido==0.2.1 for PNGs):
  import arena
  layers = [arena.Layer("us vs Galactus", pts_xyz, "blue"), ...]
  fig = arena.figure(layers, title="...")
  arena.write_html(fig, "arena_heatmap.html"); arena.write_views(fig, out_dir)
"""
import math
import numpy as np

L, W, H = 5120.0, 4096.0, 2044.0
CORNER = 1152.0                    # corner cut: diagonal walls on |x|+|y| = W+L-CORNER = 8064
GW, GH, GD = 893.0, 642.775, 880.0
R_FILLET = 256.0                   # floor<->wall and wall<->ceiling transition radius
R_VCORNER = 420.0                  # rounding of the vertical corner seams
BLUE, ORANGE = "#2f8cff", "#ff8a1f"
BG = "#05070b"

BIG_PADS = [(-3584, 0), (3584, 0), (-3072, -4096), (3072, -4096), (-3072, 4096), (3072, 4096)]
_SMALL_HALF = [(0, -4240), (-1792, -4184), (1792, -4184), (-940, -3308), (940, -3308), (0, -2816),
               (-3584, -2484), (3584, -2484), (-1788, -2300), (1788, -2300), (-2048, -1036), (0, -1024),
               (2048, -1036)]
SMALL_PADS = _SMALL_HALF + [(-1024, 0), (1024, 0)] + [(x, -y) for x, y in _SMALL_HALF]
assert len(SMALL_PADS) == 28 and len(BIG_PADS) == 6

# ------------------------------------------------------------------ footprint
def _octagon():
    c = CORNER
    return np.array([(W, -L + c), (W, L - c), (W - c, L), (-W + c, L), (-W, L - c), (-W, -L + c),
                     (-W + c, -L), (W - c, -L)], float)          # counter-clockwise

def footprint(step=70.0):
    """Closed wall footprint (rounded octagon) sampled ~every `step` uu.
    Returns P (N,2) points, Nrm (N,2) outward normals, back (N,) bool = on a goal back wall straight.
    Back walls get exact samples at x = +-GW so the goal mouth can be cut cleanly."""
    V = _octagon(); n = len(V); R = R_VCORNER
    segs = []                       # ('line', a, b) / ('arc', centre, a0, a1)
    tang = []
    for i in range(n):
        p0, p1, p2 = V[i - 1], V[i], V[(i + 1) % n]
        d0 = (p1 - p0) / np.linalg.norm(p1 - p0); d1 = (p2 - p1) / np.linalg.norm(p2 - p1)
        turn = math.acos(np.clip(np.dot(d0, d1), -1, 1)); t = R * math.tan(turn / 2)
        a, b = p1 - d0 * t, p1 + d1 * t
        cen = a + np.array([-d0[1], d0[0]]) * R                 # left normal (CCW polygon -> inside)
        tang.append((a, b, cen))
    for i in range(n):
        a, b, cen = tang[i]
        a0 = math.atan2(a[1] - cen[1], a[0] - cen[0]); a1 = math.atan2(b[1] - cen[1], b[0] - cen[0])
        if a1 < a0: a1 += 2 * math.pi
        segs.append(("arc", cen, a0, a1))
        nb = tang[(i + 1) % n][0]
        segs.append(("line", b, nb))
    P, N, BK = [], [], []
    for s in segs:
        if s[0] == "arc":
            _, cen, a0, a1 = s; k = max(3, int(math.ceil(R * (a1 - a0) / step)))
            for t in np.linspace(a0, a1, k, endpoint=False):
                v = np.array([math.cos(t), math.sin(t)]); P.append(cen + R * v); N.append(v); BK.append(False)
        else:
            _, a, b = s; d = b - a; ln = np.linalg.norm(d); nrm = np.array([d[1], -d[0]]) / ln
            back = abs(nrm[1]) > 0.99
            cuts = [0.0, 1.0]
            if back:
                for gx in (-GW, GW):
                    f = (gx - a[0]) / d[0]
                    if 0 < f < 1: cuts.append(f)
            cuts = sorted(cuts)
            for c0, c1 in zip(cuts[:-1], cuts[1:]):
                k = max(1, int(math.ceil(ln * (c1 - c0) / step)))
                for f in np.linspace(c0, c1, k, endpoint=False):
                    P.append(a + d * f); N.append(nrm); BK.append(back)
    return np.array(P), np.array(N), np.array(BK)

def _profile(r=R_FILLET, n_arc=7):
    """Wall cross-section: (inward distance from wall d, height z), floor tangent -> wall -> ceiling tangent."""
    th = np.linspace(0, math.pi / 2, n_arc)
    lo = [(r - r * math.sin(t), r - r * math.cos(t)) for t in th]
    mid_z = sorted(set([r, GH, 900.0, 1200.0, 1500.0, H - r]))
    mid = [(0.0, z) for z in mid_z[1:-1]]
    hi = [(r - r * math.cos(t), H - r + r * math.sin(t)) for t in th]
    return np.array(lo + mid + hi)

def wall_mesh(step=70.0):
    """Arena skin (fillets + walls + ceiling transition) as a triangle mesh with the goal mouths cut out.
    Returns X,Y,Z vertex arrays and I,J,K triangles."""
    P, N, BK = footprint(step); prof = _profile()
    ns, nt = len(P), len(prof)
    X = (P[:, None, 0] - N[:, None, 0] * prof[None, :, 0]).ravel()
    Y = (P[:, None, 1] - N[:, None, 1] * prof[None, :, 0]).ravel()
    Z = np.broadcast_to(prof[None, :, 1], (ns, nt)).ravel().copy()
    I, J, K = [], [], []
    for s in range(ns):
        s2 = (s + 1) % ns
        for t in range(nt - 1):
            a, b, c, d = s * nt + t, s2 * nt + t, s2 * nt + t + 1, s * nt + t + 1
            cx = (X[a] + X[b] + X[c] + X[d]) / 4; cz = (Z[a] + Z[b] + Z[c] + Z[d]) / 4
            if BK[s] and BK[s2] and abs(cx) < GW and cz < GH: continue          # goal mouth
            I += [a, a]; J += [b, c]; K += [c, d]
    return X, Y, Z, np.array(I), np.array(J), np.array(K)

_PATH = None
_FOOT = None
def floor_mask(x, y, inset=R_FILLET - 40):
    """Flat-floor region: footprint shrunk to where the curved transition starts, plus goal floors."""
    global _FOOT
    from matplotlib.path import Path
    if _FOOT is None:
        P, N, _ = footprint(40); _FOOT = Path(P - N * inset)
    x = np.asarray(x, float); y = np.asarray(y, float)
    f = _FOOT.contains_points(np.c_[x.ravel(), y.ravel()], radius=1.0).reshape(x.shape)
    return f | ((np.abs(x) <= GW) & (np.abs(y) <= L + GD))
def inside(x, y, pad=0.0):
    """Point inside the rounded arena footprint (or the octagon shrunk by pad) or inside a goal."""
    global _PATH
    x = np.asarray(x, float); y = np.asarray(y, float)
    if pad == 0.0:
        from matplotlib.path import Path
        if _PATH is None: _PATH = Path(footprint(40)[0])
        f = _PATH.contains_points(np.c_[x.ravel(), y.ravel()], radius=-1.0).reshape(x.shape) | \
            _PATH.contains_points(np.c_[x.ravel(), y.ravel()], radius=1.0).reshape(x.shape)
    else:
        f = (np.abs(x) <= W - pad) & (np.abs(y) <= L - pad) & (np.abs(x) + np.abs(y) <= W + L - CORNER - pad * 1.414)
    g = (np.abs(x) <= GW) & (np.abs(y) <= L + GD)
    return f | g

# ------------------------------------------------------------------ line work
def _pl(*pts): return [tuple(p) for p in pts]

def edge_rings():
    """Floor seam, wall-top seam, plus vertical ribs; each as (polyline Nx3)."""
    P, N, BK = footprint(60)
    floor = np.c_[P - N * R_FILLET, np.full(len(P), 2.0)]
    top = np.c_[P - N * R_FILLET, np.full(len(P), H)]
    mid = np.c_[P, np.full(len(P), R_FILLET)]
    up = np.c_[P, np.full(len(P), H - R_FILLET)]
    def split(ring, cut_goal):
        out, cur = [], []
        for i in range(len(ring) + 1):
            p = ring[i % len(ring)]
            if cut_goal and abs(p[0]) < GW - 1 and abs(p[1]) > L - 400:
                if cur: out.append(np.array(cur)); cur = []
                continue
            cur.append(p)
        if cur: out.append(np.array(cur))
        return out
    return {"floor": split(floor, True), "low": split(mid, True), "top": [np.vstack([top, top[:1]])],
            "high": [np.vstack([up, up[:1]])]}

def wall_ribs(every=1100.0):
    P, N, BK = footprint(20); prof = _profile()
    seg = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    out = []
    for s0 in np.arange(0, seg[-1], every):
        i = int(np.searchsorted(seg, s0))
        if BK[i] and abs(P[i, 0]) < GW + 60: continue
        out.append(np.c_[P[i, 0] - N[i, 0] * prof[:, 0], P[i, 1] - N[i, 1] * prof[:, 0], prof[:, 1]])
    return out

def goal_frames(side):
    """side=-1 our (blue) goal at -y, +1 theirs (orange)."""
    y0, y1 = side * L, side * (L + GD)
    mouth = _pl((-GW, y0, 0), (-GW, y0, GH), (GW, y0, GH), (GW, y0, 0))
    back = _pl((-GW, y1, 0), (-GW, y1, GH), (GW, y1, GH), (GW, y1, 0), (-GW, y1, 0))
    rails = [_pl((x, y0, z), (x, y1, z)) for x in (-GW, GW) for z in (0, GH)]
    return [mouth, back] + rails

def goal_mesh(side):
    y0, y1 = side * L, side * (L + GD)
    v = np.array([(-GW, y0, 0), (GW, y0, 0), (GW, y1, 0), (-GW, y1, 0),
                  (-GW, y0, GH), (GW, y0, GH), (GW, y1, GH), (-GW, y1, GH)])
    tri = [(3, 2, 6), (3, 6, 7), (0, 3, 7), (0, 7, 4), (1, 2, 6), (1, 6, 5), (4, 5, 6), (4, 6, 7)]  # back, sides, roof
    return v, np.array(tri)

def field_markings():
    z = 3.0; P = []
    P.append(_pl((-W + 40, 0, z), (W - 40, 0, z)))
    th = np.linspace(0, 2 * math.pi, 90)
    P.append([(1000 * math.cos(t), 1000 * math.sin(t), z) for t in th])
    P.append([(150 * math.cos(t), 150 * math.sin(t), z) for t in th])
    for s in (-1, 1):
        yb = s * (L - R_FILLET + 40)          # boxes start where the flat floor does
        P.append(_pl((-1800, yb, z), (-1800, yb - s * 1300, z), (1800, yb - s * 1300, z), (1800, yb, z)))   # big box
        P.append(_pl((-1100, yb, z), (-1100, yb - s * 600, z), (1100, yb - s * 600, z), (1100, yb, z)))     # goal box
        a = np.linspace(math.pi * 0.18, math.pi * 0.82, 40)
        P.append([(1150 * math.cos(t), yb - s * (1300 + 0) - s * (1150 * math.sin(t) - 1150 * math.sin(a[0])), z) for t in a])
    return P

# ------------------------------------------------------------------ densities
class Layer:
    """One density layer. min_z: cloud only shows frames above this height (ground time is the floor texture);
    set 0 for sparse layers (e.g. defence-only frames) so their ground positions get dots too."""
    def __init__(self, name, pts, hue, visible=False, min_z=110.0):
        self.name, self.pts, self.hue, self.visible, self.min_z = name, np.asarray(pts, float), hue, visible, min_z

HUES = {  # dim -> bright -> white-hot, chosen to pop against dark glass and turf
    "blue": ["#0b2a55", "#1f7bff", "#6fd3ff", "#e8fbff"],
    "cyan": ["#063a3a", "#00c2c7", "#7ff7e8", "#f0fffd"],
    "orange": ["#4a1600", "#ff6a00", "#ffc04d", "#fff6dc"],
    "red": ["#4a0710", "#ff3b4f", "#ff9aa5", "#fff0f2"],
    "yellow": ["#3d3300", "#e8c300", "#fff27a", "#ffffff"],
    "violet": ["#2a0c4f", "#a050ff", "#d9a8ff", "#fbf3ff"],
    "green": ["#06341a", "#22c55e", "#9bf5b9", "#f2fff6"],
}
def _scale(h): c = HUES[h]; return [[0, c[1]], [0.3, c[2]], [1, c[3]]]

def cloud(pts, vox=130.0, mass=0.9, max_pts=4500, min_z=110.0, seed=0):
    """Voxel point-density of airborne / wall / ceiling time (z > min_z; ground time is the floor texture):
    centres of the densest voxels covering `mass` of those frames, jittered inside the voxel so the
    cloud does not read as a lattice. Returns xyz (M,3), pct-of-ALL-frames (M,)."""
    if len(pts) == 0: return np.zeros((0, 3)), np.zeros(0)
    n_all = len(pts); pts = pts[pts[:, 2] > min_z]
    if len(pts) == 0: return np.zeros((0, 3)), np.zeros(0)
    ext = [(-W, W), (-L - GD, L + GD), (0, H)]
    nb = [int(round((e[1] - e[0]) / vox)) for e in ext]
    h, edges = np.histogramdd(np.clip(pts, [e[0] for e in ext], [e[1] - 1e-3 for e in ext]), bins=nb, range=ext)
    h = 100.0 * h / n_all
    idx = np.argwhere(h > 0); v = h[h > 0]
    o = np.argsort(-v); v, idx = v[o], idx[o]
    k = min(max_pts, int(np.searchsorted(np.cumsum(v), mass * v.sum())) + 1)
    v, idx = v[:k], idx[:k]
    c = [(edges[d][:-1] + edges[d][1:]) / 2 for d in range(3)]
    xyz = np.c_[c[0][idx[:, 0]], c[1][idx[:, 1]], c[2][idx[:, 2]]]
    xyz += np.random.default_rng(seed).uniform(-0.45 * vox, 0.45 * vox, xyz.shape)
    xyz[:, 2] = np.maximum(xyz[:, 2], max(min_z, 25.0))
    # snap voxels hugging a surface onto it so wall/ceiling time sits ON the glass
    xyz[:, 2] = np.where(xyz[:, 2] > H - vox, H - 25, xyz[:, 2])
    return xyz, v

FLOOR_RES = 80.0
def floor_grid():
    xs = np.arange(-W, W + 1, FLOOR_RES); ys = np.arange(-L - GD, L + GD + 1, FLOOR_RES)
    return xs, ys

def floor_heat(pts, sigma=1.6):
    """Smoothed xy projection on the floor grid, normalised 0..1 (sqrt-compressed)."""
    from scipy.ndimage import gaussian_filter
    xs, ys = floor_grid()
    ex = np.r_[xs - FLOOR_RES / 2, xs[-1] + FLOOR_RES / 2]; ey = np.r_[ys - FLOOR_RES / 2, ys[-1] + FLOOR_RES / 2]
    h, _, _ = np.histogram2d(pts[:, 0], pts[:, 1], bins=(ex, ey)) if len(pts) else (np.zeros((len(xs), len(ys))), 0, 0)
    h = gaussian_filter(h, sigma)
    hi = np.percentile(h[h > 0], 99.5) if (h > 0).any() else 1.0
    return np.clip(h / hi, 0, 1) ** 0.9            # (nx, ny)

def turf_value(heat):
    """Map heat 0..1 to the floor colourscale: 0..0.08 = mowed turf stripes, above = heat."""
    xs, ys = floor_grid()
    stripe = ((np.floor((ys + L) / 512.0) % 2)[None, :] * np.ones((len(xs), 1)))
    base = 0.02 + 0.035 * stripe
    v = np.where(heat > 0.15, 0.09 + 0.91 * (heat - 0.15) / 0.85, base)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    v = np.where(floor_mask(X, Y), v, np.nan)
    return v.T                                       # plotly surface wants (ny, nx)

def floor_scale(hue):
    """Turf stripes for 0..0.085, then a team-hued heat ramp."""
    c = HUES[hue]
    return [[0.0, "#0c2716"], [0.055, "#14391f"], [0.085, "#174325"], [0.09, "#1d4a35"], [0.3, c[0]],
            [0.62, c[1]], [0.85, c[2]], [1.0, c[3]]]
FLOOR_SCALE = floor_scale("blue")

def wall_heat(pts, X, Y, Z, radius=230.0):
    """Per-vertex density of frames hugging the arena skin (within `radius`), 0..1."""
    from scipy.spatial import cKDTree
    if len(pts) == 0: return np.zeros(len(X))
    off = pts[pts[:, 2] > 140]                 # driving next to the wall is floor time, not wall time
    near = off[(np.abs(off[:, 0]) > W - 300) | (np.abs(off[:, 1]) > L - 300) | (off[:, 2] > H - 300)
               | (np.abs(off[:, 0]) + np.abs(off[:, 1]) > W + L - CORNER - 300)]
    if len(near) == 0: return np.zeros(len(X))
    cnt = cKDTree(near).query_ball_point(np.c_[X, Y, Z], radius, return_length=True).astype(float)
    cnt = 100.0 * cnt / len(pts)
    hi = np.percentile(cnt[cnt > 0], 98) if (cnt > 0).any() else 1.0
    return np.sqrt(np.clip(cnt / hi, 0, 1))

WALL_SCALE = [[0.0, "#122036"], [0.08, "#18294a"], [0.25, "#6a2bb0"], [0.5, "#ff3d8b"], [0.78, "#ffb347"], [1.0, "#fffbe0"]]

# ------------------------------------------------------------------ figure
CAMERAS = {
    "Broadcast": dict(eye=dict(x=-1.3, y=0.0, z=0.85), center=dict(x=0, y=0, z=-0.15), up=dict(x=0, y=0, z=1), projection=dict(type="perspective")),
    "Behind our goal": dict(eye=dict(x=0.0, y=-1.72, z=0.6), center=dict(x=0, y=-0.05, z=-0.12), up=dict(x=0, y=0, z=1), projection=dict(type="perspective")),
    "Behind their goal": dict(eye=dict(x=0.0, y=1.72, z=0.6), center=dict(x=0, y=0.05, z=-0.12), up=dict(x=0, y=0, z=1), projection=dict(type="perspective")),
    "Top-down": dict(eye=dict(x=0.0, y=-0.0001, z=1.25), center=dict(x=0, y=0, z=0), up=dict(x=-1, y=0, z=0),
                     projection=dict(type="orthographic")),
    "Corner": dict(eye=dict(x=-1.25, y=-1.45, z=0.8), center=dict(x=0, y=0, z=-0.12), up=dict(x=0, y=0, z=1), projection=dict(type="perspective")),
}
ASPECT = dict(x=8192 / 11960 * 1.9, y=1.9, z=2044 / 11960 * 1.9)

def _line_trace(polys, color, width, opacity=1.0, name=None):
    xs, ys, zs = [], [], []
    for pl in polys:
        for p in pl: xs.append(float(p[0])); ys.append(float(p[1])); zs.append(float(p[2]))
        xs.append(None); ys.append(None); zs.append(None)
    return dict(type="scatter3d", mode="lines", x=xs, y=ys, z=zs, hoverinfo="skip", showlegend=False,
                line=dict(color=color, width=width), opacity=opacity, name=name or "lines")

def _team_split(polys):
    """Split polylines at y=0 into blue (our half) and orange halves."""
    b, o = [], []
    for pl in polys:
        pl = np.asarray(pl, float)
        for tgt, m in ((b, pl[:, 1] <= 60), (o, pl[:, 1] >= -60)):
            cur = []
            for p, k in zip(pl, m):
                if k: cur.append(p)
                elif cur: tgt.append(cur); cur = []
            if cur: tgt.append(cur)
    return b, o

def arena_traces(skin_intensity=None, ceiling=True):
    """Static arena traces. Returns (traces, index of the skin mesh)."""
    T = []
    X, Y, Z, I, J, K = wall_mesh()
    skin = dict(type="mesh3d", x=X.round(1), y=Y.round(1), z=Z.round(1), i=I, j=J, k=K, name="arena walls",
                intensity=(skin_intensity if skin_intensity is not None else np.zeros(len(X))).round(3),
                colorscale=WALL_SCALE, cmin=0, cmax=1, showscale=False, opacity=0.5, flatshading=False,
                lighting=dict(ambient=0.75, diffuse=0.5, specular=0.25, roughness=0.6, fresnel=0.6),
                hoverinfo="skip", showlegend=False)
    T.append(skin); skin_idx = 0
    if ceiling:
        P, N, _ = footprint(160); C = P - N * R_FILLET
        cx = np.r_[0.0, C[:, 0]]; cy = np.r_[0.0, C[:, 1]]; cz = np.full(len(cx), H)
        n = len(C); i = np.zeros(n, int); j = np.arange(1, n + 1); k = np.r_[np.arange(2, n + 1), 1]
        T.append(dict(type="mesh3d", x=cx, y=cy, z=cz, i=i, j=j, k=k, color="#1b2a44", opacity=0.10,
                      hoverinfo="skip", showlegend=False, name="ceiling"))
    for side, col in ((-1, BLUE), (1, ORANGE)):
        v, t = goal_mesh(side)
        T.append(dict(type="mesh3d", x=v[:, 0], y=v[:, 1], z=v[:, 2], i=t[:, 0], j=t[:, 1], k=t[:, 2], color=col,
                      opacity=0.33, hoverinfo="skip", showlegend=False, name="goal",
                      lighting=dict(ambient=0.9, diffuse=0.4)))
    # glow: wide faint line under a thin bright one
    rings = edge_rings(); ribs = wall_ribs()
    b, o = _team_split(rings["floor"] + rings["top"])
    for polys, col in ((b, BLUE), (o, ORANGE)):
        T.append(_line_trace(polys, col, 14, 0.18)); T.append(_line_trace(polys, col, 6, 0.35)); T.append(_line_trace(polys, col, 2.5, 1.0))
    T.append(_line_trace(rings["low"] + rings["high"] + ribs, "#6f86a8", 1.5, 0.35))
    for side, col in ((-1, BLUE), (1, ORANGE)):
        g = goal_frames(side)
        T.append(_line_trace(g, col, 16, 0.22)); T.append(_line_trace(g, "#ffffff" if False else col, 5, 1.0))
    T.append(_line_trace(field_markings(), "#d9e6dc", 3, 0.55))
    bx, by = zip(*BIG_PADS); sx, sy = zip(*SMALL_PADS)
    T.append(dict(type="scatter3d", mode="markers", x=bx, y=by, z=[30] * 6, name="big boost",
                  marker=dict(size=11, color="#ffb81c", opacity=0.95, symbol="circle"), hovertext="100 boost",
                  hoverinfo="text", showlegend=False))
    T.append(dict(type="scatter3d", mode="markers", x=sx, y=sy, z=[20] * 28, name="small boost",
                  marker=dict(size=4.5, color="#ffd866", opacity=0.9), hovertext="12 boost", hoverinfo="text",
                  showlegend=False))
    return T, skin_idx, (X, Y, Z)

def figure(layers, title="", floor_layer=0, subtitle=""):
    """Plotly figure (dict form): arena + per-layer point clouds (legend toggles) +
    floor/wall heat texture chosen from a dropdown + camera presets."""
    import plotly.graph_objects as go
    heats = [floor_heat(l.pts) for l in layers]
    WX, WY, WZ = wall_mesh()[:3]
    walls = [wall_heat(l.pts, WX, WY, WZ) for l in layers]
    T, skin_idx, _ = arena_traces(walls[floor_layer] if layers else None)
    xs, ys = floor_grid()
    floor_idx = len(T)
    XX, YY = np.meshgrid(xs, ys)
    T.insert(0, dict(type="surface", x=xs, y=ys, z=np.where(floor_mask(XX, YY), 0.0, np.nan),
                     surfacecolor=np.round(turf_value(heats[floor_layer]), 3) if layers else None,
                     colorscale=floor_scale(layers[floor_layer].hue) if layers else FLOOR_SCALE, cmin=0, cmax=1, showscale=False, hoverinfo="skip", name="floor heat",
                     lighting=dict(ambient=0.95, diffuse=0.2, specular=0.05), showlegend=False))
    skin_idx += 1; floor_idx = 0
    n_static = len(T)
    for l in layers:
        xyz, v = cloud(l.pts, min_z=l.min_z)
        lv = np.log10(v); lo, hi = (lv.min(), lv.max()) if len(v) else (0, 1)
        nv = (lv - lo) / max(hi - lo, 1e-9)
        T.append(dict(type="scatter3d", mode="markers", name=l.name, x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2].round(0),
                      visible=True if l.visible else "legendonly",
                      marker=dict(size=(2.0 + 6.0 * nv).round(1), color=nv.round(3), colorscale=_scale(l.hue),
                                  cmin=0, cmax=1, opacity=0.7, line=dict(width=0)),
                      customdata=v.round(4), hovertemplate=l.name + "<br>x %{x:.0f} y %{y:.0f} z %{z:.0f}"
                      "<br>%{customdata:.3f}% of frames<extra></extra>"))
    cam_btns = [dict(label=k, method="relayout", args=[{"scene.camera": c}]) for k, c in CAMERAS.items()]
    # restyle floor + wall skin together: two restyle calls are not allowed in one button, so use "update"
    floor_btns = [dict(label=f"Floor + wall heat: {l.name}", method="restyle",
                       args=[{"surfacecolor": [np.round(turf_value(heats[i]), 3), None],
                              "colorscale": [floor_scale(l.hue), WALL_SCALE],
                              "intensity": [None, walls[i].round(3)]}, [floor_idx, skin_idx]])
                  for i, l in enumerate(layers)]
    lay = dict(
        template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=BG, margin=dict(l=0, r=0, t=60, b=0),
        title=dict(text=f"<b>{title}</b>" + (f"<br><span style='font-size:12px;color:#8b949e'>{subtitle}</span>" if subtitle else ""),
                   x=0.5, y=0.97, font=dict(size=22, color="#e6edf3")),
        legend=dict(x=0.01, y=0.86, bgcolor="rgba(13,17,23,0.75)", bordercolor="#30363d", borderwidth=1,
                    font=dict(size=15), itemsizing="constant", title=dict(text="3D density (click to toggle)")),
        scene=dict(xaxis=dict(visible=False, range=[-W - 120, W + 120]), yaxis=dict(visible=False, range=[-L - GD - 60, L + GD + 60]),
                   zaxis=dict(visible=False, range=[-5, H + 40]), aspectmode="manual", aspectratio=ASPECT,
                   bgcolor=BG, camera=CAMERAS["Broadcast"], dragmode="orbit"),
        updatemenus=[dict(type="buttons", direction="right", x=0.5, xanchor="center", y=0.03, yanchor="bottom",
                          buttons=cam_btns, bgcolor="#161b22", bordercolor="#30363d", font=dict(color="#e6edf3"), showactive=False),
                     dict(type="dropdown", direction="down", x=0.99, xanchor="right", y=0.92, buttons=floor_btns,
                          active=floor_layer, bgcolor="#161b22", bordercolor="#30363d", font=dict(color="#e6edf3"))],
        annotations=[dict(text="blue = our goal (-y) · orange = their goal (+y) · floor and walls tinted by the selected layer",
                          x=0.99, y=0.02, xref="paper", yref="paper", xanchor="right", showarrow=False,
                          font=dict(size=11, color="#8b949e"))])
    fig = go.Figure(data=[_clean(t) for t in T], layout=lay)
    fig._arena = dict(n_static=n_static, floor_idx=floor_idx, skin_idx=skin_idx, heats=heats, walls=walls, hues=[l.hue for l in layers])
    return fig

def _clean(t):
    t = dict(t); typ = t.pop("type")
    import plotly.graph_objects as go
    cls = {"scatter3d": go.Scatter3d, "mesh3d": go.Mesh3d, "surface": go.Surface}[typ]
    return cls(**t)

def write_html(fig, path):
    fig.write_html(path, include_plotlyjs="cdn", full_html=True, config={"displaylogo": False, "responsive": True})

def set_view(fig, layer_on, floor_layer, camera, title=None):
    """Mutate fig for a static render: which cloud layers are shown, which heat skin, which camera."""
    a = fig._arena
    for i, tr in enumerate(fig.data[a["n_static"]:]):
        tr.visible = (i in layer_on)
    fig.data[a["floor_idx"]].surfacecolor = np.round(turf_value(a["heats"][floor_layer]), 3)
    fig.data[a["skin_idx"]].intensity = a["walls"][floor_layer].round(3)
    fig.data[a["floor_idx"]].colorscale = floor_scale(a["hues"][floor_layer])
    fig.layout.scene.camera = CAMERAS[camera]
    k = 1.38 if camera == "Top-down" else 1.0      # orthographic ignores eye distance; zoom via aspect
    fig.layout.scene.aspectratio = {a_: v * k for a_, v in ASPECT.items()}
    if title: fig.layout.title.text = title
    fig.layout.showlegend = camera != "Top-down"

def write_png(fig, path, width=1800, height=1100, scale=1):
    lay_backup = (fig.layout.updatemenus,)
    fig.layout.updatemenus = []
    try:
        fig.write_image(path, width=width, height=height, scale=scale, engine="kaleido")
    finally:
        fig.layout.updatemenus = lay_backup[0]
