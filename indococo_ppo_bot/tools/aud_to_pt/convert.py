"""Convert an indococo-train exported actor (.safetensors, PyTorch layout) into a
rlbot_submission-style 1v1.pt (plain state_dict, DiscreteFF keys model.{0,2,4,6,8})."""
import sys, collections, torch
import json, struct, numpy as np
def load_file(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]; header = json.loads(f.read(n)); data = f.read()
    dt = {"F32": np.float32, "F16": np.float16, "F64": np.float64}
    return {k: torch.from_numpy(np.frombuffer(data[m["data_offsets"][0]:m["data_offsets"][1]], dtype=dt[m["dtype"]]).reshape(m["shape"]).astype(np.float32))
            for k, m in header.items() if k != "__metadata__"}
src, ref, out = sys.argv[1:4]
st = load_file(src)
refsd = torch.load(ref, map_location="cpu")
assert isinstance(refsd, dict), type(refsd)
sd = collections.OrderedDict()
for k, v in refsd.items():
    w = st[k]
    assert w.shape == v.shape, (k, w.shape, v.shape)
    sd[k] = w.to(v.dtype).contiguous().clone()
extra = set(st) - set(sd)
assert not extra, extra
torch.save(sd, out)
print("wrote", out, [(k, tuple(v.shape)) for k, v in sd.items()])
