"""Load the converted .pt exactly as rlbot_submission/bot.py does and compare with Rust probe."""
import ast, json, sys, os, collections, torch, numpy as np
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, "sub"))
from discrete import DiscreteFF
src = open(os.path.join(here, "sub/bot.py")).read()
fn = [n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "model_info_from_dict"][0]
ns = {"OrderedDict": collections.OrderedDict}; exec(compile(ast.Module([fn], []), "bot.py", "exec"), ns)
def load(path):
    sd = torch.load(path, map_location="cpu")
    inputs, n_actions, layers = ns["model_info_from_dict"](sd)
    assert inputs == 52 + 40 * 1
    pol = DiscreteFF(inputs, n_actions, layers, torch.device("cpu")); pol.load_state_dict(sd); pol.eval()
    return pol, (inputs, n_actions, layers)
pt = sys.argv[1]
pol, info = load(pt); print("bot.py view:", info)
ref, rinfo = load(os.path.join(here, "sub/orig_1v1.pt")); print("orig 1v1.pt view:", rinfo)
probe = np.array([np.sin(np.float32(i) * np.float32(0.37)) * 0.8 for i in range(92)], np.float32)
rust = np.array(json.loads(open(os.path.join(here, "probe_rust.txt")).read().split(" ", 1)[1]), np.float32)
with torch.no_grad(): py = pol.get_output(probe).numpy()
rp = np.exp(rust - rust.max()); rp /= rp.sum()
print("rust out sum", rust.sum()); rust = rp
print("probe: max|prob diff| %.3e argmax rust %d py %d" % (np.abs(py - rust).max(), rust.argmax(), py.argmax()))
