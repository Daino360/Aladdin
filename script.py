import torch, re, pprint

CKPT = "checkpoints/GenieRedux_Guided_CoinRun_80mln_v1.0/model.pt"

ckpt = torch.load(CKPT, map_location="cpu")
sd = ckpt.get("model", ckpt)

print("Top-level keys in checkpoint['model']:", list(sd.keys()))

# If there is a saved config, print a summary — it often contains dims.
cfg = ckpt.get("config")
if cfg:
    print("\n--- Saved config (keys only) ---")
    print(list(cfg.keys()))
    print("\nconfig snippet:")
    pprint.pp({k: cfg[k] for k in list(cfg.keys())[:10]})
else:
    print("\n(No saved 'config' found in checkpoint)")

# Grab the dynamics sub-dict (what your loader uses)
dyn = sd.get("dynamics")
if not isinstance(dyn, dict):
    raise SystemExit("No 'dynamics' dict found in checkpoint['model']")

print("\nNumber of dynamics params:", len(dyn))

def grep(sub):
    return {k: v for k, v in dyn.items() if sub in k}

def first_shape(d):
    return next(iter(d.values())).shape if d else None

# Try common names
action = grep("action")
cond   = grep("cond") | grep("condition") | grep("task") | grep("game")
inp    = grep("input") | grep("in_proj") | grep("img_proj") | grep("patch") | grep("stem") | grep("proj")
pos    = grep("pos")

print("\n=== Guessing key shapes ===")
if action:
    for k,v in list(action.items())[:5]:
        print("action key:", k, tuple(v.shape))
else:
    print("action embedding not found by substring")

if cond:
    for k,v in list(cond.items())[:5]:
        print("cond key:", k, tuple(v.shape))
else:
    print("conditioning not found by substring")

if inp:
    for k,v in list(inp.items())[:5]:
        print("input key:", k, tuple(v.shape))
else:
    print("input/patch proj not found by substrings")

if pos:
    for k,v in list(pos.items())[:3]:
        print("pos key:", k, tuple(v.shape))

# Heuristics: try to infer in_channels/patch_size from any 4D conv weight
conv4 = [(k,v) for k,v in dyn.items() if hasattr(v, "dim") and v.dim()==4]
if conv4:
    k,v = conv4[0]
    out_c, in_c, kh, kw = v.shape
    print(f"\nFirst conv-like weight: {k} shape={tuple(v.shape)} -> in_channels={in_c}, patch={kh}x{kw}")
else:
    print("\nNo 4D conv weights found (model may be pure MLP/Transformer on tokens)")

# Estimate number of layers
layer_names = set()
for k in dyn:
    m = re.search(r"(layers|blocks)\.(\d+)", k)
    if m: layer_names.add(m.group(2))
print("estimated dynamics layers:", len(layer_names))

# Show a small sample of keys to help manual mapping
print("\n--- Sample of dynamics keys ---")
for k in list(dyn.keys())[:40]:
    shp = getattr(dyn[k], "shape", None)
    print(" ", k, tuple(shp) if shp is not None else type(dyn[k]))
