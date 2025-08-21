import inspect
import sys

try:
    from fightingcv_attention.attention.ViP import WeightedPermuteMLP as WPM
except Exception as e:
    print("IMPORT_ERROR:", e)
    sys.exit(1)

print("Class:", WPM)
try:
    src = inspect.getsource(WPM)
    print("=== SOURCE START ===")
    print(src)
    print("=== SOURCE END ===")
except Exception as e:
    print("GETSOURCE_ERROR:", e)

# Try to locate forward function source
try:
    import ast
    src = inspect.getsource(WPM)
    mod = ast.parse(src)
    for node in ast.walk(mod):
        if isinstance(node, ast.FunctionDef) and node.name == 'forward':
            print("Found forward definition at line:", node.lineno)
            break
except Exception as e:
    pass

# Try dry-run on both layouts with d_model=512, H=W=16
import torch

model = WPM(512, seg_dim=8)
print("Model created:", model.__class__.__name__)

x_bhwc = torch.randn(1, 16, 16, 512)
x_bchw = torch.randn(1, 512, 16, 16)

for name, x in [("BHWC", x_bhwc), ("BCHW", x_bchw)]:
    try:
        y = model(x)
        print(f"{name} -> OK, out shape:", tuple(y.shape))
    except Exception as e:
        print(f"{name} -> ERROR:", repr(e))