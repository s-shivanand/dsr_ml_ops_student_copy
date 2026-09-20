"""
inspect_onnx.py — Read the embedded schema from an ONNX file and run it.

Demonstrates the slide's point about ONNX:
1. Every .onnx file embeds typed input/output tensor specs in the binary.
2. The runtime reads them before inference and rejects mismatched inputs.

Usage:
    python inspect_onnx.py

Requires:
    python export_onnx.py   (to produce iris_classifier.onnx first)
"""

import pathlib

import numpy as np
import onnx
import onnxruntime as ort

HERE = pathlib.Path(__file__).resolve().parent
ONNX_PATH = HERE / "iris_classifier.onnx"

# ── 1. Read the embedded schema without running anything ──────────────────────
model = onnx.load(str(ONNX_PATH))
graph = model.graph

print("=" * 70)
print("Embedded schema (from the .onnx binary)")
print("=" * 70)
for name, tensor in [("inputs", graph.input), ("outputs", graph.output)]:
    print(f"\n[{name}]")
    for t in tensor:
        elem = onnx.TensorProto.DataType.Name(t.type.tensor_type.elem_type)
        dims = [
            d.dim_value if d.HasField("dim_value") else d.dim_param
            for d in t.type.tensor_type.shape.dim
        ]
        print(f"  {t.name:18s}  {elem:<10s}  shape={dims}")

# ── 2. Run inference on the setosa sample ─────────────────────────────────────
session = ort.InferenceSession(str(ONNX_PATH))

sample = np.array([[5.1, 3.5, 1.4, 0.2]], dtype=np.float32)
label, proba = session.run(None, {"float_input": sample})

print("\n" + "=" * 70)
print("Inference (setosa sample)")
print("=" * 70)
print(f"  label        : {label[0]}")
print(f"  probabilities: {np.round(proba[0], 4)}")

# ── 3. Show the runtime rejecting a mismatched input ──────────────────────────
print("\n" + "=" * 70)
print("Mismatched input (wrong shape: 3 features instead of 4)")
print("=" * 70)
bad_sample = np.array([[5.1, 3.5, 1.4]], dtype=np.float32)
try:
    session.run(None, {"float_input": bad_sample})
except Exception as err:  # noqa: BLE001 — deliberately show the rejection
    print(f"  REJECTED: {type(err).__name__}: {err}")
