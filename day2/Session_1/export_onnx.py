"""
export_onnx.py — Convert the sklearn iris pipeline to ONNX.

The ONNX format embeds typed input/output tensor specs directly in the
binary. Converting from sklearn *forces* you to declare those types up
front (see `initial_types` below), which catches train/serve skew at
export time rather than at runtime.

Usage:
    python export_onnx.py

Outputs:
    iris_classifier.onnx — the exported model (typed graph + weights)
"""

import pathlib

from skl2onnx import to_onnx
from skl2onnx.common.data_types import FloatTensorType

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

HERE      = pathlib.Path(__file__).resolve().parent
DATA_PATH = HERE.parent.parent / "day1" / "Session_1" / "iris.csv"

FEATURE_COLS = ["septal_length", "sepal_width", "petal_length", "petal_width"]
TARGET_COL   = "class"

# Load and fit the same pipeline as train.py.
df = pd.read_csv(DATA_PATH)
X = df[FEATURE_COLS].values
y = df[TARGET_COL].values

pipeline = Pipeline([
    ("scaler",     StandardScaler()),
    ("classifier", LogisticRegression(max_iter=300, random_state=42)),
])
pipeline.fit(X, y)

# ── The key step: declare the input schema explicitly ────────────────────────
# FloatTensorType([None, 4]) = a batch of float rows, each with 4 features.
# If the serving side expects a different shape or dtype, the mismatch is
# discoverable from the file itself before anything is deployed.
initial_types = [("float_input", FloatTensorType([None, 4]))]

# zipmap=False keeps outputs as plain tensors (label + probability matrix)
# instead of a list-of-dicts, which is friendlier for inspection.
onx = to_onnx(
    pipeline,
    initial_types=initial_types,
    options={"zipmap": False},
)

onnx_path = HERE / "iris_classifier.onnx"
onnx_path.write_bytes(onx.SerializeToString())

print(f"Model exported to {onnx_path}")
print("Input  : float_input  float[N, 4]")
print("Output : label string[N] | probabilities float[N, 3]")
