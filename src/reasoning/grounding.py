"""Support-selected quantitative observations for local language hypotheses.

This supplies measurements, not primitive shape labels or an answer key. Axes
are selected from supports only; queries cannot change them. Photo-like supports
disable ink geometry automatically because thresholded photographic regions do
not have the semantics of objects in a line drawing.
"""
from __future__ import annotations

import json
import numpy as np

from src.vision.structural_features import StructuralEncoder


class MeasurementGrounding:
    def __init__(self, *, max_axes=8):
        self.encoder = StructuralEncoder()
        self.max_axes, self.indices, self.support_text = max_axes, (), ""

    @property
    def names(self): return tuple(self.encoder.names[i] for i in self.indices)

    def fit(self, positive, negative):
        batch = self.encoder.encode(list(positive)+list(negative))
        n = len(positive)
        entropy = self.encoder.names.index("quality.gray_entropy")
        saturation = self.encoder.names.index("quality.saturation_p90")
        drawing_like = (batch.values[:, entropy] < .55) & (batch.values[:, saturation] < .15)
        if np.mean(drawing_like) < .8: return self
        x = batch.values.astype(float)
        difference = np.abs(x[:n].mean(axis=0)-x[n:].mean(axis=0))
        scale = x[:n].std(axis=0)+x[n:].std(axis=0)
        scores = difference/np.maximum(scale, np.maximum(np.abs(x).mean(axis=0)*.03, 1e-5))
        allowed = batch.valid.all(axis=0) & np.array([not name.startswith("quality.") for name in batch.names])
        selected = []
        for i in np.argsort(-scores, kind="stable"):
            if not allowed[i] or scores[i] < 1.: continue
            # Avoid eight near-identical quantiles of the same measurement.
            if any(abs(np.corrcoef(x[:, i], x[:, j])[0, 1]) > .97 for j in selected): continue
            selected.append(int(i))
            if len(selected) >= self.max_axes: break
        self.indices = tuple(selected)
        rows = []
        for i, values in enumerate(x):
            label = f"A{i+1}" if i < n else f"B{i-n+1}"
            rows.append({"image": label, "values": [round(float(values[j]), 5) for j in selected]})
        if selected:
            self.support_text = (" Approximate measured drawing observations follow. Coordinates, lengths and areas are normalized "
                                 "to the panel. They are candidate evidence, not an intended-rule oracle. Compare them with the "
                                 "actual images; a correlation can be accidental. Do not mistake an outline for an open curve. "
                                 "Measurements: " + json.dumps({"columns": list(self.names), "rows": rows}))
        return self

    def describe(self, image):
        if not self.indices: return ""
        batch = self.encoder.encode([image])
        values = {batch.names[i]: round(float(batch.values[0, i]), 5) if batch.valid[0, i] else None for i in self.indices}
        return " Approximate measured observations of this image (no group label): " + json.dumps(values)
