# Export to ONNX

---

> Save the model as a portable graph, then run it anywhere — no PyTorch needed.

---

## Key Insight

[ONNX](/shared/glossary/#onnx) is a framework-neutral file format that stores a model as a graph of operations. Exporting a [CNN](/shared/glossary/#cnn) to ONNX lets you run it with a separate engine like [ONNX Runtime](/shared/glossary/#onnx-runtime), on hardware or in environments where PyTorch is not installed.

## Why This Matters

Many production and edge targets cannot install PyTorch but can run ONNX. Comparing the ONNX output against PyTorch confirms the export preserved the model's math instead of silently changing it.
