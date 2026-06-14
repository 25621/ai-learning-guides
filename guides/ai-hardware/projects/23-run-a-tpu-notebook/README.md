# Run a TPU Notebook

## Key Insight

Training and running machine learning models on a Google [TPU](/shared/glossary/#tpu) requires compiling computations using the [XLA](/shared/glossary/#xla) compiler instead of relying on standard GPU runtime paradigms. By utilizing high-level frameworks like [JAX](/shared/glossary/#jax) or PyTorch/XLA, developers can write models in Python and let the backend optimize the execution graph for the TPU's matrix multiply units. This workflow is crucial for scaling up training across large [TPU pods](/shared/glossary/#tpu-pod), where matrix operations are executed with extreme hardware efficiency.
