
**Use *triton client* to register an onnx model.**

```python

import tritonclient.grpc as triton
import numpy as np

# Create a Triton Inference Server client object
triton_client = triton.InferenceServerClient(url="localhost:8001")

# Create a model metadata object
model_metadata = triton.ModelMetadata(
    name="my_onnx_model",
    version="1",
    inputs=[triton.ModelInput("input", triton.DataType.FLOAT, [1, 3, 224, 224])],
    outputs=[triton.ModelOutput("output", triton.DataType.FLOAT, [1, 1000])],
)

# Create a model configuration object
model_config = triton.ModelConfig(
    name="my_onnx_model",
    version="1",
    platform="onnxruntime",
    max_batch_size=1,
    input=[
        triton.ModelInputConfig(
            name="input",
            data_type=triton.DataType.FLOAT,
            dims=[1, 3, 224, 224],
            preprocess=triton.InferenceServerClient.Preprocess(
                triton.InferenceServerClient.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                )
            ),
        )
    ],
    output=[triton.ModelOutputConfig(name="output", data_type=triton.DataType.FLOAT)],
)

# Push the model and configuration to Triton Server
triton_client.load_model(model_name="my_onnx_model", model_path="/path/to/my/onnx/model", model_version="1", model_metadata=model_metadata, model_config=model_config)


```