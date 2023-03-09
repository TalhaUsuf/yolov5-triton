from fastapi import FastAPI, File, UploadFile
from fastapi.responses import RedirectResponse
from typing import List
import os
from pathlib import Path
# from tritonclient.grpc import InferenceServerClient
# from tritonclient.grpc import DataType, ModelMetadata


# ==========================================================================
#                             define tags for end-points                                 
# ==========================================================================

tags_metadata = [
    {
        "name": "model",
        "description": "end points related to model upload",
    },
    {
        "name": "inference",
        "description": "end points related to the inference",
    },

]



description = """
Model checkpoint will be uploaded to the server and after conversion to onnx, the model will be served for inference using **Triton**

## Steps

 - model conversion to onnx
 - push model to model-repository be served by triton
 - send inference request to triton
 - pre-post processing steps to be done on the server

"""

app = FastAPI(
    title="Yolov5 Detection Task",
    description=description,
    version="1.0",
    terms_of_service="talhayousuf_@outlook.com",
    contact={
        "name": "Talha Yousuf",
        "email": "talhayousuf_@outlook.com",
    },
    license_info={
        "name": "Apache 2.0",
        "url": "https://www.apache.org/licenses/LICENSE-2.0.html",
    },
)


@app.get("/")
async def get_docs():
    return RedirectResponse(url="/docs")


# ==========================================================================
#                             file upload endpoint
# ==========================================================================


@app.post("/upload/", tags=['model'])
async def upload_file_and_strings(file: UploadFile = File(...), class_names: List[str] = []):

    # write the onnx file to disk
    file_name = file.filename
    with open(file_name, "wb") as f:
        f.write(await file.read())

    assert Path(file_name).exists(), "uploaded file not found"
    
    
    
    # # upload model to triton server
    # client = InferenceServerClient(url="localhost:8001")

    # # Check the connection status
    # client.is_server_live()
    
    # # model metadata for triton model registration
    # metadata = ModelMetadata(
    # name="densenet_onnx",
    # version="1",
    # inputs=[{"name": "data_0", "data_type": DataType.FLOAT32, "shape": [1, 3, 224, 224]}],
    # outputs=[{"name": "fc6_1", "data_type": DataType.FLOAT32, "shape": [1, 1000, 1, 1]}]
    # )
    
    # # Upload the model
    # client.load_model("densenet_onnx", "1", file_name, metadata=metadata)

    # Return a JSON response with the file name and list of strings
    return {"file_name": file_name, "classes": class_names}
