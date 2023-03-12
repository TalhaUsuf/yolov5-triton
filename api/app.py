from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import RedirectResponse
from typing import List
import os
from pathlib import Path
import yaml
from typing import Optional, Union, List, Dict
from pydantic import BaseModel
from rich.console import Console
from .utils import create_versioned_dir
import subprocess
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


# get default classes
_CLASSES = list(yaml.load(open("classes.yml", 'r'), Loader=yaml.SafeLoader)['names'].values())




@app.get("/")
async def get_docs():
    return RedirectResponse(url="/docs")


# ==========================================================================
#                             file upload endpoint
# ==========================================================================



# class UploadModel(BaseModel):
#     file: UploadFile = File(...)
#     class_names: Union[List[str], None] = _CLASSES



@app.post("/upload/", tags=['model'])
async def upload_file_and_strings(file: UploadFile = File(...), class_names: Union[List[str], None] = _CLASSES):

    #🔵 check the temp. model directory, if model already exists, then delete it
    _model_files = [k for k in Path("tmp").iterdir() if k.is_file()]
    Console().log(f"{len(_model_files)} model files found in tmp directory, deleting them")
    for _file in _model_files:
        Path(_file).unlink()
    
    
    #🔵 write the uploaded file to the temp. directory
    file_name = file.filename
    with open("tmp/"+file_name, "wb") as f:
        f.write(await file.read())

    #🔵 if model not saved to correct path then raise exception
    if not Path("tmp/"+file_name).exists():
        raise HTTPException(status_code=409, detail="uploaded file not found at expected location")
    
    
    #🔵 convert to onnx

    result = subprocess.run(['python3', 'converter/yolov5/export.py' , f'--weights {Path("tmp/"+file_name).as_posix()}', '--dynamic',  '--include onnx'],  stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    if result.returncode != 0:
        raise HTTPException(status_code=404, detail="conversion to onnx failed, check your uploaded file")
    
    # rename the onnx file
    _onnx_file = Path("tmp/"+file_name).parents[0] / Path("tmp/"+file_name).stem + ".onnx"
    if not Path(_onnx_file).exists():
        raise HTTPException(status_code=500, detail="onnx file cannot be copied to the model repository")
    
    
    
    
    
    # #🔵 version this model
    # create_versioned_dir()
    
    
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
    
