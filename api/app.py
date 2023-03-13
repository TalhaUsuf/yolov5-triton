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
from shutil import copy2
import tritonclient.http as httpclient
import torch
from PIL import Image
import shutil
import numpy as np
# from tritonclient.grpc import InferenceServerClient
# from tritonclient.grpc import DataType, ModelMetadata




try:
    triton_client = httpclient.InferenceServerClient(
        url='localhost:8000', verbose=True)
except Exception as e:
    raise HTTPException(status_code=500, detail="triton server is not live")

    

# ==========================================================================
#                             helper fuctions                                  
# ==========================================================================
def load_image(img_path: str):
    """
    Loads an encoded image as an array of bytes.
    
    """
    return np.fromfile(img_path, dtype='uint8')

def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords
def clip_coords(boxes, shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[:, 0].clamp_(0, shape[1])  # x1
        boxes[:, 1].clamp_(0, shape[0])  # y1
        boxes[:, 2].clamp_(0, shape[1])  # x2
        boxes[:, 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2





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

classes = _CLASSES


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
    
    global classes
    classes = class_names
    
    #🔵 write the uploaded file to the temp. directory
    file_name = file.filename
    with open("tmp/"+file_name, "wb") as f:
        f.write(await file.read())
        

    #🔵 if model not saved to correct path then raise exception
    if not Path("tmp/"+file_name).exists():
        raise HTTPException(status_code=409, detail="uploaded file not found at expected location")
    
    
    #🔵 wait for convert to onnx
    Console().log(f"{Path('tmp/'+file_name).as_posix()} conversion to onnx started", style='red')
    # result = subprocess.run(['python3', 'converter/yolov5/export.py' , f'--weights {Path("tmp/"+file_name).as_posix()}', '--dynamic',  '--include onnx'],  stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    os.system(f"python3 ./converter/yolov5/export.py --weights {Path('tmp/'+file_name).as_posix()}  --dynamic  --include onnx")
    _converted_file = Path('tmp/'+file_name).with_suffix('.onnx')
    status = _converted_file.exists()
    Console().log(f"💀\t onnx conversion status is : {status}")
    if not status:
        raise HTTPException(status_code=404, detail="conversion to onnx failed, check your uploaded file")
    
    #🔵 rename the currently converted onnx file to model.onnx
    Path(_converted_file).rename(Path("tmp/model.onnx"))
    if not Path("tmp/model.onnx").exists():
        raise HTTPException(status_code=500, detail="model.onnx file not found in the tmp directory")
    
    
    #🔵 check existing versions of yolov5 and create new versioned directory
    # in which model will be saved
    versioned_dir = create_versioned_dir("models/yolov5")
    Console().log(f"[green]versioned directory created : [red]{versioned_dir}")
    if not Path(versioned_dir).exists():
        raise HTTPException(status_code=500, detail="versioned directory not found")
    
    #🔵 copy new onnx model to versioned_dir
    copy2(Path("tmp/model.onnx").as_posix(), Path(versioned_dir))
    if not Path(versioned_dir+"/model.onnx").exists():
        raise HTTPException(status_code=500, detail="model.onnx file not pushed in the versioned directory")
    
    
    # # make triton client
    # tritonclient = httpclient.InferenceServerClient(url="http://localhost:8000", verbose=True)
    # # check if server is live
    # _status = tritonclient.is_server_live()
    # if not _status:
    #     raise HTTPException(status_code=500, detail="triton server is not live")
    # Console().log(f"Triton server status: {_status}", style="green")

    return {
        "version" : versioned_dir.split("/")[-1]
    }
    

# ==========================================================================
#                             inference endpoint                                  
# ==========================================================================



@app.post("/infer/", tags=['inferece'])
def upload_image(image: UploadFile = File(...)):


    
    
    
    
    global classes
    Console().log(f"classes : {classes}")
    
    
    file_name = "image.jpg"
    # if the tmp image already exists, then delete it
    if Path(file_name).exists():
        Path(file_name).unlink()
        
    with open(file_name, "wb") as f:
        shutil.copyfileobj(image.file, f)
        
    # raise exception if image not saved to correct path
    if not Path(file_name).exists():
        raise HTTPException(status_code=409, detail="uploaded file not found at expected location")
    
    
    
    # ⚡ actual inference (ensemble)
    inputs = []
    outputs = []
    input_name = "INPUT_ENSEMBLE"
    
    output_name = "OUTPUT_ENSEMBLE"
    
    # get H,W needed for re-scaling of coordinates
    sz = Image.open(file_name).size
    # read image as bytes
    image_data = load_image(file_name) # shape (1005970,)
    # convert to batch
    image_data = np.expand_dims(image_data, axis=0)  # shape (1, 1005970)

    inputs.append(httpclient.InferInput(input_name, image_data.shape, "UINT8"))
    outputs.append(httpclient.InferRequestedOutput(output_name))

    

    
    inputs[0].set_data_from_numpy(image_data)
    
    results = triton_client.infer(model_name="yolo_ensemble",
                                inputs=inputs,
                                outputs=outputs)

    
    output0_data = results.as_numpy(output_name)
    Console().log(f"⚡\t[red]after inference")
    # import joblib
    
    # joblib.dump(output0_data, "ensemble_out.pkl")
    # print(output0_data.shape)
    
    
    # # --------------------------------------------------------------------------
    # #                        map detections to labels                        
    # # --------------------------------------------------------------------------
    
    for i, det in enumerate(output0_data):
        # im0 --> resized image
        # im --> original image
        det[:, :4] = scale_coords([640, 640], det[:, :4], sz).round()
    # print(det)
    
    idx2classes = {k:v for k,v in enumerate(classes)}
    # #annotate the image
    _classes = []
    _conf = []
    _coordinates = []
    for *xyxy, conf, _cls in reversed(det):
        c = int(_cls)  # integer class
        # label = f'{names[c]} {conf:.2f}'
        print(f"detected : {c} with conf. {conf}, coordinates : {xyxy}")
        _classes.append(c)
        _conf.append(conf)
        _coordinates.append(xyxy)
        # annotator.box_label(xyxy, label, color=colors(c, True))
    return {
        "classes" : _classes,
        "conf" : _conf,
        "coordinates" : _coordinates
    }

    
    
    