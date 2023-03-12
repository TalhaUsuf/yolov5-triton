import os

def create_versioned_dir(path):
    '''
    checks if a numerically named subdir. exists inside the path given, if it exists then creates the next integer incremented sub-dir.
    else if no subdirs exist then creates a sub-dir named 1.

    Parameters
    ----------
    path : str
        model-repository which needs to be mounted inside the triton-server 💻
        
    💀Note:
    --------
    The directory tree structure is the same as the one used by triton-server for model-repository.
    
    ```markdown
        models/
            ├── yolo_ensemble
            │   ├── 1
            │   └── config.pbtxt
            ├── yolo_postprocess
            │   ├── 1
            │   │   ├── __pycache__
            │   │   │   └── model.cpython-38.pyc
            │   │   └── model.py
            │   └── config.pbtxt
            ├── yolo_preprocess
            │   ├── 1
            │   │   ├── __pycache__
            │   │   │   ├── model.cpython-38.pyc
            │   │   │   └── pre_processing_utils.cpython-38.pyc
            │   │   └── model.py
            │   └── config.pbtxt
            └── yolov5
                ├── 1
                │   └── model.onnx
                ├── 2
                │   └── model.onnx
                └── config.pbtxt
    ```
    
    '''    
    if not os.path.exists(path):
        os.makedirs(path)
        os.makedirs(os.path.join(path, '1'))
    else:
        subdirs = [int(d) for d in os.listdir(path) if os.path.isdir(os.path.join(path, d)) and d.isnumeric()]
        if len(subdirs) == 0:
            os.makedirs(os.path.join(path, '1'))
        else:
            new_dir = str(max(subdirs) + 1)
            os.makedirs(os.path.join(path, new_dir))