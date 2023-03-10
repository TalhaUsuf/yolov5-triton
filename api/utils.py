import os

def create_versioned_dir(path)->str:
    '''
    checks if a numerically named subdir. exists inside the path given, if it exists then creates the next integer incremented sub-dir.
    else if no subdirs exist then creates a sub-dir named 1.

    Parameters
    ----------
    path : str
        model-repository which needs to be mounted inside the triton-server π»
        
    πNote:
    --------
    The directory tree structure is the same as the one used by triton-server for model-repository.
    
    ```markdown
        models/
            βββ yolo_ensemble
            β   βββ 1
            β   βββ config.pbtxt
            βββ yolo_postprocess
            β   βββ 1
            β   β   βββ __pycache__
            β   β   β   βββ model.cpython-38.pyc
            β   β   βββ model.py
            β   βββ config.pbtxt
            βββ yolo_preprocess
            β   βββ 1
            β   β   βββ __pycache__
            β   β   β   βββ model.cpython-38.pyc
            β   β   β   βββ pre_processing_utils.cpython-38.pyc
            β   β   βββ model.py
            β   βββ config.pbtxt
            βββ yolov5
                βββ 1
                β   βββ model.onnx
                βββ 2
                β   βββ model.onnx
                βββ config.pbtxt
    ```
    Return
    -------
    str
        directory path of the newly created versioned directory
    ''' 
    return_path = None   
    if not os.path.exists(path):
        os.makedirs(path)
        os.makedirs(os.path.join(path, '1'))
        return_path = os.path.join(path, '1')
    else:
        subdirs = [int(d) for d in os.listdir(path) if os.path.isdir(os.path.join(path, d)) and d.isnumeric()]
        if len(subdirs) == 0:
            # if there is no numeric sub-dir then create a sub-dir named 1
            os.makedirs(os.path.join(path, '1'))
            return_path = os.path.join(path, '1')
        else:
            new_dir = str(max(subdirs) + 1)
            os.makedirs(os.path.join(path, new_dir))
            return_path = os.path.join(path, new_dir)
    
    return return_path