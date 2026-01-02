from fastapi import APIRouter,UploadFile,HTTPException
from PIL import Image
from io import BytesIO
from service.Core.logic.onnx_inference import malaria_detector
import numpy as np
detect_router=APIRouter()
@detect_router.post('/detect')
def detect(im:UploadFile):
     if im.filename.split('.')[-1] in ("jpg","jpeg",'png'):
         pass
     else:
         raise HTTPException(status_code=415,detail="Only JPG, JPEG, and PNG images are supported")
     
     image=Image.open(BytesIO(im.file.read()))
     image=np.array(image)
     
     return malaria_detector(image)
