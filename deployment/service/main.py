from fastapi import FastAPI
from pydantic import BaseModel
from service.api.api import main_router
import onnxruntime as rt
app=FastAPI(project_name="Malaria Detection")
app.include_router(main_router)

# --- Load ONNX model ---
    
providers = ['CPUExecutionProvider']
m_q = rt.InferenceSession('quantized_model2.onnx', providers=providers)


