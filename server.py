from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import torch
from diffusers import AutoPipelineForImage2Image, LCMScheduler
import base64
from PIL import Image
import io
import json
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Variable global para el pipeline
pipe = None

def load_model():
    """Cargar el modelo de difusión"""
    global pipe
    logger.info("Cargando modelo LCM...")
    
    pipe = AutoPipelineForImage2Image.from_pretrained(
        "SimianLuo/LCM_Dreamshaper_v7",
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False
    ).to("cuda")
    
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    pipe.enable_xformers_memory_efficient_attention()
    
    logger.info("¡Modelo cargado correctamente!")

# Cargar modelo al iniciar
load_model()

@app.get("/")
async def get():
    return HTMLResponse(content=open("index.html", "r").read())

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("Cliente conectado")
    
    while True:
        try:
            data = await websocket.receive_json()
            
            # Decodificar imagen
            img_data = base64.b64decode(data['image'].split(',')[1])
            image = Image.open(io.BytesIO(img_data)).convert("RGB")
            
            # Generar
            prompt = f"{data['style']}, high quality, detailed"
            
            with torch.no_grad():
                result = pipe(
                    prompt=prompt,
                    image=image,
                    num_inference_steps=4,
                    guidance_scale=1.0,
                    strength=0.5
                ).images[0]
            
            # Enviar resultado
            buffered = io.BytesIO()
            result.save(buffered, format="JPEG", quality=90)
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            await websocket.send_json({
                'image': f'data:image/jpeg;base64,{img_str}'
            })
            
        except Exception as e:
            logger.error(f"Error: {e}")
            break
    
    logger.info("Cliente desconectado")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")