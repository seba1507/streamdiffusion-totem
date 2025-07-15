from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import torch
from diffusers import DiffusionPipeline, LCMScheduler
import numpy as np
from PIL import Image
import cv2
import base64
import io
import asyncio
import time
from collections import deque
import threading
import queue

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class VideoStreamProcessor:
    def __init__(self):
        self.pipe = None
        self.processing_queue = queue.Queue(maxsize=2)
        self.output_queue = queue.Queue(maxsize=5)
        self.is_running = False
        self.init_model()
        
    def init_model(self):
        print("Cargando modelo optimizado...")
        
        # Modelo Turbo para máxima velocidad
        self.pipe = DiffusionPipeline.from_pretrained(
            "stabilityai/sd-turbo",
            torch_dtype=torch.float16,
            variant="fp16",
            safety_checker=None
        ).to("cuda")
        
        # Optimizaciones SIN cpu_offload ni torch.compile
        self.pipe.enable_xformers_memory_efficient_attention()
        self.pipe.vae.enable_slicing()
        self.pipe.vae.enable_tiling()
        
        # Configurar scheduler para SD-Turbo
        self.pipe.scheduler = LCMScheduler.from_config(self.pipe.scheduler.config)
        
        # Deshabilitar torch._dynamo para evitar errores
        import torch._dynamo
        torch._dynamo.config.suppress_errors = True
        
        print("Modelo optimizado listo!")
        
    def start_processing(self):
        self.is_running = True
        threading.Thread(target=self._process_loop, daemon=True).start()
        
    def _process_loop(self):
        """Loop de procesamiento continuo en thread separado"""
        while self.is_running:
            try:
                # Obtener frame de la cola
                frame_data = self.processing_queue.get(timeout=0.1)
                
                # Procesar
                with torch.inference_mode():
                    with torch.cuda.amp.autocast():
                        result = self.pipe(
                            prompt=frame_data['prompt'],
                            image=frame_data['image'],
                            num_inference_steps=1,  # SD-Turbo solo necesita 1 paso!
                            strength=0.5,
                            guidance_scale=0.0,  # Sin guidance para velocidad
                        ).images[0]
                
                # Agregar a cola de salida
                self.output_queue.put({
                    'image': result,
                    'timestamp': frame_data.get('timestamp', time.time())
                })
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error en procesamiento: {e}")
    
    def add_frame(self, image, prompt, timestamp=None):
        """Agregar frame para procesar (no bloqueante)"""
        try:
            self.processing_queue.put_nowait({
                'image': image,
                'prompt': prompt,
                'timestamp': timestamp or time.time()
            })
        except queue.Full:
            # Si la cola está llena, descartar frames antiguos
            try:
                self.processing_queue.get_nowait()
                self.processing_queue.put_nowait({
                    'image': image,
                    'prompt': prompt,
                    'timestamp': timestamp or time.time()
                })
            except:
                pass
    
    def get_latest_frame(self):
        """Obtener el frame más reciente procesado"""
        result = None
        # Vaciar cola y quedarse con el último
        while not self.output_queue.empty():
            try:
                result = self.output_queue.get_nowait()
            except:
                break
        return result

# Instancia global del procesador
processor = VideoStreamProcessor()
processor.start_processing()

@app.get("/")
async def get():
    return HTMLResponse(content=open("index_fast.html", "r").read())

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Cliente conectado - Modo streaming")
    
    # Variables para FPS
    last_frame_time = time.time()
    frame_count = 0
    fps_list = deque(maxlen=30)  # Para promedio móvil
    
    try:
        while True:
            # Recibir frame
            data = await websocket.receive_json()
            
            if data['type'] == 'frame':
                # Decodificar imagen
                img_data = base64.b64decode(data['image'].split(',')[1])
                image = Image.open(io.BytesIO(img_data)).convert("RGB")
                
                # Redimensionar para velocidad
                image = image.resize((512, 512), Image.Resampling.LANCZOS)
                
                # Construir prompt completo
                style = data.get('prompt', 'cyberpunk')
                full_prompt = f"{style} style, high quality, detailed, sharp focus"
                
                # Agregar a procesamiento con timestamp
                timestamp = data.get('timestamp', time.time())
                processor.add_frame(image, full_prompt, timestamp)
                
                # Obtener último frame procesado (si hay)
                processed_data = processor.get_latest_frame()
                
                if processed_data:
                    processed = processed_data['image']
                    original_timestamp = processed_data['timestamp']
                    
                    # Codificar y enviar
                    buffered = io.BytesIO()
                    processed.save(buffered, format="JPEG", quality=85)
                    img_str = base64.b64encode(buffered.getvalue()).decode()
                    
                    await websocket.send_json({
                        'type': 'frame',
                        'image': f'data:image/jpeg;base64,{img_str}',
                        'timestamp': original_timestamp
                    })
                    
                    # Calcular FPS
                    frame_count += 1
                    current_time = time.time()
                    elapsed = current_time - last_frame_time
                    
                    if elapsed >= 0.1:  # Actualizar cada 100ms
                        fps = frame_count / elapsed
                        fps_list.append(fps)
                        avg_fps = sum(fps_list) / len(fps_list)
                        print(f"FPS: {fps:.1f} (avg: {avg_fps:.1f})")
                        frame_count = 0
                        last_frame_time = current_time
                else:
                    # Si no hay frame procesado, reenviar el original
                    # para mantener fluidez visual
                    await websocket.send_json({
                        'type': 'frame',
                        'image': data['image'],
                        'timestamp': data.get('timestamp', time.time())
                    })
            
    except Exception as e:
        print(f"Error en websocket: {e}")
    finally:
        print("Cliente desconectado")

if __name__ == "__main__":
    import uvicorn
    # Configuración optimizada para producción
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info",
        access_log=False,  # Desactivar logs de acceso para mejor rendimiento
        loop="uvloop"  # Loop más rápido si está disponible
    )
