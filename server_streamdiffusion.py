#!/usr/bin/env python3
"""
StreamDiffusion SDXL Server - Optimizado para máximo FPS
Diseñado para RTX 4090 en RunPod
"""

import os
import sys
import torch
import numpy as np
from PIL import Image
import base64
import io
import time
import asyncio
import json
from typing import Optional

# Configuración crítica de CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Verificar CUDA antes de continuar
if not torch.cuda.is_available():
    print("❌ ERROR: CUDA no está disponible")
    print("Ejecuta: export CUDA_VISIBLE_DEVICES=0")
    sys.exit(1)

print(f"✅ CUDA disponible: {torch.cuda.get_device_name(0)}")
print(f"✅ Memoria GPU: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")

# Importaciones de StreamDiffusion
try:
    from diffusers import StableDiffusionXLPipeline, AutoencoderTiny
    from streamdiffusion import StreamDiffusion
    from streamdiffusion.acceleration.tensorrt import accelerate_with_tensorrt
    from streamdiffusion.acceleration.tensorrt.engine import AutoencoderKLEngine
    from streamdiffusion.acceleration.tensorrt.models import VAE, UNet, CLIPEncoder
    from streamdiffusion.image_utils import postprocess_image
except ImportError as e:
    print(f"❌ Error importando StreamDiffusion: {e}")
    print("Instala con: pip install git+https://github.com/cumulo-autumn/StreamDiffusion.git@main#egg=streamdiffusion[tensorrt]")
    sys.exit(1)

# FastAPI imports
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import uvicorn

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class StreamDiffusionEngine:
    def __init__(self):
        self.stream = None
        self.device = torch.device("cuda:0")
        self.dtype = torch.float16
        self.init_model()
        
    def init_model(self):
        """Inicializar StreamDiffusion con configuración óptima"""
        print("🚀 Inicializando StreamDiffusion SDXL...")
        
        try:
            # Cargar modelo base SDXL-Turbo
            model_id = "stabilityai/sdxl-turbo"
            print(f"📦 Cargando modelo: {model_id}")
            
            # Pipeline con configuración optimizada
            pipe = StableDiffusionXLPipeline.from_pretrained(
                model_id,
                torch_dtype=self.dtype,
                use_safetensors=True,
                variant="fp16"
            ).to(self.device)
            
            # Usar TinyVAE para máxima velocidad
            print("⚡ Cargando TinyVAE...")
            pipe.vae = AutoencoderTiny.from_pretrained(
                "madebyollin/taesdxl",
                torch_dtype=self.dtype
            ).to(self.device)
            
            # Habilitar optimizaciones
            pipe.enable_xformers_memory_efficient_attention()
            
            # Crear StreamDiffusion con configuración de 1 paso
            print("🔧 Configurando StreamDiffusion...")
            self.stream = StreamDiffusion(
                pipe,
                t_index_list=[32],  # Un solo timestep para máxima velocidad
                torch_dtype=self.dtype,
                cfg_type="none",
                width=512,
                height=512,
            )
            
            # Pre-calentar el modelo
            print("🔥 Pre-calentando modelo...")
            self.warmup()
            
            print("✅ StreamDiffusion SDXL listo!")
            print("🎯 Configurado para máximo FPS con 1 paso de inferencia")
            
        except Exception as e:
            print(f"❌ Error inicializando modelo: {e}")
            raise
    
    def warmup(self):
        """Pre-calentar el modelo para evitar latencia inicial"""
        prompt = "a photo"
        for _ in range(3):
            self.stream.prepare(
                prompt=prompt,
                num_inference_steps=1,
            )
            
    def process_image(self, prompt: str = "a photo, high quality") -> Optional[Image.Image]:
        """Procesar imagen con StreamDiffusion"""
        try:
            # Preparar generación
            self.stream.prepare(
                prompt=prompt,
                num_inference_steps=1,
            )
            
            # Generar imagen
            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=self.dtype):
                    # StreamDiffusion usa un proceso diferente al estándar
                    x_output = self.stream()
                    
                    # Post-procesar resultado
                    if x_output is not None:
                        image = postprocess_image(x_output, output_type="pil")[0]
                        return image
                        
        except Exception as e:
            print(f"Error procesando: {e}")
            return None

# Instancia global
engine = None

@app.on_event("startup")
async def startup_event():
    global engine
    engine = StreamDiffusionEngine()

# HTML simple para testing
HTML_CONTENT = """
<!DOCTYPE html>
<html>
<head>
    <title>StreamDiffusion SDXL - Max FPS</title>
    <style>
        body { margin: 0; background: #000; color: #fff; font-family: Arial; }
        #container { display: flex; height: 100vh; align-items: center; justify-content: center; }
        canvas { border: 2px solid #0f0; margin: 10px; }
        #controls { position: fixed; bottom: 20px; left: 50%; transform: translateX(-50%); 
                   background: rgba(0,0,0,0.8); padding: 20px; border-radius: 10px; }
        button { padding: 10px 20px; margin: 0 10px; cursor: pointer; }
        #stats { position: fixed; top: 20px; right: 20px; background: rgba(0,0,0,0.8); 
                padding: 15px; border-radius: 10px; font-family: monospace; }
    </style>
</head>
<body>
    <div id="container">
        <canvas id="input" width="512" height="512"></canvas>
        <canvas id="output" width="512" height="512"></canvas>
    </div>
    <div id="stats">
        <div>FPS: <span id="fps">0</span></div>
        <div>Latency: <span id="latency">0</span>ms</div>
    </div>
    <div id="controls">
        <button onclick="start()">Start</button>
        <button onclick="stop()">Stop</button>
        <select id="style">
            <option value="">Default</option>
            <option value="cyberpunk style">Cyberpunk</option>
            <option value="anime style">Anime</option>
        </select>
    </div>
    
    <script>
        let ws = null;
        let streaming = false;
        let video = null;
        const inputCanvas = document.getElementById('input');
        const outputCanvas = document.getElementById('output');
        const inputCtx = inputCanvas.getContext('2d');
        const outputCtx = outputCanvas.getContext('2d');
        
        let frameCount = 0;
        let lastTime = Date.now();
        
        async function start() {
            video = document.createElement('video');
            video.srcObject = await navigator.mediaDevices.getUserMedia({video: true});
            video.play();
            
            ws = new WebSocket(`${location.protocol === 'https:' ? 'wss:' : 'ws:'}//${location.host}/ws`);
            ws.onopen = () => { 
                streaming = true; 
                sendFrame(); 
            };
            
            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                if (data.image) {
                    const img = new Image();
                    img.onload = () => {
                        outputCtx.drawImage(img, 0, 0);
                        updateStats();
                    };
                    img.src = data.image;
                }
            };
        }
        
        function sendFrame() {
            if (!streaming) return;
            
            inputCtx.drawImage(video, 0, 0, 512, 512);
            const imageData = inputCanvas.toDataURL('image/jpeg', 0.8);
            
            ws.send(JSON.stringify({
                image: imageData,
                style: document.getElementById('style').value,
                timestamp: Date.now()
            }));
            
            // StreamDiffusion puede manejar alto FPS
            setTimeout(sendFrame, 33); // ~30 FPS
        }
        
        function updateStats() {
            frameCount++;
            const now = Date.now();
            if (now - lastTime >= 1000) {
                document.getElementById('fps').textContent = frameCount;
                frameCount = 0;
                lastTime = now;
            }
        }
        
        function stop() {
            streaming = false;
            if (ws) ws.close();
            if (video && video.srcObject) {
                video.srcObject.getTracks().forEach(t => t.stop());
            }
        }
    </script>
</body>
</html>
"""

@app.get("/")
async def get():
    return HTMLResponse(content=HTML_CONTENT)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("🎮 Cliente conectado")
    
    frame_times = []
    
    try:
        while True:
            data = await websocket.receive_json()
            
            start_time = time.time()
            
            # Procesar con StreamDiffusion
            style = data.get('style', '')
            prompt = f"{style} photo, high quality" if style else "photo, high quality"
            
            result = engine.process_image(prompt)
            
            if result:
                # Codificar resultado
                buffered = io.BytesIO()
                result.save(buffered, format="JPEG", quality=85)
                img_str = base64.b64encode(buffered.getvalue()).decode()
                
                # Enviar respuesta
                await websocket.send_json({
                    'image': f'data:image/jpeg;base64,{img_str}',
                    'timestamp': data.get('timestamp')
                })
                
                # Calcular FPS
                process_time = (time.time() - start_time) * 1000
                frame_times.append(process_time)
                
                if len(frame_times) > 30:
                    frame_times.pop(0)
                    avg_time = sum(frame_times) / len(frame_times)
                    fps = 1000 / avg_time if avg_time > 0 else 0
                    print(f"📊 FPS: {fps:.1f} | Latencia: {avg_time:.1f}ms")
                    
    except WebSocketDisconnect:
        print("👋 Cliente desconectado")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 STREAMDIFFUSION SDXL SERVER")
    print("="*60)
    print("⚡ Optimizado para RTX 4090")
    print("🎯 Target: 100+ FPS con 1 paso de inferencia")
    print("📍 URL: http://0.0.0.0:8000")
    print("="*60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
