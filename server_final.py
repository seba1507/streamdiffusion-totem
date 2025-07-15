#!/usr/bin/env python3
"""
Servidor Final - Solución definitiva con corrección de CUDA
Para RunPod con RTX 4090
"""

import os
import sys

# CRÍTICO: Configurar CUDA antes de cualquier import de torch
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

import torch

# Verificar CUDA inmediatamente
if not torch.cuda.is_available():
    print("❌ CUDA no disponible. Ejecuta estos comandos:")
    print("export CUDA_VISIBLE_DEVICES=0")
    print("export CUDA_DEVICE_ORDER=PCI_BUS_ID")
    sys.exit(1)

print(f"✅ CUDA OK: {torch.cuda.get_device_name(0)}")

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
import base64
from PIL import Image
import io
import time
import json

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class UltraFastProcessor:
    def __init__(self):
        self.pipe = None
        self.device = torch.device("cuda:0")  # Explícito
        self.init_model()
        
    def init_model(self):
        print("🚀 Cargando modelo optimizado...")
        
        # Usar modelo que sabemos que funciona
        model_id = "runwayml/stable-diffusion-v1-5"
        
        self.pipe = DiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            safety_checker=None,
            requires_safety_checker=False
        )
        
        # Mover a GPU específica
        self.pipe = self.pipe.to(self.device)
        
        # Configurar scheduler ultra rápido
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipe.scheduler.config,
            use_karras_sigmas=True
        )
        
        # Optimizaciones
        self.pipe.set_progress_bar_config(disable=True)
        self.pipe.enable_xformers_memory_efficient_attention()
        
        # Pre-calentar
        print("🔥 Pre-calentando...")
        with torch.no_grad():
            _ = self.pipe(
                "test",
                num_inference_steps=1,
                width=512,
                height=512,
                guidance_scale=0.0
            ).images[0]
        
        print("✅ Modelo listo!")
    
    def process(self, prompt, num_steps=1):
        with torch.no_grad():
            result = self.pipe(
                prompt,
                num_inference_steps=num_steps,
                width=512,
                height=512,
                guidance_scale=0.0
            ).images[0]
        return result

processor = UltraFastProcessor()

# HTML simplificado
HTML_CONTENT = """
<!DOCTYPE html>
<html>
<head>
    <title>Ultra Fast Diffusion</title>
    <style>
        body { margin: 0; background: #111; color: white; font-family: Arial; }
        .container { display: flex; height: 100vh; align-items: center; justify-content: center; gap: 20px; }
        canvas { width: 512px; height: 512px; border: 2px solid #0f0; }
        .controls { position: fixed; bottom: 20px; left: 50%; transform: translateX(-50%); 
                   background: rgba(0,0,0,0.9); padding: 20px; border-radius: 10px; }
        button { padding: 10px 20px; margin: 0 5px; font-size: 16px; cursor: pointer; }
        .stats { position: fixed; top: 20px; right: 20px; background: rgba(0,0,0,0.9); 
                padding: 15px; border-radius: 10px; font-family: monospace; }
    </style>
</head>
<body>
    <div class="container">
        <video id="video" autoplay muted width="512" height="512"></video>
        <canvas id="output" width="512" height="512"></canvas>
    </div>
    <div class="stats">
        FPS: <span id="fps">0</span> | Latency: <span id="latency">0</span>ms
    </div>
    <div class="controls">
        <button onclick="start()">Start</button>
        <button onclick="stop()">Stop</button>
        <select id="steps">
            <option value="1">1 Step (Max Speed)</option>
            <option value="2">2 Steps</option>
            <option value="4">4 Steps</option>
        </select>
    </div>
    
    <script>
        let ws = null;
        let streaming = false;
        let frameCount = 0;
        let lastTime = Date.now();
        
        async function start() {
            const video = document.getElementById('video');
            video.srcObject = await navigator.mediaDevices.getUserMedia({video: true});
            
            ws = new WebSocket(`${location.protocol === 'https:' ? 'wss:' : 'ws:'}//${location.host}/ws`);
            ws.onopen = () => { streaming = true; sendFrame(); };
            ws.onmessage = (e) => {
                const data = JSON.parse(e.data);
                if (data.image) {
                    const img = new Image();
                    img.onload = () => {
                        document.getElementById('output').getContext('2d').drawImage(img, 0, 0);
                        updateStats(data.latency);
                    };
                    img.src = data.image;
                }
            };
        }
        
        function sendFrame() {
            if (!streaming) return;
            const video = document.getElementById('video');
            const canvas = document.createElement('canvas');
            canvas.width = canvas.height = 512;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(video, 0, 0, 512, 512);
            
            ws.send(JSON.stringify({
                image: canvas.toDataURL('image/jpeg', 0.8),
                steps: parseInt(document.getElementById('steps').value)
            }));
            
            setTimeout(sendFrame, 50); // 20 FPS max
        }
        
        function updateStats(latency) {
            frameCount++;
            const now = Date.now();
            if (now - lastTime >= 1000) {
                document.getElementById('fps').textContent = frameCount;
                frameCount = 0;
                lastTime = now;
            }
            if (latency) {
                document.getElementById('latency').textContent = Math.round(latency * 1000);
            }
        }
        
        function stop() {
            streaming = false;
            if (ws) ws.close();
            const video = document.getElementById('video');
            if (video.srcObject) {
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
    print("Cliente conectado")
    
    try:
        while True:
            data = await websocket.receive_json()
            start = time.time()
            
            # Decodificar imagen (por ahora ignoramos, generamos desde texto)
            steps = data.get('steps', 1)
            
            # Generar
            result = processor.process("a photo", num_steps=steps)
            
            # Codificar
            buffered = io.BytesIO()
            result.save(buffered, format="JPEG", quality=85)
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            # Enviar
            await websocket.send_json({
                'image': f'data:image/jpeg;base64,{img_str}',
                'latency': time.time() - start
            })
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    import uvicorn
    print("\n🎯 SERVIDOR FINAL - SOLUCIÓN DEFINITIVA")
    print("📍 http://0.0.0.0:8000\n")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="error")
