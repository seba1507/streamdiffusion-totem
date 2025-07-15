#!/usr/bin/env python3
import os
import sys

# Configurar CUDA antes de importar torch
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

import torch

# Verificar CUDA
if not torch.cuda.is_available():
    print("ERROR: CUDA no disponible")
    sys.exit(1)

print(f"CUDA OK: {torch.cuda.get_device_name(0)}")

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
import base64
from PIL import Image
import io
import time
import json
import asyncio

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class FastDiffusion:
    def __init__(self):
        self.pipe = None
        self.device = torch.device("cuda:0")
        self.dtype = torch.float16
        self.init_model()
        
    def init_model(self):
        print("Cargando modelo...")
        
        # Modelo estable y rápido
        model_id = "runwayml/stable-diffusion-v1-5"
        
        self.pipe = DiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=self.dtype,
            safety_checker=None,
            requires_safety_checker=False,
            local_files_only=False
        )
        
        self.pipe = self.pipe.to(self.device)
        
        # Scheduler optimizado
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipe.scheduler.config,
            use_karras_sigmas=True,
            algorithm_type="dpmsolver++",
            solver_type="midpoint",
            lower_order_final=True
        )
        
        # Optimizaciones
        self.pipe.set_progress_bar_config(disable=True)
        
        # XFormers si está disponible
        try:
            self.pipe.enable_xformers_memory_efficient_attention()
            print("XFormers habilitado")
        except:
            print("XFormers no disponible, usando atención estándar")
        
        # Pre-calentar
        print("Pre-calentando modelo...")
        with torch.no_grad():
            _ = self.pipe(
                "test",
                num_inference_steps=1,
                width=512,
                height=512,
                guidance_scale=0.0
            ).images[0]
        
        print("Modelo listo!")
    
    def generate(self, prompt, steps=1):
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                result = self.pipe(
                    prompt,
                    num_inference_steps=steps,
                    width=512,
                    height=512,
                    guidance_scale=0.0
                ).images[0]
        return result

# Instancia global
processor = FastDiffusion()

# HTML para interfaz
HTML_CONTENT = """
<!DOCTYPE html>
<html>
<head>
    <title>Fast Diffusion Stream</title>
    <style>
        body {
            margin: 0;
            background: #000;
            color: #fff;
            font-family: Arial, sans-serif;
            overflow: hidden;
        }
        .container {
            display: flex;
            height: 100vh;
            align-items: center;
            justify-content: center;
            gap: 20px;
        }
        video, canvas {
            width: 512px;
            height: 512px;
            border: 2px solid #00ff00;
            background: #111;
        }
        .controls {
            position: fixed;
            bottom: 20px;
            left: 50%;
            transform: translateX(-50%);
            background: rgba(0,0,0,0.9);
            padding: 20px;
            border-radius: 10px;
            display: flex;
            gap: 10px;
            align-items: center;
        }
        button {
            padding: 10px 20px;
            font-size: 16px;
            cursor: pointer;
            background: #0066cc;
            color: white;
            border: none;
            border-radius: 5px;
        }
        button:hover {
            background: #0052a3;
        }
        button:disabled {
            background: #666;
            cursor: not-allowed;
        }
        select {
            padding: 10px;
            font-size: 16px;
            border-radius: 5px;
        }
        .stats {
            position: fixed;
            top: 20px;
            right: 20px;
            background: rgba(0,0,0,0.9);
            padding: 15px;
            border-radius: 10px;
            font-family: monospace;
            font-size: 14px;
        }
        .stat-value {
            color: #00ff00;
            font-weight: bold;
        }
    </style>
</head>
<body>
    <div class="container">
        <video id="video" autoplay muted playsinline></video>
        <canvas id="output"></canvas>
    </div>
    
    <div class="stats">
        <div>FPS: <span id="fps" class="stat-value">0</span></div>
        <div>Latencia: <span id="latency" class="stat-value">0</span>ms</div>
        <div>Frames: <span id="frames" class="stat-value">0</span></div>
    </div>
    
    <div class="controls">
        <button id="startBtn" onclick="start()">Iniciar</button>
        <button id="stopBtn" onclick="stop()" disabled>Detener</button>
        <select id="style">
            <option value="">Fotorealista</option>
            <option value="cyberpunk">Cyberpunk</option>
            <option value="anime">Anime</option>
            <option value="oil painting">Óleo</option>
            <option value="watercolor">Acuarela</option>
        </select>
        <select id="steps">
            <option value="1">1 paso (Máx velocidad)</option>
            <option value="2">2 pasos</option>
            <option value="4">4 pasos</option>
        </select>
    </div>
    
    <script>
        let ws = null;
        let streaming = false;
        let video = null;
        let canvas = null;
        let ctx = null;
        
        // Estadísticas
        let frameCount = 0;
        let totalFrames = 0;
        let lastTime = Date.now();
        let latencies = [];
        
        async function start() {
            try {
                // Configurar video
                video = document.getElementById('video');
                canvas = document.getElementById('output');
                ctx = canvas.getContext('2d');
                canvas.width = 512;
                canvas.height = 512;
                
                // Obtener cámara
                const stream = await navigator.mediaDevices.getUserMedia({
                    video: {
                        width: { ideal: 1280 },
                        height: { ideal: 720 }
                    }
                });
                video.srcObject = stream;
                
                // Conectar WebSocket
                const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                ws = new WebSocket(`${protocol}//${window.location.host}/ws`);
                
                ws.onopen = () => {
                    console.log('Conectado');
                    streaming = true;
                    document.getElementById('startBtn').disabled = true;
                    document.getElementById('stopBtn').disabled = false;
                    sendFrame();
                };
                
                ws.onmessage = (event) => {
                    const data = JSON.parse(event.data);
                    if (data.image) {
                        const img = new Image();
                        img.onload = () => {
                            ctx.drawImage(img, 0, 0, 512, 512);
                            updateStats(data.processingTime);
                        };
                        img.src = data.image;
                    }
                };
                
                ws.onerror = (error) => {
                    console.error('Error WebSocket:', error);
                    stop();
                };
                
                ws.onclose = () => {
                    console.log('Desconectado');
                    stop();
                };
                
            } catch (error) {
                console.error('Error:', error);
                alert('Error: ' + error.message);
            }
        }
        
        function sendFrame() {
            if (!streaming || !ws || ws.readyState !== WebSocket.OPEN) return;
            
            // Capturar frame del video
            const tempCanvas = document.createElement('canvas');
            tempCanvas.width = 512;
            tempCanvas.height = 512;
            const tempCtx = tempCanvas.getContext('2d');
            
            // Dibujar video centrado y recortado
            const videoAspect = video.videoWidth / video.videoHeight;
            let sx, sy, sw, sh;
            
            if (videoAspect > 1) {
                sw = video.videoHeight;
                sh = video.videoHeight;
                sx = (video.videoWidth - sw) / 2;
                sy = 0;
            } else {
                sw = video.videoWidth;
                sh = video.videoWidth;
                sx = 0;
                sy = (video.videoHeight - sh) / 2;
            }
            
            tempCtx.drawImage(video, sx, sy, sw, sh, 0, 0, 512, 512);
            
            // Enviar datos
            const imageData = tempCanvas.toDataURL('image/jpeg', 0.8);
            ws.send(JSON.stringify({
                image: imageData,
                style: document.getElementById('style').value,
                steps: parseInt(document.getElementById('steps').value),
                timestamp: Date.now()
            }));
            
            // Programar siguiente frame (20 FPS máximo)
            setTimeout(sendFrame, 50);
        }
        
        function updateStats(processingTime) {
            frameCount++;
            totalFrames++;
            
            if (processingTime) {
                latencies.push(processingTime);
                if (latencies.length > 30) latencies.shift();
            }
            
            const now = Date.now();
            if (now - lastTime >= 1000) {
                // Actualizar FPS
                document.getElementById('fps').textContent = frameCount;
                frameCount = 0;
                lastTime = now;
                
                // Actualizar latencia promedio
                if (latencies.length > 0) {
                    const avgLatency = latencies.reduce((a, b) => a + b) / latencies.length;
                    document.getElementById('latency').textContent = Math.round(avgLatency);
                }
            }
            
            document.getElementById('frames').textContent = totalFrames;
        }
        
        function stop() {
            streaming = false;
            
            if (ws) {
                ws.close();
                ws = null;
            }
            
            if (video && video.srcObject) {
                video.srcObject.getTracks().forEach(track => track.stop());
                video.srcObject = null;
            }
            
            document.getElementById('startBtn').disabled = false;
            document.getElementById('stopBtn').disabled = true;
        }
        
        // Limpiar al cerrar la página
        window.addEventListener('beforeunload', stop);
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
            # Recibir datos del cliente
            data = await websocket.receive_json()
            
            # Medir tiempo de procesamiento
            start_time = time.time()
            
            # Preparar prompt
            style = data.get('style', '')
            if style:
                prompt = f"{style} style, high quality, detailed"
            else:
                prompt = "a photo, high quality, detailed"
            
            # Obtener número de pasos
            steps = data.get('steps', 1)
            
            # Generar imagen
            try:
                result = processor.generate(prompt, steps=steps)
                
                # Codificar resultado
                buffered = io.BytesIO()
                result.save(buffered, format="JPEG", quality=85)
                img_str = base64.b64encode(buffered.getvalue()).decode()
                
                # Calcular tiempo de procesamiento
                processing_time = (time.time() - start_time) * 1000
                
                # Enviar respuesta
                await websocket.send_json({
                    'image': f'data:image/jpeg;base64,{img_str}',
                    'processingTime': processing_time,
                    'timestamp': data.get('timestamp')
                })
                
                print(f"Frame procesado en {processing_time:.1f}ms")
                
            except Exception as e:
                print(f"Error generando imagen: {e}")
            
    except Exception as e:
        print(f"Error en WebSocket: {e}")
    finally:
        print("Cliente desconectado")

if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*60)
    print("FAST DIFFUSION STREAM SERVER")
    print("="*60)
    print("GPU detectada correctamente")
    print("Optimizado para máxima velocidad")
    print("URL: http://0.0.0.0:8000")
    print("="*60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
