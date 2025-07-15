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
from diffusers import AutoPipelineForImage2Image, LCMScheduler
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
        
        # Usar LCM para máxima velocidad con calidad
        model_id = "SimianLuo/LCM_Dreamshaper_v7"
        
        self.pipe = AutoPipelineForImage2Image.from_pretrained(
            model_id,
            torch_dtype=self.dtype,
            safety_checker=None,
            requires_safety_checker=False
        )
        
        self.pipe = self.pipe.to(self.device)
        
        # Scheduler optimizado para LCM
        self.pipe.scheduler = LCMScheduler.from_config(self.pipe.scheduler.config)
        
        # Optimizaciones
        self.pipe.set_progress_bar_config(disable=True)
        
        # XFormers si está disponible
        try:
            self.pipe.enable_xformers_memory_efficient_attention()
            print("XFormers habilitado")
        except:
            print("XFormers no disponible, usando atención estándar")
        
        # Pre-calentar con imagen dummy
        print("Pre-calentando modelo...")
        dummy_image = Image.new('RGB', (512, 512), color='red')
        with torch.no_grad():
            _ = self.pipe(
                "test",
                image=dummy_image,
                num_inference_steps=4,
                guidance_scale=1.0,
                strength=0.5
            ).images[0]
        
        print("Modelo listo!")
    
    def generate(self, prompt, input_image, steps=4):
        """Generar imagen usando image-to-image"""
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                result = self.pipe(
                    prompt=prompt,
                    image=input_image,
                    num_inference_steps=steps,
                    guidance_scale=1.0,  # Habilitar guidance para coherencia
                    strength=0.5  # Balance entre preservar y transformar
                ).images[0]
        return result

# Instancia global
processor = FastDiffusion()

# HTML para interfaz (mismo contenido, solo agregando logs para debug)
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
        .debug {
            position: fixed;
            top: 20px;
            left: 20px;
            background: rgba(0,0,0,0.9);
            padding: 15px;
            border-radius: 10px;
            font-family: monospace;
            font-size: 12px;
            max-width: 300px;
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
    
    <div class="debug">
        <div>Modo: Image-to-Image</div>
        <div>Prompt: <span id="currentPrompt">ninguno</span></div>
        <div>Pasos: <span id="currentSteps">4</span></div>
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
            <option value="4" selected>4 pasos (Recomendado)</option>
            <option value="2">2 pasos (Rápido)</option>
            <option value="6">6 pasos (Calidad)</option>
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
                
                // Esperar a que el video esté listo
                await new Promise(resolve => {
                    video.onloadedmetadata = resolve;
                });
                
                // Conectar WebSocket
                const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                ws = new WebSocket(`${protocol}//${window.location.host}/ws`);
                
                ws.onopen = () => {
                    console.log('Conectado al servidor');
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
                            ctx.clearRect(0, 0, 512, 512);
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
                    console.log('Desconectado del servidor');
                    stop();
                };
                
            } catch (error) {
                console.error('Error:', error);
                alert('Error: ' + error.message);
            }
        }
        
        function sendFrame() {
            if (!streaming || !ws || ws.readyState !== WebSocket.OPEN) return;
            
            // Verificar que el video esté reproduciéndose
            if (video.videoWidth === 0 || video.videoHeight === 0) {
                setTimeout(sendFrame, 100);
                return;
            }
            
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
            
            // Obtener configuración actual
            const style = document.getElementById('style').value;
            const steps = parseInt(document.getElementById('steps').value);
            
            // Actualizar debug info
            document.getElementById('currentPrompt').textContent = style || 'fotorealista';
            document.getElementById('currentSteps').textContent = steps;
            
            // Enviar datos con imagen
            const imageData = tempCanvas.toDataURL('image/jpeg', 0.8);
            ws.send(JSON.stringify({
                image: imageData,
                style: style,
                steps: steps,
                timestamp: Date.now()
            }));
            
            // Programar siguiente frame (limitado a 20 FPS)
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
            
            // Limpiar canvas
            if (ctx) {
                ctx.clearRect(0, 0, 512, 512);
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
            
            # Verificar que se recibió una imagen
            if 'image' not in data:
                print("ERROR: No se recibió imagen del cliente")
                continue
            
            # Medir tiempo de procesamiento
            start_time = time.time()
            
            # Decodificar imagen de entrada
            try:
                img_data = base64.b64decode(data['image'].split(',')[1])
                input_image = Image.open(io.BytesIO(img_data)).convert("RGB")
                print(f"Imagen recibida: {input_image.size}")
            except Exception as e:
                print(f"Error decodificando imagen: {e}")
                continue
            
            # Preparar prompt
            style = data.get('style', '')
            if style:
                prompt = f"professional photo in {style} style, high quality, detailed, sharp focus"
            else:
                prompt = "professional photo, high quality, detailed, sharp focus, photorealistic"
            
            # Obtener número de pasos
            steps = data.get('steps', 4)
            
            print(f"Procesando: '{prompt}' con {steps} pasos")
            
            # Generar imagen usando image-to-image
            try:
                result = processor.generate(prompt, input_image, steps=steps)
                
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
                import traceback
                traceback.print_exc()
            
    except Exception as e:
        print(f"Error en WebSocket: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Cliente desconectado")

if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*60)
    print("FAST DIFFUSION STREAM SERVER")
    print("="*60)
    print("GPU detectada correctamente")
    print("Modo: Image-to-Image Style Transfer")
    print("Modelo: LCM Dreamshaper v7")
    print("URL: http://0.0.0.0:8000")
    print("="*60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
