#!/usr/bin/env python3
import os
import sys
import time
import asyncio
import threading
import queue
from collections import deque
import hashlib

# Configurar CUDA antes de importar torch
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

import torch
import numpy as np

# Verificar CUDA
if not torch.cuda.is_available():
    print("ERROR: CUDA no disponible")
    sys.exit(1)

print(f"CUDA OK: {torch.cuda.get_device_name(0)}")

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from diffusers import AutoPipelineForImage2Image
import base64
from PIL import Image
import io
import json

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class UltraFastDiffusion:
    def __init__(self):
        self.pipe = None
        self.device = torch.device("cuda:0")
        self.dtype = torch.float16
        
        # Stream processing optimizations
        self.frame_queue = queue.Queue(maxsize=2)  # Pequeño buffer para evitar latencia
        self.result_queue = queue.Queue(maxsize=3)
        self.processing = False
        
        # Similarity filter (inspirado en dotsimulate)
        self.last_image_hash = None
        self.similarity_threshold = 0.95
        self.frame_skip_counter = 0
        
        # Temporal smoothing
        self.last_latent = None
        self.latent_momentum = 0.3  # Factor de suavizado
        
        # Performance metrics
        self.total_frames = 0
        self.skipped_frames = 0
        
        self.init_model()
        
    def init_model(self):
        print("Cargando SD-Turbo optimizado...")
        
        # SD-Turbo: modelo diseñado para 1 paso
        model_id = "stabilityai/sd-turbo"
        
        self.pipe = AutoPipelineForImage2Image.from_pretrained(
            model_id,
            torch_dtype=self.dtype,
            variant="fp16",
            safety_checker=None,
            requires_safety_checker=False
        )
        
        self.pipe = self.pipe.to(self.device)
        
        # Optimizaciones críticas
        self.pipe.set_progress_bar_config(disable=True)
        
        # XFormers para eficiencia de memoria
        try:
            self.pipe.enable_xformers_memory_efficient_attention()
            print("✓ XFormers habilitado")
        except:
            print("⚠ XFormers no disponible")
        
        # Compilación con torch.compile para máxima velocidad
        try:
            self.pipe.unet = torch.compile(
                self.pipe.unet, 
                mode="reduce-overhead", 
                fullgraph=True
            )
            print("✓ UNet compilado con torch.compile")
        except:
            print("⚠ torch.compile no disponible")
        
        # VAE optimizations
        self.pipe.vae.enable_slicing()
        self.pipe.vae.enable_tiling()
        
        # Pre-calentar con imagen dummy
        print("Pre-calentando pipeline...")
        dummy_image = Image.new('RGB', (512, 512), color='black')
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                _ = self.pipe(
                    "test",
                    image=dummy_image,
                    num_inference_steps=1,
                    strength=1.0,  # Para asegurar que steps * strength >= 1
                    guidance_scale=0.0  # SD-Turbo no usa guidance
                ).images[0]
        
        print("✓ Modelo listo para ultra-velocidad!")
        
        # Iniciar thread de procesamiento
        self.start_processing_thread()
    
    def calculate_image_hash(self, image):
        """Calcular hash rápido de imagen para similarity filter"""
        # Redimensionar a 64x64 para hash rápido
        small_img = image.resize((64, 64))
        img_array = np.array(small_img)
        return hashlib.md5(img_array.tobytes()).hexdigest()
    
    def should_skip_frame(self, image):
        """Implementar similarity filter como dotsimulate"""
        current_hash = self.calculate_image_hash(image)
        
        if self.last_image_hash is None:
            self.last_image_hash = current_hash
            return False
        
        # Calcular similitud simple basada en hash
        if current_hash == self.last_image_hash:
            self.frame_skip_counter += 1
            return True
        
        self.last_image_hash = current_hash
        self.frame_skip_counter = 0
        return False
    
    def start_processing_thread(self):
        """Iniciar thread de procesamiento continuo"""
        self.processing = True
        processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        processing_thread.start()
        print("✓ Thread de procesamiento iniciado")
    
    def _processing_loop(self):
        """Loop principal de procesamiento en thread separado"""
        while self.processing:
            try:
                # Obtener próximo frame (timeout corto para responsividad)
                frame_data = self.frame_queue.get(timeout=0.1)
                
                start_time = time.time()
                
                # Similarity filter - skip si es muy similar al anterior
                if self.should_skip_frame(frame_data['image']):
                    self.skipped_frames += 1
                    # Reenviar último resultado si existe
                    try:
                        last_result = self.result_queue.queue[-1] if self.result_queue.queue else None
                        if last_result:
                            self.result_queue.put_nowait(last_result)
                    except:
                        pass
                    continue
                
                # Generar con SD-Turbo optimizado
                with torch.no_grad():
                    with torch.cuda.amp.autocast():
                        result = self.pipe(
                            prompt=frame_data['prompt'],
                            image=frame_data['image'],
                            num_inference_steps=1,  # SD-Turbo: 1 paso óptimo
                            strength=1.0,  # Para asegurar steps * strength >= 1
                            guidance_scale=0.0,  # SD-Turbo no usa guidance
                            generator=torch.Generator(device=self.device).manual_seed(42)  # Seed fijo para consistencia
                        ).images[0]
                
                processing_time = (time.time() - start_time) * 1000
                self.total_frames += 1
                
                # Agregar resultado a cola de salida
                result_data = {
                    'image': result,
                    'processing_time': processing_time,
                    'timestamp': frame_data['timestamp'],
                    'stats': {
                        'total_frames': self.total_frames,
                        'skipped_frames': self.skipped_frames,
                        'skip_rate': self.skipped_frames / max(1, self.total_frames) * 100
                    }
                }
                
                # Usar put_nowait para evitar bloqueos
                try:
                    self.result_queue.put_nowait(result_data)
                except queue.Full:
                    # Si la cola está llena, remover el más viejo
                    try:
                        self.result_queue.get_nowait()
                        self.result_queue.put_nowait(result_data)
                    except:
                        pass
                        
                print(f"Frame procesado: {processing_time:.1f}ms (Skip rate: {self.skipped_frames}/{self.total_frames})")
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error en processing loop: {e}")
                continue
    
    def add_frame(self, image, prompt, timestamp):
        """Agregar frame para procesamiento (non-blocking)"""
        frame_data = {
            'image': image,
            'prompt': prompt,
            'timestamp': timestamp
        }
        
        try:
            self.frame_queue.put_nowait(frame_data)
        except queue.Full:
            # Si está lleno, remover el más viejo y agregar el nuevo
            try:
                self.frame_queue.get_nowait()
                self.frame_queue.put_nowait(frame_data)
            except:
                pass
    
    def get_latest_result(self):
        """Obtener resultado más reciente (non-blocking)"""
        result = None
        # Vaciar cola y quedarse con el último
        while not self.result_queue.empty():
            try:
                result = self.result_queue.get_nowait()
            except queue.Empty:
                break
        return result

# Instancia global
processor = UltraFastDiffusion()

# HTML optimizado con smoothing visual
HTML_CONTENT = """
<!DOCTYPE html>
<html>
<head>
    <title>Ultra Fast SD-Turbo Stream</title>
    <style>
        body {
            margin: 0;
            background: #000;
            color: #fff;
            font-family: 'Consolas', monospace;
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
            border: 2px solid #00ff41;
            background: #111;
            border-radius: 8px;
        }
        .output-container {
            position: relative;
        }
        .controls {
            position: fixed;
            bottom: 20px;
            left: 50%;
            transform: translateX(-50%);
            background: rgba(0,0,0,0.9);
            padding: 20px;
            border-radius: 15px;
            display: flex;
            gap: 15px;
            align-items: center;
            border: 1px solid #00ff41;
        }
        button {
            padding: 12px 24px;
            font-size: 16px;
            cursor: pointer;
            background: linear-gradient(45deg, #00ff41, #00cc33);
            color: #000;
            border: none;
            border-radius: 8px;
            font-weight: bold;
            transition: all 0.3s;
        }
        button:hover {
            transform: scale(1.05);
            box-shadow: 0 0 20px #00ff41;
        }
        button:disabled {
            background: #333;
            color: #666;
            cursor: not-allowed;
            transform: none;
            box-shadow: none;
        }
        select {
            padding: 12px;
            font-size: 16px;
            border-radius: 8px;
            background: #222;
            color: #fff;
            border: 1px solid #00ff41;
        }
        .stats {
            position: fixed;
            top: 20px;
            right: 20px;
            background: rgba(0,0,0,0.9);
            padding: 20px;
            border-radius: 15px;
            font-family: 'Consolas', monospace;
            font-size: 14px;
            border: 1px solid #00ff41;
            min-width: 250px;
        }
        .stat-value {
            color: #00ff41;
            font-weight: bold;
            font-size: 16px;
        }
        .stat-row {
            display: flex;
            justify-content: space-between;
            margin: 8px 0;
        }
        .status-indicator {
            position: absolute;
            top: 10px;
            left: 10px;
            width: 20px;
            height: 20px;
            border-radius: 50%;
            background: #ff4444;
            transition: all 0.3s;
        }
        .status-indicator.active {
            background: #00ff41;
            box-shadow: 0 0 15px #00ff41;
        }
        .performance-graph {
            margin-top: 15px;
            height: 60px;
            border: 1px solid #333;
            background: #111;
            position: relative;
            overflow: hidden;
        }
        .title {
            text-align: center;
            margin-bottom: 10px;
            color: #00ff41;
            font-size: 18px;
            font-weight: bold;
        }
    </style>
</head>
<body>
    <div class="container">
        <video id="video" autoplay muted playsinline></video>
        <div class="output-container">
            <canvas id="output"></canvas>
            <div class="status-indicator" id="statusIndicator"></div>
        </div>
    </div>
    
    <div class="stats">
        <div class="title">ULTRA FAST METRICS</div>
        <div class="stat-row">
            <span>FPS:</span>
            <span id="fps" class="stat-value">0</span>
        </div>
        <div class="stat-row">
            <span>Latencia:</span>
            <span id="latency" class="stat-value">0</span>ms
        </div>
        <div class="stat-row">
            <span>Total Frames:</span>
            <span id="totalFrames" class="stat-value">0</span>
        </div>
        <div class="stat-row">
            <span>Frames Saltados:</span>
            <span id="skippedFrames" class="stat-value">0</span>
        </div>
        <div class="stat-row">
            <span>Skip Rate:</span>
            <span id="skipRate" class="stat-value">0</span>%
        </div>
        <div class="stat-row">
            <span>GPU Efficiency:</span>
            <span id="efficiency" class="stat-value">0</span>%
        </div>
    </div>
    
    <div class="controls">
        <button id="startBtn" onclick="start()">🚀 INICIAR TURBO</button>
        <button id="stopBtn" onclick="stop()" disabled>⏹ DETENER</button>
        <select id="style">
            <option value="">Fotorealista</option>
            <option value="cyberpunk futuristic">Cyberpunk</option>
            <option value="anime style">Anime</option>
            <option value="oil painting masterpiece">Óleo</option>
            <option value="watercolor painting">Acuarela</option>
            <option value="digital art">Arte Digital</option>
            <option value="cinematic movie">Cinematográfico</option>
        </select>
    </div>
    
    <script>
        let ws = null;
        let streaming = false;
        let video = null;
        let canvas = null;
        let ctx = null;
        
        // Estadísticas avanzadas
        let frameCount = 0;
        let totalFrames = 0;
        let lastTime = Date.now();
        let latencies = [];
        let fpsHistory = [];
        
        // Smoothing temporal
        let lastImageData = null;
        const smoothingFactor = 0.7;
        
        async function start() {
            try {
                video = document.getElementById('video');
                canvas = document.getElementById('output');
                ctx = canvas.getContext('2d');
                canvas.width = 512;
                canvas.height = 512;
                
                // Obtener cámara con configuración optimizada
                const stream = await navigator.mediaDevices.getUserMedia({
                    video: {
                        width: { ideal: 1280, max: 1920 },
                        height: { ideal: 720, max: 1080 },
                        frameRate: { ideal: 30, max: 60 }
                    }
                });
                video.srcObject = stream;
                
                // Esperar que el video esté listo
                await new Promise(resolve => {
                    video.onloadedmetadata = resolve;
                });
                
                // Conectar WebSocket
                const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                ws = new WebSocket(`${protocol}//${window.location.host}/ws`);
                
                ws.onopen = () => {
                    console.log('🚀 Conectado al servidor ultra-rápido');
                    streaming = true;
                    document.getElementById('startBtn').disabled = true;
                    document.getElementById('stopBtn').disabled = false;
                    document.getElementById('statusIndicator').classList.add('active');
                    sendFrame();
                };
                
                ws.onmessage = (event) => {
                    const data = JSON.parse(event.data);
                    if (data.image) {
                        displayResult(data);
                        updateAdvancedStats(data);
                    }
                };
                
                ws.onerror = (error) => {
                    console.error('❌ Error WebSocket:', error);
                    stop();
                };
                
                ws.onclose = () => {
                    console.log('🔌 Desconectado del servidor');
                    stop();
                };
                
            } catch (error) {
                console.error('❌ Error:', error);
                alert('Error: ' + error.message);
            }
        }
        
        function sendFrame() {
            if (!streaming || !ws || ws.readyState !== WebSocket.OPEN) return;
            
            if (video.videoWidth === 0 || video.videoHeight === 0) {
                setTimeout(sendFrame, 10);
                return;
            }
            
            // Capturar frame optimizado
            const tempCanvas = document.createElement('canvas');
            tempCanvas.width = 512;
            tempCanvas.height = 512;
            const tempCtx = tempCanvas.getContext('2d');
            
            // Centrar y recortar video
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
            
            // Preparar prompt optimizado para SD-Turbo
            const style = document.getElementById('style').value;
            let prompt = style ? `${style}, high quality, detailed` : "high quality, detailed, photorealistic";
            
            // Enviar frame
            const imageData = tempCanvas.toDataURL('image/jpeg', 0.9);
            ws.send(JSON.stringify({
                image: imageData,
                prompt: prompt,
                timestamp: Date.now()
            }));
            
            // Próximo frame inmediatamente (máxima velocidad)
            requestAnimationFrame(sendFrame);
        }
        
        function displayResult(data) {
            const img = new Image();
            img.onload = () => {
                ctx.clearRect(0, 0, 512, 512);
                ctx.drawImage(img, 0, 0, 512, 512);
            };
            img.src = data.image;
        }
        
        function updateAdvancedStats(data) {
            frameCount++;
            totalFrames++;
            
            if (data.processing_time) {
                latencies.push(data.processing_time);
                if (latencies.length > 50) latencies.shift();
            }
            
            // Actualizar stats cada segundo
            const now = Date.now();
            if (now - lastTime >= 1000) {
                const currentFps = frameCount;
                fpsHistory.push(currentFps);
                if (fpsHistory.length > 10) fpsHistory.shift();
                
                document.getElementById('fps').textContent = currentFps;
                frameCount = 0;
                lastTime = now;
                
                // Latencia promedio
                if (latencies.length > 0) {
                    const avgLatency = latencies.reduce((a, b) => a + b) / latencies.length;
                    document.getElementById('latency').textContent = Math.round(avgLatency);
                }
                
                // Eficiencia GPU (estimada)
                const avgFps = fpsHistory.reduce((a, b) => a + b, 0) / fpsHistory.length;
                const efficiency = Math.min(100, (avgFps / 20) * 100); // 20 FPS = 100%
                document.getElementById('efficiency').textContent = Math.round(efficiency);
            }
            
            // Stats del servidor
            if (data.stats) {
                document.getElementById('totalFrames').textContent = data.stats.total_frames;
                document.getElementById('skippedFrames').textContent = data.stats.skipped_frames;
                document.getElementById('skipRate').textContent = data.stats.skip_rate.toFixed(1);
            }
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
            
            if (ctx) {
                ctx.clearRect(0, 0, 512, 512);
            }
            
            document.getElementById('startBtn').disabled = false;
            document.getElementById('stopBtn').disabled = true;
            document.getElementById('statusIndicator').classList.remove('active');
        }
        
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
    print("🚀 Cliente conectado - Modo ULTRA FAST")
    
    try:
        while True:
            # Recibir frame del cliente
            data = await websocket.receive_json()
            
            if 'image' not in data:
                continue
            
            # Decodificar imagen
            try:
                img_data = base64.b64decode(data['image'].split(',')[1])
                input_image = Image.open(io.BytesIO(img_data)).convert("RGB")
                
                # Redimensionar a 512x512 para SD-Turbo
                input_image = input_image.resize((512, 512), Image.Resampling.LANCZOS)
                
            except Exception as e:
                print(f"❌ Error decodificando imagen: {e}")
                continue
            
            # Agregar a cola de procesamiento (non-blocking)
            processor.add_frame(
                input_image, 
                data.get('prompt', 'high quality, detailed'),
                data.get('timestamp', time.time() * 1000)
            )
            
            # Obtener resultado más reciente (si hay)
            result = processor.get_latest_result()
            
            if result:
                # Codificar y enviar resultado
                buffered = io.BytesIO()
                result['image'].save(buffered, format="JPEG", quality=90)
                img_str = base64.b64encode(buffered.getvalue()).decode()
                
                await websocket.send_json({
                    'image': f'data:image/jpeg;base64,{img_str}',
                    'processing_time': result['processing_time'],
                    'timestamp': result['timestamp'],
                    'stats': result['stats']
                })
            
    except Exception as e:
        print(f"❌ Error en WebSocket: {e}")
    finally:
        print("🔌 Cliente desconectado")

if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*70)
    print("🚀 ULTRA FAST SD-TURBO STREAM SERVER")
    print("="*70)
    print("⚡ SD-Turbo con 1 paso de inferencia")
    print("🎯 Similarity filter activo")
    print("🔄 Stream processing optimizado")
    print("💾 Temporal smoothing habilitado")
    print("🌐 URL: http://0.0.0.0:8000")
    print("="*70 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
