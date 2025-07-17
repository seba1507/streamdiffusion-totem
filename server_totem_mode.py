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
        self.frame_queue = queue.Queue(maxsize=2)
        self.result_queue = queue.Queue(maxsize=3)
        self.processing = False
        
        # Similarity filter
        self.last_image_hash = None
        self.similarity_threshold = 0.95
        self.frame_skip_counter = 0
        
        # Performance metrics
        self.total_frames = 0
        self.skipped_frames = 0
        
        self.init_model()
        
    def init_model(self):
        print("Cargando SD-Turbo optimizado...")
        
        model_id = "stabilityai/sd-turbo"
        
        self.pipe = AutoPipelineForImage2Image.from_pretrained(
            model_id,
            torch_dtype=self.dtype,
            variant="fp16",
            safety_checker=None,
            requires_safety_checker=False
        )
        
        self.pipe = self.pipe.to(self.device)
        
        # Optimizaciones graduales con manejo de errores
        self.pipe.set_progress_bar_config(disable=True)
        
        # Intentar XFormers primero
        xformers_enabled = False
        try:
            self.pipe.enable_xformers_memory_efficient_attention()
            xformers_enabled = True
            print("✓ XFormers habilitado")
        except Exception as e:
            print(f"⚠ XFormers no disponible: {e}")
        
        # Compilación de torch.compile - solo si XFormers no está habilitado
        compile_enabled = False
        if not xformers_enabled:
            try:
                torch._dynamo.config.suppress_errors = True
                
                self.pipe.unet = torch.compile(
                    self.pipe.unet, 
                    mode="reduce-overhead", 
                    fullgraph=False
                )
                compile_enabled = True
                print("✓ UNet compilado con torch.compile")
            except Exception as e:
                print(f"⚠ torch.compile no disponible: {e}")
        else:
            print("⚠ torch.compile deshabilitado (conflicto con XFormers)")
        
        # VAE optimizations
        try:
            self.pipe.vae.enable_slicing()
            self.pipe.vae.enable_tiling()
            print("✓ VAE optimizado (slicing + tiling)")
        except Exception as e:
            print(f"⚠ Optimizaciones VAE fallaron: {e}")
        
        # Pre-calentar con configuración segura
        print("Pre-calentando pipeline...")
        dummy_image = Image.new('RGB', (512, 512), color='black')
        
        try:
            with torch.no_grad():
                if xformers_enabled or compile_enabled:
                    with torch.cuda.amp.autocast():
                        _ = self.pipe(
                            "test",
                            image=dummy_image,
                            num_inference_steps=2,
                            strength=0.8,
                            guidance_scale=0.0
                        ).images[0]
                else:
                    _ = self.pipe(
                        "test",
                        image=dummy_image,
                        num_inference_steps=2,
                        strength=0.8,
                        guidance_scale=0.0
                    ).images[0]
            print("✓ Pre-calentamiento exitoso")
        except Exception as e:
            print(f"⚠ Error en pre-calentamiento: {e}")
            print("Continuando sin pre-calentamiento...")
        
        print("✓ Modelo listo!")
        
        # Performance summary
        optimizations = []
        if xformers_enabled:
            optimizations.append("XFormers")
        if compile_enabled:
            optimizations.append("torch.compile")
        optimizations.append("VAE optimizado")
        
        print(f"Optimizaciones activas: {', '.join(optimizations)}")
        
        # Iniciar thread de procesamiento
        self.start_processing_thread()
    
    def calculate_image_hash(self, image):
        """Calcular hash rápido de imagen para similarity filter"""
        try:
            small_img = image.resize((64, 64))
            img_array = np.array(small_img)
            return hashlib.md5(img_array.tobytes()).hexdigest()
        except Exception:
            return str(time.time())
    
    def should_skip_frame(self, image):
        """Implementar similarity filter"""
        try:
            current_hash = self.calculate_image_hash(image)
            
            if self.last_image_hash is None:
                self.last_image_hash = current_hash
                return False
            
            if current_hash == self.last_image_hash:
                self.frame_skip_counter += 1
                return True
            
            self.last_image_hash = current_hash
            self.frame_skip_counter = 0
            return False
        except Exception:
            return False
    
    def start_processing_thread(self):
        """Iniciar thread de procesamiento continuo"""
        self.processing = True
        processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        processing_thread.start()
        print("✓ Thread de procesamiento iniciado")
    
    def calculate_optimal_steps(self, strength):
        """Calcular pasos óptimos basado en strength para SD-Turbo"""
        # Para SD-Turbo: num_inference_steps * strength debe ser >= 1
        if strength >= 1.0:
            return 1
        elif strength >= 0.5:
            return 2
        elif strength >= 0.25:
            return 4
        else:
            return max(1, int(1.0 / strength))
    
    def _processing_loop(self):
        """Loop principal de procesamiento en thread separado"""
        while self.processing:
            try:
                frame_data = self.frame_queue.get(timeout=0.1)
                
                start_time = time.time()
                
                # Similarity filter
                if self.should_skip_frame(frame_data['image']):
                    self.skipped_frames += 1
                    try:
                        last_result = self.result_queue.queue[-1] if self.result_queue.queue else None
                        if last_result:
                            self.result_queue.put_nowait(last_result)
                    except:
                        pass
                    continue
                
                # Calcular pasos óptimos basado en strength
                strength = frame_data.get('strength', 0.8)
                steps = self.calculate_optimal_steps(strength)
                guidance_scale = frame_data.get('guidance_scale', 0.0)
                
                # Generar con SD-Turbo usando solo el prompt personalizado
                try:
                    with torch.no_grad():
                        use_autocast = True
                        try:
                            if use_autocast:
                                with torch.cuda.amp.autocast():
                                    result = self.pipe(
                                        prompt=frame_data['prompt'],
                                        image=frame_data['image'],
                                        num_inference_steps=steps,
                                        strength=strength,
                                        guidance_scale=guidance_scale,
                                        generator=torch.Generator(device=self.device).manual_seed(42)
                                    ).images[0]
                            else:
                                result = self.pipe(
                                    prompt=frame_data['prompt'],
                                    image=frame_data['image'],
                                    num_inference_steps=steps,
                                    strength=strength,
                                    guidance_scale=guidance_scale,
                                    generator=torch.Generator(device=self.device).manual_seed(42)
                                ).images[0]
                        except Exception as autocast_error:
                            print(f"Autocast falló, usando modo estándar: {autocast_error}")
                            result = self.pipe(
                                prompt=frame_data['prompt'],
                                image=frame_data['image'],
                                num_inference_steps=steps,
                                strength=strength,
                                guidance_scale=guidance_scale
                            ).images[0]
                    
                    processing_time = (time.time() - start_time) * 1000
                    self.total_frames += 1
                    
                    result_data = {
                        'image': result,
                        'processing_time': processing_time,
                        'timestamp': frame_data['timestamp'],
                        'stats': {
                            'total_frames': self.total_frames,
                            'skipped_frames': self.skipped_frames,
                            'skip_rate': self.skipped_frames / max(1, self.total_frames) * 100,
                            'strength': strength,
                            'steps': steps,
                            'guidance_scale': guidance_scale
                        }
                    }
                    
                    try:
                        self.result_queue.put_nowait(result_data)
                    except queue.Full:
                        try:
                            self.result_queue.get_nowait()
                            self.result_queue.put_nowait(result_data)
                        except:
                            pass
                            
                    print(f"Frame procesado: {processing_time:.1f}ms (Strength: {strength}, Steps: {steps})")
                    
                except Exception as process_error:
                    print(f"Error procesando frame: {process_error}")
                    continue
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error en processing loop: {e}")
                continue
    
    def add_frame(self, image, prompt, strength, guidance_scale, timestamp):
        """Agregar frame para procesamiento con parámetros ajustables"""
        frame_data = {
            'image': image,
            'prompt': prompt,
            'strength': strength,
            'guidance_scale': guidance_scale,
            'timestamp': timestamp
        }
        
        try:
            self.frame_queue.put_nowait(frame_data)
        except queue.Full:
            try:
                self.frame_queue.get_nowait()
                self.frame_queue.put_nowait(frame_data)
            except:
                pass
    
    def get_latest_result(self):
        """Obtener resultado más reciente (non-blocking)"""
        result = None
        while not self.result_queue.empty():
            try:
                result = self.result_queue.get_nowait()
            except queue.Empty:
                break
        return result

# Instancia global
processor = UltraFastDiffusion()

# HTML con controles simplificados y modo totem
HTML_CONTENT = """
<!DOCTYPE html>
<html>
<head>
    <title>SD-Turbo Totem Mode</title>
    <style>
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }
        
        body {
            background: #000;
            color: #fff;
            font-family: 'Consolas', monospace;
            overflow: hidden;
            height: 100vh;
        }
        
        .container {
            display: flex;
            height: 100vh;
            align-items: center;
            justify-content: center;
            gap: 20px;
            padding: 20px;
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
            left: 20px;
            right: 20px;
            background: rgba(0,0,0,0.95);
            padding: 20px;
            border-radius: 15px;
            border: 1px solid #00ff41;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            z-index: 1000;
        }
        
        .control-group {
            display: flex;
            flex-direction: column;
            gap: 10px;
        }
        
        .control-label {
            color: #00ff41;
            font-weight: bold;
            font-size: 14px;
            text-transform: uppercase;
            margin-bottom: 5px;
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
            text-transform: uppercase;
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
        
        .totem-btn {
            background: linear-gradient(45deg, #ff6b00, #ff4500);
            color: #fff;
            font-size: 18px;
            padding: 15px 30px;
        }
        
        .totem-btn:hover {
            box-shadow: 0 0 20px #ff6b00;
        }
        
        input[type="range"] {
            width: 100%;
            height: 8px;
            -webkit-appearance: none;
            background: #333;
            outline: none;
            border-radius: 4px;
        }
        
        input[type="range"]::-webkit-slider-thumb {
            -webkit-appearance: none;
            width: 24px;
            height: 24px;
            background: #00ff41;
            border-radius: 50%;
            cursor: pointer;
        }
        
        textarea {
            padding: 12px;
            font-size: 14px;
            border-radius: 8px;
            background: #222;
            color: #fff;
            border: 1px solid #00ff41;
            font-family: 'Consolas', monospace;
            resize: vertical;
            min-height: 80px;
            max-height: 120px;
        }
        
        .range-value {
            color: #00ff41;
            font-weight: bold;
            text-align: center;
            font-size: 18px;
            margin-top: 8px;
        }
        
        .stats {
            position: fixed;
            top: 20px;
            right: 20px;
            background: rgba(0,0,0,0.95);
            padding: 20px;
            border-radius: 15px;
            font-family: 'Consolas', monospace;
            font-size: 12px;
            border: 1px solid #00ff41;
            min-width: 250px;
            z-index: 1000;
        }
        
        .stat-value {
            color: #00ff41;
            font-weight: bold;
        }
        
        .stat-row {
            display: flex;
            justify-content: space-between;
            margin: 8px 0;
            border-bottom: 1px solid #333;
            padding-bottom: 6px;
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
        
        .title {
            text-align: center;
            margin-bottom: 15px;
            color: #00ff41;
            font-size: 16px;
            font-weight: bold;
            text-transform: uppercase;
        }
        
        /* Modo Totem - Pantalla Completa Vertical */
        .totem-mode {
            position: fixed;
            top: 0;
            left: 0;
            width: 100vw;
            height: 100vh;
            background: #000;
            z-index: 9999;
            display: none;
            justify-content: center;
            align-items: center;
        }
        
        .totem-canvas {
            max-width: 70vh;
            max-height: 90vh;
            width: auto;
            height: auto;
            border: none;
            border-radius: 0;
            box-shadow: 0 0 50px rgba(0, 255, 65, 0.3);
        }
        
        .totem-exit {
            position: absolute;
            top: 20px;
            right: 20px;
            background: rgba(255, 0, 0, 0.8);
            color: white;
            border: none;
            border-radius: 50%;
            width: 60px;
            height: 60px;
            font-size: 24px;
            cursor: pointer;
            z-index: 10000;
        }
        
        .totem-info {
            position: absolute;
            bottom: 20px;
            left: 50%;
            transform: translateX(-50%);
            background: rgba(0,0,0,0.8);
            padding: 15px 30px;
            border-radius: 10px;
            border: 1px solid #00ff41;
            text-align: center;
        }
        
        @media (max-width: 768px) {
            .controls {
                grid-template-columns: 1fr;
                bottom: 10px;
                left: 10px;
                right: 10px;
                padding: 15px;
            }
            
            .container {
                flex-direction: column;
                gap: 10px;
            }
            
            video, canvas {
                width: 90vw;
                height: 90vw;
                max-width: 400px;
                max-height: 400px;
            }
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
    
    <!-- Modo Totem -->
    <div class="totem-mode" id="totemMode">
        <button class="totem-exit" onclick="exitTotemMode()">✕</button>
        <canvas id="totemCanvas" class="totem-canvas"></canvas>
        <div class="totem-info">
            <div style="color: #00ff41; font-size: 18px; font-weight: bold;">MODO TOTEM ACTIVO</div>
            <div style="margin-top: 8px;">Transformación en tiempo real con IA</div>
        </div>
    </div>
    
    <div class="stats">
        <div class="title">Métricas del Sistema</div>
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
            <span>Skip Rate:</span>
            <span id="skipRate" class="stat-value">0</span>%
        </div>
        <div class="stat-row">
            <span>Strength:</span>
            <span id="currentStrength" class="stat-value">0.8</span>
        </div>
        <div class="stat-row">
            <span>Guidance:</span>
            <span id="currentGuidance" class="stat-value">0.0</span>
        </div>
    </div>
    
    <div class="controls">
        <div class="control-group">
            <div class="control-label">Control del Sistema</div>
            <button id="startBtn" onclick="start()">🚀 Iniciar Stream</button>
            <button id="stopBtn" onclick="stop()" disabled>⏹ Detener</button>
            <button class="totem-btn" onclick="enterTotemMode()" id="totemBtn" disabled>📺 Modo Totem</button>
        </div>
        
        <div class="control-group">
            <div class="control-label">Prompt de Transformación</div>
            <textarea id="customPrompt" placeholder="Describe el estilo que deseas aplicar (ej: cyberpunk style, neon lights, futuristic city)">cyberpunk style, neon lights, futuristic, high quality, detailed</textarea>
        </div>
        
        <div class="control-group">
            <div class="control-label">Intensidad de Transformación</div>
            <input type="range" id="strengthSlider" min="0.3" max="1.0" step="0.1" value="0.8" oninput="updateStrengthValue()">
            <div class="range-value" id="strengthValue">0.8</div>
        </div>
        
        <div class="control-group">
            <div class="control-label">Adherencia al Prompt</div>
            <input type="range" id="guidanceSlider" min="0.0" max="3.0" step="0.1" value="1.0" oninput="updateGuidanceValue()">
            <div class="range-value" id="guidanceValue">1.0</div>
        </div>
    </div>
    
    <script>
        let ws = null;
        let streaming = false;
        let video = null;
        let canvas = null;
        let ctx = null;
        let totemCanvas = null;
        let totemCtx = null;
        let totemMode = false;
        
        let frameCount = 0;
        let totalFrames = 0;
        let lastTime = Date.now();
        let latencies = [];
        
        function updateStrengthValue() {
            const value = document.getElementById('strengthSlider').value;
            document.getElementById('strengthValue').textContent = value;
        }
        
        function updateGuidanceValue() {
            const value = document.getElementById('guidanceSlider').value;
            document.getElementById('guidanceValue').textContent = value;
        }
        
        function enterTotemMode() {
            if (!streaming) {
                alert('Debe iniciar el stream antes de activar el modo totem');
                return;
            }
            
            totemMode = true;
            document.getElementById('totemMode').style.display = 'flex';
            
            // Configurar canvas del totem
            totemCanvas = document.getElementById('totemCanvas');
            totemCtx = totemCanvas.getContext('2d');
            
            // Configurar dimensiones para pantalla vertical
            const screenHeight = window.innerHeight;
            const screenWidth = window.innerWidth;
            
            // Calcular dimensiones óptimas para orientación vertical
            if (screenHeight > screenWidth) {
                // Pantalla ya es vertical
                totemCanvas.width = screenWidth * 0.8;
                totemCanvas.height = screenWidth * 0.8;
            } else {
                // Pantalla horizontal - optimizar para visualización vertical
                totemCanvas.width = screenHeight * 0.7;
                totemCanvas.height = screenHeight * 0.7;
            }
            
            // Ocultar controles y stats en modo totem
            document.querySelector('.controls').style.display = 'none';
            document.querySelector('.stats').style.display = 'none';
            
            console.log('Modo Totem activado');
        }
        
        function exitTotemMode() {
            totemMode = false;
            document.getElementById('totemMode').style.display = 'none';
            
            // Mostrar controles y stats
            document.querySelector('.controls').style.display = 'grid';
            document.querySelector('.stats').style.display = 'block';
            
            console.log('Modo Totem desactivado');
        }
        
        async function start() {
            try {
                video = document.getElementById('video');
                canvas = document.getElementById('output');
                ctx = canvas.getContext('2d');
                canvas.width = 512;
                canvas.height = 512;
                
                const stream = await navigator.mediaDevices.getUserMedia({
                    video: {
                        width: { ideal: 1280 },
                        height: { ideal: 720 },
                        frameRate: { ideal: 30 }
                    }
                });
                video.srcObject = stream;
                
                await new Promise(resolve => {
                    video.onloadedmetadata = resolve;
                });
                
                const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                ws = new WebSocket(`${protocol}//${window.location.host}/ws`);
                
                ws.onopen = () => {
                    console.log('Conectado al servidor');
                    streaming = true;
                    document.getElementById('startBtn').disabled = true;
                    document.getElementById('stopBtn').disabled = false;
                    document.getElementById('totemBtn').disabled = false;
                    document.getElementById('statusIndicator').classList.add('active');
                    sendFrame();
                };
                
                ws.onmessage = (event) => {
                    const data = JSON.parse(event.data);
                    if (data.image) {
                        const img = new Image();
                        img.onload = () => {
                            // Actualizar canvas principal
                            ctx.clearRect(0, 0, 512, 512);
                            ctx.drawImage(img, 0, 0, 512, 512);
                            
                            // Actualizar canvas del totem si está activo
                            if (totemMode && totemCtx) {
                                totemCtx.clearRect(0, 0, totemCanvas.width, totemCanvas.height);
                                totemCtx.drawImage(img, 0, 0, totemCanvas.width, totemCanvas.height);
                            }
                        };
                        img.src = data.image;
                        updateStats(data);
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
            
            if (video.videoWidth === 0 || video.videoHeight === 0) {
                setTimeout(sendFrame, 10);
                return;
            }
            
            const tempCanvas = document.createElement('canvas');
            tempCanvas.width = 512;
            tempCanvas.height = 512;
            const tempCtx = tempCanvas.getContext('2d');
            
            // Fit por altura - cortar lados si es necesario
            const videoAspect = video.videoWidth / video.videoHeight;
            const targetAspect = 1; // 512/512 = 1
            
            let sx, sy, sw, sh;
            
            if (videoAspect > targetAspect) {
                // Video más ancho - cortar lados
                sh = video.videoHeight;
                sw = video.videoHeight;
                sx = (video.videoWidth - sw) / 2;
                sy = 0;
            } else {
                // Video más alto - cortar arriba/abajo
                sw = video.videoWidth;
                sh = video.videoWidth;
                sx = 0;
                sy = (video.videoHeight - sh) / 2;
            }
            
            tempCtx.drawImage(video, sx, sy, sw, sh, 0, 0, 512, 512);
            
            // Obtener parámetros únicamente del prompt personalizado
            const prompt = document.getElementById('customPrompt').value || "high quality, detailed, photorealistic";
            const strength = parseFloat(document.getElementById('strengthSlider').value);
            const guidance = parseFloat(document.getElementById('guidanceSlider').value);
            
            const imageData = tempCanvas.toDataURL('image/jpeg', 0.95);
            ws.send(JSON.stringify({
                image: imageData,
                prompt: prompt,
                strength: strength,
                guidance_scale: guidance,
                timestamp: Date.now()
            }));
            
            requestAnimationFrame(sendFrame);
        }
        
        function updateStats(data) {
            frameCount++;
            totalFrames++;
            
            if (data.processing_time) {
                latencies.push(data.processing_time);
                if (latencies.length > 50) latencies.shift();
            }
            
            const now = Date.now();
            if (now - lastTime >= 1000) {
                document.getElementById('fps').textContent = frameCount;
                frameCount = 0;
                lastTime = now;
                
                if (latencies.length > 0) {
                    const avgLatency = latencies.reduce((a, b) => a + b) / latencies.length;
                    document.getElementById('latency').textContent = Math.round(avgLatency);
                }
            }
            
            if (data.stats) {
                document.getElementById('totalFrames').textContent = data.stats.total_frames;
                document.getElementById('skipRate').textContent = data.stats.skip_rate.toFixed(1);
                document.getElementById('currentStrength').textContent = data.stats.strength;
                document.getElementById('currentGuidance').textContent = data.stats.guidance_scale;
            }
        }
        
        function stop() {
            streaming = false;
            
            if (totemMode) {
                exitTotemMode();
            }
            
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
            document.getElementById('totemBtn').disabled = true;
            document.getElementById('statusIndicator').classList.remove('active');
        }
        
        // Manejo de teclas para modo totem
        document.addEventListener('keydown', function(event) {
            if (event.key === 'Escape' && totemMode) {
                exitTotemMode();
            }
            if (event.key === 'F11') {
                event.preventDefault();
                if (!totemMode && streaming) {
                    enterTotemMode();
                }
            }
        });
        
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
    print("🚀 Cliente conectado - Modo Totem SD-Turbo")
    
    try:
        while True:
            data = await websocket.receive_json()
            
            if 'image' not in data:
                continue
            
            try:
                img_data = base64.b64decode(data['image'].split(',')[1])
                input_image = Image.open(io.BytesIO(img_data)).convert("RGB")
                
                # Asegurar dimensiones exactas
                if input_image.size != (512, 512):
                    input_image = input_image.resize((512, 512), Image.Resampling.LANCZOS)
                
            except Exception as e:
                print(f"❌ Error decodificando imagen: {e}")
                continue
            
            # Usar únicamente el prompt personalizado del usuario
            prompt = data.get('prompt', 'high quality, detailed, photorealistic')
            strength = data.get('strength', 0.8)
            guidance_scale = data.get('guidance_scale', 1.0)
            
            processor.add_frame(
                input_image, 
                prompt,
                strength,
                guidance_scale,
                data.get('timestamp', time.time() * 1000)
            )
            
            result = processor.get_latest_result()
            
            if result:
                buffered = io.BytesIO()
                result['image'].save(buffered, format="JPEG", quality=95)
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
    print("🎪 SD-TURBO TOTEM MODE - INSTALACIÓN INTERACTIVA")
    print("="*70)
    print("⚡ Generación basada únicamente en prompt personalizado")
    print("📺 Modo Totem con pantalla completa vertical")
    print("🎨 Transformaciones en tiempo real para activaciones")
    print("🌐 URL: http://0.0.0.0:8000")
    print("="*70 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
