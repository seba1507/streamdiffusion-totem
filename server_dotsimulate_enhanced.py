#!/usr/bin/env python3
import os
import sys
import time
import asyncio
import threading
import queue
from collections import deque
import hashlib
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import cv2

# Configurar CUDA antes de importar torch
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

import torch
import torch.nn.functional as F

# Verificar CUDA
if not torch.cuda.is_available():
    print("ERROR: CUDA no disponible")
    sys.exit(1)

print(f"CUDA OK: {torch.cuda.get_device_name(0)}")

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from diffusers import AutoPipelineForImage2Image, LCMScheduler, DPMSolverMultistepScheduler
import base64
import io
import json

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class DotSimulateEnhancedDiffusion:
    def __init__(self):
        self.pipe = None
        self.device = torch.device("cuda:0")
        self.dtype = torch.float16
        
        # DotSimulate-style processing queues
        self.frame_queue = queue.Queue(maxsize=3)
        self.result_queue = queue.Queue(maxsize=5)
        self.processing = False
        
        # Enhanced similarity filter with stochastic properties
        self.last_image_hash = None
        self.similarity_threshold = 0.92
        self.frame_skip_counter = 0
        self.stochastic_filter_enabled = True
        
        # Temporal feedback system (DotSimulate technique)
        self.frame_buffer = deque(maxlen=3)
        self.temporal_smoothing = 0.15
        self.last_latent = None
        
        # Performance metrics
        self.total_frames = 0
        self.skipped_frames = 0
        self.preprocessing_time = 0
        
        # DotSimulate-style prompt enhancement
        self.quality_boosters = [
            "highly detailed", "8K resolution", "professional photography",
            "dramatic lighting", "sharp focus", "masterpiece", "ultra-detailed",
            "photorealistic", "award-winning", "trending on artstation"
        ]
        
        self.init_model()
        
    def init_model(self):
        print("Inicializando Enhanced StreamDiffusion (DotSimulate-style)...")
        
        # Usar LCM Dreamshaper para mejor calidad que SD-Turbo
        model_id = "SimianLuo/LCM_Dreamshaper_v7"
        
        self.pipe = AutoPipelineForImage2Image.from_pretrained(
            model_id,
            torch_dtype=self.dtype,
            safety_checker=None,
            requires_safety_checker=False,
            use_safetensors=True
        )
        
        self.pipe = self.pipe.to(self.device)
        
        # Configurar scheduler DPM++ 2M (DotSimulate default)
        self.pipe.scheduler = LCMScheduler.from_config(self.pipe.scheduler.config)
        
        # Optimizaciones avanzadas
        self.pipe.set_progress_bar_config(disable=True)
        
        # XFormers con manejo de errores
        xformers_enabled = False
        try:
            self.pipe.enable_xformers_memory_efficient_attention()
            xformers_enabled = True
            print("✓ XFormers memory-efficient attention habilitado")
        except Exception as e:
            print(f"⚠ XFormers no disponible: {e}")
        
        # TensorRT-style compilation (cuando disponible)
        if not xformers_enabled:
            try:
                torch._dynamo.config.suppress_errors = True
                self.pipe.unet = torch.compile(
                    self.pipe.unet, 
                    mode="reduce-overhead", 
                    fullgraph=False
                )
                print("✓ UNet compilado con torch.compile")
            except Exception as e:
                print(f"⚠ torch.compile no disponible: {e}")
        
        # VAE optimizations (TinyVAE-style)
        try:
            self.pipe.vae.enable_slicing()
            self.pipe.vae.enable_tiling()
            print("✓ VAE optimizado con slicing y tiling")
        except Exception as e:
            print(f"⚠ Optimizaciones VAE fallaron: {e}")
        
        # Pre-calentamiento con configuración DotSimulate
        print("Pre-calentando pipeline con configuración DotSimulate...")
        dummy_image = Image.new('RGB', (512, 512), color=(64, 64, 64))
        
        try:
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    _ = self.pipe(
                        "highly detailed test image, professional photography",
                        image=dummy_image,
                        num_inference_steps=4,
                        strength=0.75,
                        guidance_scale=8.0  # DotSimulate default range
                    ).images[0]
            print("✓ Pre-calentamiento exitoso con parámetros DotSimulate")
        except Exception as e:
            print(f"⚠ Error en pre-calentamiento: {e}")
        
        print("✓ Enhanced StreamDiffusion listo!")
        self.start_processing_thread()
    
    def preprocess_image_dotsimulate_style(self, image):
        """
        Preprocessing avanzado inspirado en técnicas de DotSimulate
        Crea texturas ricas y variaciones de densidad para mejores transformaciones
        """
        start_time = time.time()
        
        # Conversión a numpy para procesamiento avanzado
        img_array = np.array(image)
        
        # 1. Enhancers de contraste y saturación (DotSimulate technique)
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.2)  # Incrementar contraste
        
        enhancer = ImageEnhance.Color(image)
        image = enhancer.enhance(1.1)  # Incrementar saturación
        
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(1.1)  # Incrementar nitidez
        
        # 2. Edge enhancement para mejor definición
        try:
            edges = image.filter(ImageFilter.FIND_EDGES)
            # Blend edges con imagen original
            img_array = np.array(image)
            edges_array = np.array(edges)
            enhanced = img_array + (edges_array * 0.1).astype(np.uint8)
            enhanced = np.clip(enhanced, 0, 255)
            image = Image.fromarray(enhanced)
        except Exception:
            pass  # Fallback a imagen original
        
        # 3. Noise injection para crear texturas ricas (DotSimulate style)
        img_array = np.array(image).astype(np.float32)
        
        # Generar ruido multi-frecuencia
        noise_low = np.random.normal(0, 2, img_array.shape)
        noise_high = np.random.normal(0, 1, img_array.shape)
        
        # Aplicar ruido con diferentes intensidades por canal
        img_array[:,:,0] += noise_low[:,:,0] * 0.8  # Rojo
        img_array[:,:,1] += noise_low[:,:,1] * 0.6  # Verde
        img_array[:,:,2] += noise_high[:,:,2] * 0.4  # Azul
        
        # Normalizar y convertir back
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)
        image = Image.fromarray(img_array)
        
        # 4. Aplicar filtro de densidad variable
        # Crear máscara de densidad basada en luminancia
        gray = image.convert('L')
        gray_array = np.array(gray)
        density_mask = (gray_array > 128).astype(np.float32)
        
        # Aplicar variaciones de densidad
        img_array = np.array(image).astype(np.float32)
        for c in range(3):
            img_array[:,:,c] *= (0.85 + density_mask * 0.3)
        
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)
        image = Image.fromarray(img_array)
        
        self.preprocessing_time = (time.time() - start_time) * 1000
        return image
    
    def enhance_prompt_dotsimulate_style(self, base_prompt):
        """
        Prompt engineering siguiendo metodología DotSimulate
        Estructura: Subject + Medium + Style + Lighting + Quality
        """
        if not base_prompt.strip():
            base_prompt = "professional portrait"
        
        # Seleccionar quality boosters basado en el prompt
        quality_terms = []
        if "cyberpunk" in base_prompt.lower():
            quality_terms = ["neon lighting", "highly detailed", "8K", "dramatic lighting", "sharp focus"]
        elif "anime" in base_prompt.lower():
            quality_terms = ["studio quality", "detailed", "vibrant colors", "sharp focus", "masterpiece"]
        elif "oil painting" in base_prompt.lower():
            quality_terms = ["classical technique", "highly detailed", "masterpiece", "museum quality", "professional"]
        else:
            quality_terms = self.quality_boosters[:4]  # Default selection
        
        # Estructura DotSimulate: prompt base + técnica + calidad
        enhanced_prompt = f"{base_prompt}, {', '.join(quality_terms)}"
        
        return enhanced_prompt
    
    def calculate_image_similarity(self, image):
        """Similarity filter mejorado con características estocásticas"""
        try:
            # Reducir imagen para hash rápido
            small_img = image.resize((32, 32))
            img_array = np.array(small_img)
            
            # Hash perceptual más sofisticado
            return hashlib.md5(img_array.tobytes()).hexdigest()
        except Exception:
            return str(time.time())
    
    def should_skip_frame_stochastic(self, image):
        """
        Stochastic Similarity Filter (DotSimulate technique)
        Reduce procesamiento cuando los cambios frame-to-frame son mínimos
        """
        if not self.stochastic_filter_enabled:
            return False
            
        try:
            current_hash = self.calculate_image_similarity(image)
            
            if self.last_image_hash is None:
                self.last_image_hash = current_hash
                return False
            
            # Comparación estocástica con threshold adaptivo
            similarity_score = 1.0 if current_hash == self.last_image_hash else 0.0
            
            # Agregar componente estocástico
            random_factor = np.random.uniform(0.0, 0.1)
            effective_threshold = self.similarity_threshold + random_factor
            
            if similarity_score > effective_threshold:
                self.frame_skip_counter += 1
                return True
            
            self.last_image_hash = current_hash
            self.frame_skip_counter = 0
            return False
        except Exception:
            return False
    
    def apply_temporal_feedback(self, generated_image):
        """
        Sistema de feedback temporal para reducir "saltiness"
        Técnica inspirada en DotSimulate TouchDesigner workflows
        """
        if len(self.frame_buffer) == 0:
            self.frame_buffer.append(generated_image)
            return generated_image
        
        try:
            # Convertir imágenes a arrays
            current_array = np.array(generated_image).astype(np.float32)
            previous_array = np.array(self.frame_buffer[-1]).astype(np.float32)
            
            # Aplicar smoothing temporal
            smoothed = (current_array * (1 - self.temporal_smoothing) + 
                       previous_array * self.temporal_smoothing)
            
            smoothed = np.clip(smoothed, 0, 255).astype(np.uint8)
            smoothed_image = Image.fromarray(smoothed)
            
            self.frame_buffer.append(smoothed_image)
            return smoothed_image
            
        except Exception as e:
            print(f"Temporal feedback error: {e}")
            self.frame_buffer.append(generated_image)
            return generated_image
    
    def calculate_optimal_steps_dotsimulate(self, strength, guidance_scale):
        """
        Cálculo de pasos optimizado según metodología DotSimulate
        Timesteps adaptativos: [32, 45] para img2img estándar
        """
        # DotSimulate usa configuraciones específicas
        if strength >= 0.8 and guidance_scale >= 7.0:
            return 6  # Máxima calidad para transformaciones intensas
        elif strength >= 0.6:
            return 4  # Balance calidad/velocidad
        else:
            return 2  # Velocidad para cambios sutiles
    
    def start_processing_thread(self):
        """Iniciar thread de procesamiento continuo"""
        self.processing = True
        processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        processing_thread.start()
        print("✓ Enhanced processing thread iniciado")
    
    def _processing_loop(self):
        """Loop principal con optimizaciones DotSimulate"""
        while self.processing:
            try:
                frame_data = self.frame_queue.get(timeout=0.1)
                
                start_time = time.time()
                
                # Stochastic similarity filter
                if self.should_skip_frame_stochastic(frame_data['image']):
                    self.skipped_frames += 1
                    try:
                        last_result = self.result_queue.queue[-1] if self.result_queue.queue else None
                        if last_result:
                            self.result_queue.put_nowait(last_result)
                    except:
                        pass
                    continue
                
                # Preprocessing DotSimulate-style
                processed_image = self.preprocess_image_dotsimulate_style(frame_data['image'])
                
                # Prompt enhancement
                enhanced_prompt = self.enhance_prompt_dotsimulate_style(frame_data['prompt'])
                
                # Parámetros optimizados
                strength = frame_data.get('strength', 0.75)
                guidance_scale = frame_data.get('guidance_scale', 8.0)  # DotSimulate range: 7-12
                steps = self.calculate_optimal_steps_dotsimulate(strength, guidance_scale)
                
                print(f"Procesando: prompt='{enhanced_prompt[:50]}...', strength={strength}, guidance={guidance_scale}, steps={steps}")
                
                # Generación con configuración DotSimulate
                try:
                    with torch.no_grad():
                        with torch.cuda.amp.autocast():
                            result = self.pipe(
                                prompt=enhanced_prompt,
                                image=processed_image,
                                num_inference_steps=steps,
                                strength=strength,
                                guidance_scale=guidance_scale,
                                generator=torch.Generator(device=self.device).manual_seed(42)
                            ).images[0]
                    
                    # Aplicar temporal feedback
                    result = self.apply_temporal_feedback(result)
                    
                    processing_time = (time.time() - start_time) * 1000
                    self.total_frames += 1
                    
                    result_data = {
                        'image': result,
                        'processing_time': processing_time,
                        'preprocessing_time': self.preprocessing_time,
                        'timestamp': frame_data['timestamp'],
                        'enhanced_prompt': enhanced_prompt,
                        'stats': {
                            'total_frames': self.total_frames,
                            'skipped_frames': self.skipped_frames,
                            'skip_rate': self.skipped_frames / max(1, self.total_frames) * 100,
                            'strength': strength,
                            'steps': steps,
                            'guidance_scale': guidance_scale,
                            'temporal_smoothing': self.temporal_smoothing
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
                            
                    print(f"Frame procesado: {processing_time:.1f}ms (Preprocessing: {self.preprocessing_time:.1f}ms)")
                    
                except Exception as process_error:
                    print(f"Error procesando frame: {process_error}")
                    continue
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error en processing loop: {e}")
                continue
    
    def add_frame(self, image, prompt, strength, guidance_scale, timestamp):
        """Agregar frame para procesamiento"""
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
        """Obtener resultado más reciente"""
        result = None
        while not self.result_queue.empty():
            try:
                result = self.result_queue.get_nowait()
            except queue.Empty:
                break
        return result

# Instancia global
processor = DotSimulateEnhancedDiffusion()

# HTML con interfaz avanzada DotSimulate-style
HTML_CONTENT = """
<!DOCTYPE html>
<html>
<head>
    <title>Enhanced StreamDiffusion - DotSimulate Style</title>
    <style>
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }
        
        body {
            background: linear-gradient(135deg, #0a0a0a, #1a1a2e);
            color: #fff;
            font-family: 'Consolas', 'Monaco', monospace;
            overflow: hidden;
            height: 100vh;
        }
        
        .container {
            display: flex;
            height: 100vh;
            align-items: center;
            justify-content: center;
            gap: 30px;
            padding: 20px;
        }
        
        video, canvas {
            width: 512px;
            height: 512px;
            border: 3px solid #00ff88;
            background: #111;
            border-radius: 12px;
            box-shadow: 0 0 30px rgba(0, 255, 136, 0.3);
        }
        
        .output-container {
            position: relative;
        }
        
        .controls {
            position: fixed;
            bottom: 20px;
            left: 20px;
            right: 20px;
            background: rgba(10, 10, 10, 0.95);
            padding: 25px;
            border-radius: 20px;
            border: 2px solid #00ff88;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 25px;
            z-index: 1000;
            backdrop-filter: blur(10px);
        }
        
        .control-group {
            display: flex;
            flex-direction: column;
            gap: 12px;
        }
        
        .control-label {
            color: #00ff88;
            font-weight: bold;
            font-size: 14px;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 8px;
        }
        
        button {
            padding: 15px 30px;
            font-size: 16px;
            cursor: pointer;
            background: linear-gradient(45deg, #00ff88, #00cc66);
            color: #000;
            border: none;
            border-radius: 10px;
            font-weight: bold;
            transition: all 0.3s;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        button:hover {
            transform: scale(1.05);
            box-shadow: 0 0 25px #00ff88;
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
            padding: 18px 35px;
        }
        
        .totem-btn:hover {
            box-shadow: 0 0 25px #ff6b00;
        }
        
        input[type="range"] {
            width: 100%;
            height: 8px;
            -webkit-appearance: none;
            background: linear-gradient(90deg, #333, #00ff88);
            outline: none;
            border-radius: 4px;
        }
        
        input[type="range"]::-webkit-slider-thumb {
            -webkit-appearance: none;
            width: 24px;
            height: 24px;
            background: #00ff88;
            border-radius: 50%;
            cursor: pointer;
            box-shadow: 0 0 10px rgba(0, 255, 136, 0.5);
        }
        
        textarea {
            padding: 15px;
            font-size: 14px;
            border-radius: 10px;
            background: #1a1a1a;
            color: #fff;
            border: 2px solid #00ff88;
            font-family: 'Consolas', monospace;
            resize: vertical;
            min-height: 100px;
            max-height: 150px;
        }
        
        .range-value {
            color: #00ff88;
            font-weight: bold;
            text-align: center;
            font-size: 18px;
            margin-top: 8px;
            text-shadow: 0 0 10px #00ff88;
        }
        
        .stats {
            position: fixed;
            top: 20px;
            right: 20px;
            background: rgba(10, 10, 10, 0.95);
            padding: 25px;
            border-radius: 20px;
            font-family: 'Consolas', monospace;
            font-size: 12px;
            border: 2px solid #00ff88;
            min-width: 300px;
            z-index: 1000;
            backdrop-filter: blur(10px);
        }
        
        .stat-value {
            color: #00ff88;
            font-weight: bold;
        }
        
        .stat-row {
            display: flex;
            justify-content: space-between;
            margin: 10px 0;
            border-bottom: 1px solid #333;
            padding-bottom: 8px;
        }
        
        .status-indicator {
            position: absolute;
            top: 15px;
            left: 15px;
            width: 24px;
            height: 24px;
            border-radius: 50%;
            background: #ff4444;
            transition: all 0.3s;
        }
        
        .status-indicator.active {
            background: #00ff88;
            box-shadow: 0 0 20px #00ff88;
        }
        
        .title {
            text-align: center;
            margin-bottom: 20px;
            color: #00ff88;
            font-size: 16px;
            font-weight: bold;
            text-transform: uppercase;
            letter-spacing: 2px;
        }
        
        /* Modo Totem */
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
            box-shadow: 0 0 50px rgba(0, 255, 136, 0.5);
        }
        
        .totem-exit {
            position: absolute;
            top: 30px;
            right: 30px;
            background: rgba(255, 0, 0, 0.8);
            color: white;
            border: none;
            border-radius: 50%;
            width: 70px;
            height: 70px;
            font-size: 28px;
            cursor: pointer;
            z-index: 10000;
        }
        
        .totem-info {
            position: absolute;
            bottom: 30px;
            left: 50%;
            transform: translateX(-50%);
            background: rgba(0,0,0,0.9);
            padding: 20px 40px;
            border-radius: 15px;
            border: 2px solid #00ff88;
            text-align: center;
        }
        
        .enhanced-label {
            background: linear-gradient(45deg, #00ff88, #00cc66);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
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
    
    <!-- Modo Totem -->
    <div class="totem-mode" id="totemMode">
        <button class="totem-exit" onclick="exitTotemMode()">✕</button>
        <canvas id="totemCanvas" class="totem-canvas"></canvas>
        <div class="totem-info">
            <div style="color: #00ff88; font-size: 20px; font-weight: bold;">ENHANCED DOTSIMULATE MODE</div>
            <div style="margin-top: 10px;">Real-time AI transformation with advanced preprocessing</div>
        </div>
    </div>
    
    <div class="stats">
        <div class="title">Enhanced Metrics</div>
        <div class="stat-row">
            <span>FPS:</span>
            <span id="fps" class="stat-value">0</span>
        </div>
        <div class="stat-row">
            <span>Processing:</span>
            <span id="latency" class="stat-value">0</span>ms
        </div>
        <div class="stat-row">
            <span>Preprocessing:</span>
            <span id="preprocessing" class="stat-value">0</span>ms
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
            <span id="currentStrength" class="stat-value">0.75</span>
        </div>
        <div class="stat-row">
            <span>Guidance:</span>
            <span id="currentGuidance" class="stat-value">8.0</span>
        </div>
        <div class="stat-row">
            <span>Temporal Smoothing:</span>
            <span id="temporalSmoothing" class="stat-value">15%</span>
        </div>
    </div>
    
    <div class="controls">
        <div class="control-group">
            <div class="control-label enhanced-label">System Control</div>
            <button id="startBtn" onclick="start()">🚀 Start Enhanced Stream</button>
            <button id="stopBtn" onclick="stop()" disabled>⏹ Stop</button>
            <button class="totem-btn" onclick="enterTotemMode()" id="totemBtn" disabled>📺 Totem Mode</button>
        </div>
        
        <div class="control-group">
            <div class="control-label enhanced-label">Enhanced Prompt Engineering</div>
            <textarea id="customPrompt" placeholder="Advanced prompt (will be automatically enhanced with quality boosters)">cyberpunk style, neon lights, futuristic cityscape</textarea>
        </div>
        
        <div class="control-group">
            <div class="control-label enhanced-label">Transformation Intensity</div>
            <input type="range" id="strengthSlider" min="0.50" max="0.95" step="0.01" value="0.75" oninput="updateStrengthValue()">
            <div class="range-value" id="strengthValue">0.75</div>
        </div>
        
        <div class="control-group">
            <div class="control-label enhanced-label">CFG Scale (DotSimulate Range: 7-15)</div>
            <input type="range" id="guidanceSlider" min="7.0" max="15.0" step="0.5" value="8.0" oninput="updateGuidanceValue()">
            <div class="range-value" id="guidanceValue">8.0</div>
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
        let preprocessingTimes = [];
        
        function updateStrengthValue() {
            const value = parseFloat(document.getElementById('strengthSlider').value);
            document.getElementById('strengthValue').textContent = value.toFixed(2);
        }
        
        function updateGuidanceValue() {
            const value = parseFloat(document.getElementById('guidanceSlider').value);
            document.getElementById('guidanceValue').textContent = value.toFixed(1);
        }
        
        function enterTotemMode() {
            if (!streaming) {
                alert('Start the stream before activating totem mode');
                return;
            }
            
            totemMode = true;
            document.getElementById('totemMode').style.display = 'flex';
            
            totemCanvas = document.getElementById('totemCanvas');
            totemCtx = totemCanvas.getContext('2d');
            
            const screenHeight = window.innerHeight;
            const screenWidth = window.innerWidth;
            
            if (screenHeight > screenWidth) {
                totemCanvas.width = screenWidth * 0.8;
                totemCanvas.height = screenWidth * 0.8;
            } else {
                totemCanvas.width = screenHeight * 0.7;
                totemCanvas.height = screenHeight * 0.7;
            }
            
            document.querySelector('.controls').style.display = 'none';
            document.querySelector('.stats').style.display = 'none';
            
            console.log('Enhanced Totem Mode activated');
        }
        
        function exitTotemMode() {
            totemMode = false;
            document.getElementById('totemMode').style.display = 'none';
            document.querySelector('.controls').style.display = 'grid';
            document.querySelector('.stats').style.display = 'block';
            console.log('Totem Mode deactivated');
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
                    console.log('Connected to Enhanced StreamDiffusion server');
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
                            ctx.clearRect(0, 0, 512, 512);
                            ctx.drawImage(img, 0, 0, 512, 512);
                            
                            if (totemMode && totemCtx) {
                                totemCtx.clearRect(0, 0, totemCanvas.width, totemCanvas.height);
                                totemCtx.drawImage(img, 0, 0, totemCanvas.width, totemCanvas.height);
                            }
                        };
                        img.src = data.image;
                        updateAdvancedStats(data);
                    }
                };
                
                ws.onerror = (error) => {
                    console.error('WebSocket error:', error);
                    stop();
                };
                
                ws.onclose = () => {
                    console.log('Disconnected from server');
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
            
            // DotSimulate-style image fitting
            const videoAspect = video.videoWidth / video.videoHeight;
            const targetAspect = 1;
            
            let sx, sy, sw, sh;
            
            if (videoAspect > targetAspect) {
                sh = video.videoHeight;
                sw = video.videoHeight;
                sx = (video.videoWidth - sw) / 2;
                sy = 0;
            } else {
                sw = video.videoWidth;
                sh = video.videoWidth;
                sx = 0;
                sy = (video.videoHeight - sh) / 2;
            }
            
            tempCtx.drawImage(video, sx, sy, sw, sh, 0, 0, 512, 512);
            
            const prompt = document.getElementById('customPrompt').value || "professional photography, highly detailed";
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
        
        function updateAdvancedStats(data) {
            frameCount++;
            totalFrames++;
            
            if (data.processing_time) {
                latencies.push(data.processing_time);
                if (latencies.length > 50) latencies.shift();
            }
            
            if (data.preprocessing_time) {
                preprocessingTimes.push(data.preprocessing_time);
                if (preprocessingTimes.length > 50) preprocessingTimes.shift();
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
                
                if (preprocessingTimes.length > 0) {
                    const avgPreprocessing = preprocessingTimes.reduce((a, b) => a + b) / preprocessingTimes.length;
                    document.getElementById('preprocessing').textContent = Math.round(avgPreprocessing);
                }
            }
            
            if (data.stats) {
                document.getElementById('totalFrames').textContent = data.stats.total_frames;
                document.getElementById('skipRate').textContent = data.stats.skip_rate.toFixed(1);
                document.getElementById('currentStrength').textContent = data.stats.strength;
                document.getElementById('currentGuidance').textContent = data.stats.guidance_scale;
                document.getElementById('temporalSmoothing').textContent = 
                    Math.round(data.stats.temporal_smoothing * 100) + '%';
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
    print("🚀 Cliente conectado - Enhanced DotSimulate Mode")
    
    try:
        while True:
            data = await websocket.receive_json()
            
            if 'image' not in data:
                continue
            
            try:
                img_data = base64.b64decode(data['image'].split(',')[1])
                input_image = Image.open(io.BytesIO(img_data)).convert("RGB")
                
                if input_image.size != (512, 512):
                    input_image = input_image.resize((512, 512), Image.Resampling.LANCZOS)
                
            except Exception as e:
                print(f"❌ Error decodificando imagen: {e}")
                continue
            
            prompt = data.get('prompt', 'professional photography, highly detailed')
            strength = data.get('strength', 0.75)
            guidance_scale = data.get('guidance_scale', 8.0)
            
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
                    'preprocessing_time': result.get('preprocessing_time', 0),
                    'timestamp': result['timestamp'],
                    'enhanced_prompt': result.get('enhanced_prompt', ''),
                    'stats': result['stats']
                })
            
    except Exception as e:
        print(f"❌ Error en WebSocket: {e}")
    finally:
        print("🔌 Cliente desconectado")

if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*80)
    print("🎨 ENHANCED STREAMDIFFUSION - DOTSIMULATE METHODOLOGY")
    print("="*80)
    print("⚡ LCM Dreamshaper v7 con preprocessing avanzado")
    print("🎛️ CFG Scale optimizado (7-15 range)")
    print("🖼️ Preprocessing multi-texturizado")
    print("⏱️ Sistema de feedback temporal")
    print("🎯 Prompt engineering estructurado")
    print("📺 Modo Totem con visualización premium")
    print("🌐 URL: http://0.0.0.0:8000")
    print("="*80 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
