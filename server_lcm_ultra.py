from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import torch
from diffusers import DiffusionPipeline, LCMScheduler
import base64
from PIL import Image
import io
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import asyncio
import os

# Configuración para evitar errores de CUDA
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class UltraLCMProcessor:
    def __init__(self):
        self.pipe = None
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.init_model()
        
    def init_model(self):
        print("🚀 Inicializando LCM Ultra...")
        
        # Modelo LCM probado y estable
        self.pipe = DiffusionPipeline.from_pretrained(
            "SimianLuo/LCM_Dreamshaper_v7",
            torch_dtype=torch.float16,
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False,
            custom_pipeline="latent_consistency_txt2img",
            custom_revision="main",
        )
        
        self.pipe.to("cuda")
        
        # Configurar scheduler LCM
        self.pipe.scheduler = LCMScheduler.from_config(self.pipe.scheduler.config)
        
        # Optimizaciones
        self.pipe.set_progress_bar_config(disable=True)
        
        # XFormers si está disponible
        try:
            self.pipe.enable_xformers_memory_efficient_attention()
            print("✅ XFormers habilitado")
        except:
            print("⚠️ Usando atención estándar")
        
        # Pre-calentar
        print("🔥 Pre-calentando modelo...")
        with torch.no_grad():
            self.pipe(
                prompt="test",
                num_inference_steps=4,
                guidance_scale=8.0,
                lcm_origin_steps=50,
                output_type="pil",
                height=512,
                width=512
            ).images[0]
        
        print("✅ LCM Ultra listo!")
        
    def process_image_fast(self, prompt, seed=None):
        """Procesamiento ultra rápido"""
        if seed is None:
            seed = int(time.time() * 1000) % 100000
            
        generator = torch.Generator(device="cuda").manual_seed(seed)
        
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=torch.float16):
                result = self.pipe(
                    prompt=prompt,
                    num_inference_steps=4,  # Mínimo para LCM
                    guidance_scale=8.0,     # Óptimo para LCM
                    lcm_origin_steps=50,
                    generator=generator,
                    output_type="pil",
                    height=512,
                    width=512
                ).images[0]
                
        return result

# HTML optimizado
HTML_CONTENT = """
<!DOCTYPE html>
<html>
<head>
    <title>LCM Ultra Stream</title>
    <meta charset="UTF-8">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body { 
            background: #000; 
            color: #fff; 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
            overflow: hidden;
        }
        
        #container {
            display: flex;
            height: 100vh;
            align-items: center;
            justify-content: center;
            gap: 20px;
        }
        
        .video-container {
            position: relative;
            background: #111;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 4px 20px rgba(0,0,0,0.5);
        }
        
        video, canvas {
            width: 512px;
            height: 512px;
            display: block;
        }
        
        .label {
            position: absolute;
            top: 10px;
            left: 10px;
            background: rgba(0,0,0,0.7);
            padding: 5px 10px;
            border-radius: 5px;
            font-size: 14px;
        }
        
        #controls {
            position: fixed;
            bottom: 30px;
            left: 50%;
            transform: translateX(-50%);
            background: rgba(20,20,20,0.95);
            padding: 20px 30px;
            border-radius: 15px;
            backdrop-filter: blur(10px);
            display: flex;
            gap: 15px;
            align-items: center;
        }
        
        button {
            padding: 10px 20px;
            background: #2196F3;
            color: white;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-size: 16px;
            transition: all 0.3s;
        }
        
        button:hover:not(:disabled) {
            background: #1976D2;
            transform: translateY(-2px);
        }
        
        button:disabled {
            background: #555;
            cursor: not-allowed;
        }
        
        select {
            padding: 10px;
            border-radius: 8px;
            background: #333;
            color: white;
            border: 1px solid #555;
        }
        
        #stats {
            position: fixed;
            top: 20px;
            right: 20px;
            background: rgba(0,0,0,0.8);
            padding: 15px;
            border-radius: 10px;
            font-family: 'Monaco', 'Consolas', monospace;
            font-size: 14px;
            line-height: 1.5;
        }
        
        .stat-item {
            display: flex;
            justify-content: space-between;
            gap: 20px;
        }
        
        .stat-value {
            color: #4CAF50;
            font-weight: bold;
        }
        
        #status {
            position: fixed;
            top: 20px;
            left: 20px;
            padding: 10px 20px;
            background: rgba(0,0,0,0.8);
            border-radius: 10px;
            border-left: 4px solid #2196F3;
        }
        
        .error { border-left-color: #f44336 !important; }
        .success { border-left-color: #4CAF50 !important; }
        
        /* Modo Totem */
        .totem-mode #container {
            gap: 0;
        }
        
        .totem-mode .video-container:first-child {
            display: none;
        }
        
        .totem-mode canvas {
            width: 100vw;
            height: 100vh;
            object-fit: contain;
        }
    </style>
</head>
<body>
    <div id="status">Listo para iniciar</div>
    
    <div id="stats">
        <div class="stat-item">
            <span>FPS (envío):</span>
            <span class="stat-value" id="sendFps">0</span>
        </div>
        <div class="stat-item">
            <span>FPS (recepción):</span>
            <span class="stat-value" id="recvFps">0</span>
        </div>
        <div class="stat-item">
            <span>Latencia:</span>
            <span class="stat-value"><span id="latency">0</span>ms</span>
        </div>
        <div class="stat-item">
            <span>Procesados:</span>
            <span class="stat-value" id="processed">0</span>
        </div>
    </div>
    
    <div id="container">
        <div class="video-container">
            <video id="video" autoplay muted playsinline></video>
            <div class="label">📷 Cámara</div>
        </div>
        <div class="video-container">
            <canvas id="canvas"></canvas>
            <div class="label">🎨 AI Output</div>
        </div>
    </div>
    
    <div id="controls">
        <button id="startBtn" onclick="start()">▶️ Iniciar</button>
        <button id="stopBtn" onclick="stop()" disabled>⏹️ Detener</button>
        
        <select id="style">
            <option value="photorealistic">Fotorealista</option>
            <option value="cyberpunk neon">Cyberpunk</option>
            <option value="anime manga style">Anime</option>
            <option value="oil painting">Óleo</option>
            <option value="watercolor">Acuarela</option>
            <option value="3d pixar style">Pixar 3D</option>
        </select>
        
        <label style="display: flex; align-items: center; gap: 10px;">
            <span>FPS:</span>
            <input type="range" id="fps" min="1" max="10" value="5" style="width: 100px;">
            <span id="fpsValue">5</span>
        </label>
        
        <button onclick="toggleTotem()">🖥️ Modo Totem</button>
    </div>
    
    <script>
        // Estado global
        let ws = null;
        let stream = null;
        let isStreaming = false;
        let captureInterval = null;
        
        // Elementos DOM
        const video = document.getElementById('video');
        const canvas = document.getElementById('canvas');
        const ctx = canvas.getContext('2d');
        const status = document.getElementById('status');
        const startBtn = document.getElementById('startBtn');
        const stopBtn = document.getElementById('stopBtn');
        const fpsSlider = document.getElementById('fps');
        const fpsValue = document.getElementById('fpsValue');
        
        // Configuración
        canvas.width = 512;
        canvas.height = 512;
        
        // Estadísticas
        let stats = {
            sendCount: 0,
            recvCount: 0,
            processedTotal: 0,
            lastUpdate: Date.now(),
            timestamps: new Map()
        };
        
        // Actualizar valor de FPS
        fpsSlider.oninput = () => {
            fpsValue.textContent = fpsSlider.value;
            if (isStreaming) {
                stopCapture();
                startCapture();
            }
        };
        
        async function start() {
            try {
                updateStatus('Iniciando cámara...', 'info');
                
                // Obtener stream de video
                stream = await navigator.mediaDevices.getUserMedia({
                    video: {
                        width: { ideal: 1280 },
                        height: { ideal: 720 },
                        facingMode: 'user'
                    }
                });
                
                video.srcObject = stream;
                await video.play();
                
                // Conectar WebSocket
                updateStatus('Conectando...', 'info');
                const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                ws = new WebSocket(`${protocol}//${window.location.host}/ws`);
                
                ws.onopen = () => {
                    updateStatus('Conectado', 'success');
                    isStreaming = true;
                    startBtn.disabled = true;
                    stopBtn.disabled = false;
                    startCapture();
                };
                
                ws.onmessage = handleMessage;
                ws.onerror = () => updateStatus('Error de conexión', 'error');
                ws.onclose = () => {
                    updateStatus('Desconectado', 'error');
                    cleanup();
                };
                
            } catch (error) {
                updateStatus('Error: ' + error.message, 'error');
                console.error(error);
            }
        }
        
        function startCapture() {
            const fps = parseInt(fpsSlider.value);
            const interval = 1000 / fps;
            
            captureInterval = setInterval(() => {
                if (!isStreaming || ws.readyState !== WebSocket.OPEN) return;
                
                // Capturar frame
                const tempCanvas = document.createElement('canvas');
                tempCanvas.width = 512;
                tempCanvas.height = 512;
                const tempCtx = tempCanvas.getContext('2d');
                
                // Dibujar y recortar video
                const videoAspect = video.videoWidth / video.videoHeight;
                let sx, sy, sw, sh;
                
                if (videoAspect > 1) {
                    // Video más ancho
                    sw = video.videoHeight;
                    sh = video.videoHeight;
                    sx = (video.videoWidth - sw) / 2;
                    sy = 0;
                } else {
                    // Video más alto
                    sw = video.videoWidth;
                    sh = video.videoWidth;
                    sx = 0;
                    sy = (video.videoHeight - sh) / 2;
                }
                
                tempCtx.drawImage(video, sx, sy, sw, sh, 0, 0, 512, 512);
                
                // Enviar
                const imageData = tempCanvas.toDataURL('image/jpeg', 0.8);
                const timestamp = Date.now();
                
                ws.send(JSON.stringify({
                    image: imageData,
                    style: document.getElementById('style').value,
                    timestamp: timestamp
                }));
                
                stats.timestamps.set(timestamp, Date.now());
                stats.sendCount++;
                
            }, interval);
        }
        
        function stopCapture() {
            if (captureInterval) {
                clearInterval(captureInterval);
                captureInterval = null;
            }
        }
        
        function handleMessage(event) {
            const data = JSON.parse(event.data);
            
            if (data.image) {
                const img = new Image();
                img.onload = () => {
                    ctx.drawImage(img, 0, 0, 512, 512);
                    
                    stats.recvCount++;
                    stats.processedTotal++;
                    
                    // Calcular latencia
                    if (data.timestamp && stats.timestamps.has(data.timestamp)) {
                        const latency = Date.now() - stats.timestamps.get(data.timestamp);
                        document.getElementById('latency').textContent = Math.round(latency);
                        stats.timestamps.delete(data.timestamp);
                    }
                    
                    // Limpiar timestamps antiguos
                    const now = Date.now();
                    for (const [ts, time] of stats.timestamps) {
                        if (now - time > 5000) {
                            stats.timestamps.delete(ts);
                        }
                    }
                };
                img.src = data.image;
            }
        }
        
        function stop() {
            updateStatus('Deteniendo...', 'info');
            cleanup();
        }
        
        function cleanup() {
            isStreaming = false;
            stopCapture();
            
            if (ws) {
                ws.close();
                ws = null;
            }
            
            if (stream) {
                stream.getTracks().forEach(track => track.stop());
                stream = null;
            }
            
            startBtn.disabled = false;
            stopBtn.disabled = true;
        }
        
        function updateStatus(message, type = 'info') {
            status.textContent = message;
            status.className = type === 'error' ? 'error' : type === 'success' ? 'success' : '';
        }
        
        function toggleTotem() {
            document.body.classList.toggle('totem-mode');
        }
        
        // Actualizar estadísticas cada segundo
        setInterval(() => {
            const now = Date.now();
            const elapsed = (now - stats.lastUpdate) / 1000;
            
            if (elapsed >= 1) {
                document.getElementById('sendFps').textContent = Math.round(stats.sendCount / elapsed);
                document.getElementById('recvFps').textContent = Math.round(stats.recvCount / elapsed);
                document.getElementById('processed').textContent = stats.processedTotal;
                
                stats.sendCount = 0;
                stats.recvCount = 0;
                stats.lastUpdate = now;
            }
        }, 100);
        
        // Limpiar al cerrar
        window.addEventListener('beforeunload', cleanup);
    </script>
</body>
</html>
"""

# Procesador global
processor = None

@app.on_event("startup")
async def startup_event():
    global processor
    processor = UltraLCMProcessor()

@app.get("/")
async def get():
    return HTMLResponse(content=HTML_CONTENT)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("🚀 Cliente conectado - LCM Ultra Mode")
    
    # Estadísticas locales
    frame_count = 0
    start_time = time.time()
    
    async def process_frame(data):
        """Procesar frame en executor"""
        try:
            style = data.get('style', 'photorealistic')
            prompt = f"{style}, high quality, detailed, 8k"
            
            # Procesar en thread pool
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                processor.executor,
                processor.process_image_fast,
                prompt
            )
            
            if result:
                # Codificar resultado
                buffered = io.BytesIO()
                result.save(buffered, format="JPEG", quality=80, optimize=False)
                img_str = base64.b64encode(buffered.getvalue()).decode()
                
                return {
                    'image': f'data:image/jpeg;base64,{img_str}',
                    'timestamp': data.get('timestamp')
                }
        except Exception as e:
            print(f"Error procesando: {e}")
            return None
    
    try:
        while True:
            # Recibir datos
            data = await websocket.receive_json()
            
            # Procesar asincrónicamente
            result = await process_frame(data)
            
            if result:
                await websocket.send_json(result)
                frame_count += 1
                
                # Log FPS cada 2 segundos
                elapsed = time.time() - start_time
                if elapsed >= 2.0:
                    fps = frame_count / elapsed
                    print(f"📊 FPS servidor: {fps:.1f}")
                    frame_count = 0
                    start_time = time.time()
            
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        print("👋 Cliente desconectado")

if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*50)
    print("🚀 LCM ULTRA STREAM SERVER")
    print("="*50)
    print("📍 URL: http://0.0.0.0:8000")
    print("⚡ Modelo: LCM Dreamshaper v7")
    print("🎯 Optimizado para máximo FPS")
    print("💡 Tip: Usa 5-8 FPS para mejor balance")
    print("="*50 + "\n")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="warning",
        access_log=False
    )
