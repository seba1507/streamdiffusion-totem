from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import torch
from diffusers import AutoPipelineForImage2Image
import base64
from PIL import Image
import io
import asyncio
import time
from collections import deque
import threading
import queue
import os

# Configurar variables de entorno para debugging CUDA
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class TurboProcessor:
    def __init__(self):
        self.pipe = None
        self.processing_queue = queue.Queue(maxsize=2)
        self.output_queue = queue.Queue(maxsize=5)
        self.is_running = False
        self.init_model()
        
    def init_model(self):
        print("🚀 Cargando SDXL-Turbo...")
        
        try:
            # Cargar SDXL-Turbo específicamente para img2img
            self.pipe = AutoPipelineForImage2Image.from_pretrained(
                "stabilityai/sdxl-turbo",
                torch_dtype=torch.float16,
                variant="fp16",
                use_safetensors=True,
                low_cpu_mem_usage=True
            )
            
            # Mover a GPU
            self.pipe = self.pipe.to("cuda")
            
            # Configuraciones de seguridad
            self.pipe.set_progress_bar_config(disable=True)
            
            # XFormers si está disponible
            try:
                self.pipe.enable_xformers_memory_efficient_attention()
                print("✅ XFormers habilitado")
            except:
                print("⚠️ XFormers no disponible")
            
            # Optimizaciones de VAE
            self.pipe.vae.enable_slicing()
            self.pipe.vae.enable_tiling()
            
            # Pre-calentar con imagen correcta
            print("🔥 Pre-calentando modelo...")
            dummy_image = Image.new('RGB', (512, 512), color='black')
            with torch.no_grad():
                _ = self.pipe(
                    prompt="test",
                    image=dummy_image,
                    num_inference_steps=1,
                    strength=0.3,
                    guidance_scale=0.0,
                    output_type="pil"
                ).images[0]
            
            print("✅ Modelo SDXL-Turbo listo!")
            
        except Exception as e:
            print(f"❌ Error cargando modelo: {e}")
            raise
        
    def start_processing(self):
        self.is_running = True
        threading.Thread(target=self._process_loop, daemon=True).start()
        
    def _process_loop(self):
        """Loop de procesamiento continuo"""
        while self.is_running:
            try:
                frame_data = self.processing_queue.get(timeout=0.1)
                
                try:
                    # Procesar con manejo de errores
                    with torch.no_grad():
                        with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
                            # Llamar al pipeline correctamente
                            output = self.pipe(
                                prompt=frame_data['prompt'],
                                image=frame_data['image'],
                                num_inference_steps=2,  # 2 pasos para mejor calidad
                                strength=0.4,
                                guidance_scale=0.0,
                                width=512,
                                height=512,
                                output_type="pil",
                                return_dict=True
                            )
                            
                            # Acceder a la imagen correctamente
                            if hasattr(output, 'images') and len(output.images) > 0:
                                result_image = output.images[0]
                            else:
                                print("⚠️ No se generó imagen")
                                continue
                    
                    # Comprimir resultado
                    buffered = io.BytesIO()
                    result_image.save(buffered, format="JPEG", quality=80, optimize=False)
                    img_str = base64.b64encode(buffered.getvalue()).decode()
                    
                    # Agregar a cola de salida
                    self.output_queue.put({
                        'image': f'data:image/jpeg;base64,{img_str}',
                        'timestamp': frame_data.get('timestamp', time.time())
                    })
                    
                except torch.cuda.OutOfMemoryError:
                    print("⚠️ GPU sin memoria, limpiando...")
                    torch.cuda.empty_cache()
                except Exception as e:
                    print(f"Error procesando frame: {e}")
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error en loop: {e}")
    
    def add_frame(self, image, prompt, timestamp=None):
        """Agregar frame para procesar"""
        try:
            # Limpiar cola si está llena
            if self.processing_queue.full():
                try:
                    self.processing_queue.get_nowait()
                except:
                    pass
            
            self.processing_queue.put_nowait({
                'image': image,
                'prompt': prompt,
                'timestamp': timestamp or time.time()
            })
            return True
        except:
            return False
    
    def get_latest_frame(self):
        """Obtener el frame más reciente"""
        result = None
        while not self.output_queue.empty():
            try:
                result = self.output_queue.get_nowait()
            except:
                break
        return result

# HTML mejorado
HTML_CONTENT = """
<!DOCTYPE html>
<html>
<head>
    <title>SDXL-Turbo Stream</title>
    <style>
        body { 
            margin: 0; 
            background: #000; 
            overflow: hidden; 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
        }
        video, canvas { 
            position: absolute; 
            width: 100%; 
            height: 100%; 
            object-fit: contain; 
        }
        #hiddenVideo { display: none; }
        
        .controls {
            position: fixed;
            bottom: 20px;
            left: 50%;
            transform: translateX(-50%);
            background: rgba(0,0,0,0.9);
            padding: 20px;
            border-radius: 15px;
            backdrop-filter: blur(10px);
            z-index: 100;
            box-shadow: 0 4px 20px rgba(0,0,0,0.5);
        }
        
        button {
            padding: 12px 24px;
            margin: 0 5px;
            font-size: 16px;
            cursor: pointer;
            background: #2196F3;
            color: white;
            border: none;
            border-radius: 8px;
            transition: all 0.3s;
        }
        
        button:hover {
            background: #1976D2;
            transform: translateY(-2px);
        }
        
        button:disabled {
            background: #666;
            cursor: not-allowed;
        }
        
        select {
            padding: 10px;
            font-size: 16px;
            border-radius: 8px;
            margin: 0 5px;
        }
        
        .fps-counter {
            position: fixed;
            top: 20px;
            right: 20px;
            background: rgba(0,0,0,0.9);
            color: #0f0;
            padding: 15px;
            font-family: 'Courier New', monospace;
            font-size: 14px;
            border-radius: 10px;
            backdrop-filter: blur(10px);
        }
        
        .status {
            position: fixed;
            top: 20px;
            left: 20px;
            background: rgba(0,0,0,0.9);
            color: white;
            padding: 15px;
            border-radius: 10px;
            backdrop-filter: blur(10px);
        }
        
        .status.connected { border-left: 4px solid #4CAF50; }
        .status.disconnected { border-left: 4px solid #f44336; }
        
        .loading {
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            color: white;
            font-size: 24px;
            display: none;
        }
    </style>
</head>
<body>
    <video id="hiddenVideo" autoplay playsinline muted></video>
    <canvas id="outputCanvas"></canvas>
    
    <div class="status disconnected" id="status">
        <div>Estado: <span id="statusText">Desconectado</span></div>
    </div>
    
    <div class="fps-counter">
        <div>Send FPS: <span id="sendFps">0</span></div>
        <div>Recv FPS: <span id="recvFps">0</span></div>
        <div>Latency: <span id="latency">0</span>ms</div>
        <div>Queue: <span id="queue">0</span></div>
    </div>
    
    <div class="controls">
        <button onclick="start()" id="startBtn">▶️ Iniciar Stream</button>
        <button onclick="stop()" id="stopBtn" disabled>⏹️ Detener</button>
        
        <select id="style">
            <option value="cyberpunk neon lights">Cyberpunk</option>
            <option value="anime style colorful">Anime</option>
            <option value="oil painting artistic">Óleo</option>
            <option value="watercolor soft">Acuarela</option>
            <option value="pixel art retro">Pixel Art</option>
        </select>
        
        <label style="color: white; margin-left: 10px;">
            FPS: <input type="range" id="fps" min="1" max="10" value="5" style="vertical-align: middle;">
            <span id="fpsValue">5</span>
        </label>
    </div>
    
    <div class="loading" id="loading">Procesando...</div>
    
    <script>
        let ws = null;
        let streaming = false;
        const video = document.getElementById('hiddenVideo');
        const canvas = document.getElementById('outputCanvas');
        const ctx = canvas.getContext('2d');
        const status = document.getElementById('status');
        const statusText = document.getElementById('statusText');
        const fpsSlider = document.getElementById('fps');
        const fpsValue = document.getElementById('fpsValue');
        
        // Estadísticas
        let sendCount = 0, recvCount = 0;
        let lastSendTime = Date.now(), lastRecvTime = Date.now();
        let timestamps = new Map();
        
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
        
        fpsSlider.oninput = () => {
            fpsValue.textContent = fpsSlider.value;
        };
        
        async function start() {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({
                    video: { 
                        width: { ideal: 1280 },
                        height: { ideal: 720 }
                    }
                });
                video.srcObject = stream;
                
                const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                ws = new WebSocket(`${protocol}//${window.location.host}/ws`);
                
                ws.onopen = () => {
                    status.className = 'status connected';
                    statusText.textContent = 'Conectado';
                    document.getElementById('startBtn').disabled = true;
                    document.getElementById('stopBtn').disabled = false;
                    streaming = true;
                    streamFrames();
                };
                
                ws.onmessage = (event) => {
                    const data = JSON.parse(event.data);
                    if (data.image && !data.original) {
                        const img = new Image();
                        img.onload = () => {
                            ctx.clearRect(0, 0, canvas.width, canvas.height);
                            const scale = Math.min(canvas.width/512, canvas.height/512);
                            const x = (canvas.width - 512 * scale) / 2;
                            const y = (canvas.height - 512 * scale) / 2;
                            ctx.drawImage(img, x, y, 512 * scale, 512 * scale);
                            
                            recvCount++;
                            if (data.timestamp) {
                                const latency = Date.now() - data.timestamp;
                                document.getElementById('latency').textContent = Math.round(latency);
                            }
                        };
                        img.src = data.image;
                    }
                };
                
                ws.onerror = (error) => {
                    console.error('WebSocket error:', error);
                    statusText.textContent = 'Error de conexión';
                };
                
                ws.onclose = () => {
                    status.className = 'status disconnected';
                    statusText.textContent = 'Desconectado';
                    streaming = false;
                    document.getElementById('startBtn').disabled = false;
                    document.getElementById('stopBtn').disabled = true;
                };
                
                // Actualizar estadísticas
                setInterval(updateStats, 1000);
                
            } catch (error) {
                alert('Error: ' + error.message);
            }
        }
        
        function streamFrames() {
            if (!streaming || !ws || ws.readyState !== WebSocket.OPEN) return;
            
            const captureCanvas = document.createElement('canvas');
            captureCanvas.width = 512;
            captureCanvas.height = 512;
            const captureCtx = captureCanvas.getContext('2d');
            
            captureCtx.drawImage(video, 0, 0, 512, 512);
            const imageData = captureCanvas.toDataURL('image/jpeg', 0.8);
            
            ws.send(JSON.stringify({
                type: 'frame',
                image: imageData,
                prompt: document.getElementById('style').value,
                timestamp: Date.now()
            }));
            
            sendCount++;
            
            const delay = 1000 / parseInt(fpsSlider.value);
            setTimeout(streamFrames, delay);
        }
        
        function updateStats() {
            const now = Date.now();
            const sendFps = sendCount;
            const recvFps = recvCount;
            
            document.getElementById('sendFps').textContent = sendFps;
            document.getElementById('recvFps').textContent = recvFps;
            
            sendCount = 0;
            recvCount = 0;
        }
        
        function stop() {
            streaming = false;
            if (ws) ws.close();
            if (video.srcObject) {
                video.srcObject.getTracks().forEach(track => track.stop());
            }
        }
        
        // Redimensionar canvas cuando cambie la ventana
        window.addEventListener('resize', () => {
            canvas.width = window.innerWidth;
            canvas.height = window.innerHeight;
        });
    </script>
</body>
</html>
"""

# Instancia global
processor = None

@app.on_event("startup")
async def startup_event():
    global processor
    processor = TurboProcessor()
    processor.start_processing()

@app.get("/")
async def get():
    return HTMLResponse(content=HTML_CONTENT)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("🎮 Cliente conectado")
    
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            data = await websocket.receive_json()
            
            if data.get('type') == 'frame':
                # Decodificar imagen
                img_data = base64.b64decode(data['image'].split(',')[1])
                image = Image.open(io.BytesIO(img_data)).convert("RGB")
                
                # Asegurar tamaño correcto
                if image.size != (512, 512):
                    image = image.resize((512, 512), Image.Resampling.LANCZOS)
                
                # Prompt
                style = data.get('prompt', 'cyberpunk')
                prompt = f"{style}, high quality"
                
                # Agregar a procesamiento
                timestamp = data.get('timestamp', time.time())
                processor.add_frame(image, prompt, timestamp)
                
                # Obtener frame procesado si hay
                processed = processor.get_latest_frame()
                
                if processed:
                    await websocket.send_json(processed)
                    frame_count += 1
                    
                    # Log FPS cada 2 segundos
                    elapsed = time.time() - start_time
                    if elapsed >= 2.0:
                        fps = frame_count / elapsed
                        print(f"📊 FPS real: {fps:.1f}")
                        frame_count = 0
                        start_time = time.time()
                else:
                    # Enviar frame original si no hay procesado
                    await websocket.send_json({
                        'type': 'frame',
                        'image': data['image'],
                        'timestamp': timestamp,
                        'original': True
                    })
            
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        print("👋 Cliente desconectado")

if __name__ == "__main__":
    import uvicorn
    
    print("🎯 SDXL-Turbo Stream Server")
    print("📍 http://0.0.0.0:8000")
    print("🚀 Optimizado para máximo FPS")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="warning",
        access_log=False
    )
