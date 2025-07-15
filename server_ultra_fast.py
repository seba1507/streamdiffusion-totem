from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import torch
from diffusers import AutoPipelineForImage2Image, DPMSolverMultistepScheduler
import base64
from PIL import Image
import io
import asyncio
import time
from collections import deque
import threading
import queue
import numpy as np

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
        self.processing_queue = queue.Queue(maxsize=3)
        self.output_queue = queue.Queue(maxsize=10)
        self.is_running = False
        self.init_model()
        
    def init_model(self):
        print("🚀 Cargando SDXL-Turbo...")
        
        # SDXL-Turbo - El modelo más rápido disponible
        self.pipe = AutoPipelineForImage2Image.from_pretrained(
            "stabilityai/sdxl-turbo",
            torch_dtype=torch.float16,
            variant="fp16",
            safety_checker=None,
            requires_safety_checker=False,
            use_safetensors=True
        )
        
        # Mover a GPU
        self.pipe = self.pipe.to("cuda")
        
        # Configurar scheduler para velocidad máxima
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipe.scheduler.config,
            use_karras_sigmas=False,
            algorithm_type="sde-dpmsolver++",
            solver_order=2
        )
        
        # Optimizaciones críticas sin cpu_offload
        self.pipe.set_progress_bar_config(disable=True)
        
        # XFormers para velocidad
        try:
            self.pipe.enable_xformers_memory_efficient_attention()
            print("✅ XFormers habilitado")
        except:
            print("⚠️ XFormers no disponible, usando atención estándar")
        
        # Optimizaciones de memoria
        self.pipe.vae.enable_slicing()
        self.pipe.vae.enable_tiling()
        
        # Pre-calentar el modelo
        print("🔥 Pre-calentando modelo...")
        dummy_image = Image.new('RGB', (512, 512), color='black')
        self.pipe(
            prompt="test",
            image=dummy_image,
            num_inference_steps=1,
            strength=0.5,
            guidance_scale=0.0
        )
        
        print("✅ Modelo SDXL-Turbo listo!")
        
    def start_processing(self):
        self.is_running = True
        # Múltiples threads para procesamiento paralelo
        for i in range(2):
            threading.Thread(target=self._process_loop, daemon=True, name=f"Processor-{i}").start()
        
    def _process_loop(self):
        """Loop de procesamiento continuo"""
        while self.is_running:
            try:
                # Obtener frame con timeout corto
                frame_data = self.processing_queue.get(timeout=0.05)
                
                # Procesar con configuración ultra rápida
                with torch.inference_mode():
                    with torch.cuda.amp.autocast(dtype=torch.float16):
                        # SDXL-Turbo solo necesita 1-2 pasos!
                        result = self.pipe(
                            prompt=frame_data['prompt'],
                            image=frame_data['image'],
                            num_inference_steps=1,  # Ultra rápido
                            strength=0.3,  # Menos transformación = más rápido
                            guidance_scale=0.0,  # Sin guidance = más velocidad
                            width=512,
                            height=512,
                            output_type="pil"
                        ).images[0]
                
                # Comprimir resultado
                buffered = io.BytesIO()
                result.save(buffered, format="JPEG", quality=75, optimize=True)
                img_str = base64.b64encode(buffered.getvalue()).decode()
                
                # Agregar a cola de salida
                self.output_queue.put({
                    'image': f'data:image/jpeg;base64,{img_str}',
                    'timestamp': frame_data.get('timestamp', time.time())
                })
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error en procesamiento: {e}")
    
    def add_frame(self, image, prompt, timestamp=None):
        """Agregar frame para procesar"""
        try:
            # Si la cola está llena, descartar el más antiguo
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
        """Obtener frames procesados"""
        frames = []
        # Obtener todos los frames disponibles
        while not self.output_queue.empty() and len(frames) < 3:
            try:
                frames.append(self.output_queue.get_nowait())
            except:
                break
        return frames

# Instancia global del procesador
processor = UltraFastProcessor()
processor.start_processing()

@app.get("/")
async def get():
    try:
        return HTMLResponse(content=open("index_fast.html", "r").read())
    except:
        return HTMLResponse(content="""
<!DOCTYPE html>
<html>
<head>
    <title>SDXL-Turbo Stream</title>
    <style>
        body { margin: 0; background: #000; overflow: hidden; font-family: Arial; }
        video, canvas { position: absolute; width: 100%; height: 100%; object-fit: contain; }
        #hiddenVideo { display: none; }
        .controls { position: fixed; bottom: 20px; left: 50%; transform: translateX(-50%); 
                   background: rgba(0,0,0,0.8); padding: 20px; border-radius: 10px; z-index: 100; }
        button { padding: 10px 20px; margin: 0 5px; font-size: 16px; cursor: pointer; }
        .fps-counter { position: fixed; top: 20px; right: 20px; color: #0f0; 
                      background: rgba(0,0,0,0.8); padding: 10px; font-family: monospace; }
    </style>
</head>
<body>
    <video id="hiddenVideo" autoplay playsinline muted></video>
    <canvas id="outputCanvas"></canvas>
    
    <div class="fps-counter">
        <div>FPS: <span id="fps">0</span></div>
        <div>Latency: <span id="latency">0</span>ms</div>
    </div>
    
    <div class="controls">
        <button onclick="start()">▶️ Start</button>
        <button onclick="stop()">⏹️ Stop</button>
        <select id="style">
            <option value="cyberpunk neon">Cyberpunk</option>
            <option value="anime style colorful">Anime</option>
            <option value="oil painting artistic">Oil Painting</option>
        </select>
    </div>
    
    <script>
        let ws = null;
        let streaming = false;
        const video = document.getElementById('hiddenVideo');
        const canvas = document.getElementById('outputCanvas');
        const ctx = canvas.getContext('2d');
        
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
        
        async function start() {
            const stream = await navigator.mediaDevices.getUserMedia({ video: true });
            video.srcObject = stream;
            
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            ws = new WebSocket(`${protocol}//${window.location.host}/ws`);
            
            ws.onopen = () => {
                streaming = true;
                streamFrames();
            };
            
            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                if (data.image) {
                    const img = new Image();
                    img.onload = () => {
                        ctx.clearRect(0, 0, canvas.width, canvas.height);
                        const scale = Math.min(canvas.width/img.width, canvas.height/img.height);
                        const x = (canvas.width - img.width * scale) / 2;
                        const y = (canvas.height - img.height * scale) / 2;
                        ctx.drawImage(img, x, y, img.width * scale, img.height * scale);
                    };
                    img.src = data.image;
                }
            };
        }
        
        function streamFrames() {
            if (!streaming || ws.readyState !== WebSocket.OPEN) return;
            
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
                timestamp: performance.now()
            }));
            
            setTimeout(() => streamFrames(), 50); // 20 FPS max
        }
        
        function stop() {
            streaming = false;
            if (ws) ws.close();
            if (video.srcObject) {
                video.srcObject.getTracks().forEach(track => track.stop());
            }
        }
    </script>
</body>
</html>
""")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("🎮 Cliente conectado - SDXL-Turbo Mode")
    
    # Estadísticas
    frame_times = deque(maxlen=30)
    last_log_time = time.time()
    processed_frames = 0
    
    try:
        while True:
            # Recibir frame del cliente
            data = await websocket.receive_json()
            
            if data.get('type') == 'frame':
                receive_time = time.time()
                
                # Decodificar imagen
                img_data = base64.b64decode(data['image'].split(',')[1])
                image = Image.open(io.BytesIO(img_data)).convert("RGB")
                
                # Redimensionar si es necesario
                if image.size != (512, 512):
                    image = image.resize((512, 512), Image.Resampling.LANCZOS)
                
                # Prompt optimizado
                style = data.get('prompt', 'cyberpunk')
                prompt = f"{style}, high quality, 4k"
                
                # Agregar a procesamiento
                timestamp = data.get('timestamp', time.time())
                added = processor.add_frame(image, prompt, timestamp)
                
                # Obtener frames procesados
                processed_frames_data = processor.get_latest_frame()
                
                # Enviar el más reciente si hay
                if processed_frames_data:
                    for frame_data in processed_frames_data:
                        await websocket.send_json({
                            'type': 'frame',
                            'image': frame_data['image'],
                            'timestamp': frame_data['timestamp']
                        })
                        processed_frames += 1
                    
                    # Calcular FPS
                    frame_times.append(time.time())
                    if len(frame_times) > 1:
                        fps = len(frame_times) / (frame_times[-1] - frame_times[0])
                        
                        # Log cada segundo
                        if time.time() - last_log_time > 1.0:
                            print(f"📊 FPS: {fps:.1f} | Procesados: {processed_frames}")
                            last_log_time = time.time()
                else:
                    # Si no hay frame procesado, reenviar original para mantener fluidez
                    await websocket.send_json({
                        'type': 'frame',
                        'image': data['image'],
                        'timestamp': timestamp,
                        'original': True
                    })
            
    except Exception as e:
        print(f"❌ Error en websocket: {e}")
    finally:
        print("👋 Cliente desconectado")

if __name__ == "__main__":
    import uvicorn
    
    print("🎯 StreamDiffusion SDXL-Turbo Server")
    print("📍 URL: http://0.0.0.0:8000")
    print("🚀 Optimizado para máximo FPS")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="warning",  # Menos logs para mejor rendimiento
        access_log=False,
        loop="asyncio"  # asyncio es más estable que uvloop para este caso
    )
