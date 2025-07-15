from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import torch
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, EulerDiscreteScheduler
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import base64
from PIL import Image
import io
import time
import asyncio
from collections import deque
import os

# Configurar CUDA
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
torch.backends.cudnn.benchmark = True

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class LightningProcessor:
    def __init__(self):
        self.pipe = None
        self.init_model()
        
    def init_model(self):
        print("⚡ Cargando Lightning SDXL...")
        
        try:
            # Configuración base
            base = "stabilityai/stable-diffusion-xl-base-1.0"
            repo = "ByteDance/SDXL-Lightning"
            ckpt = "sdxl_lightning_4step_unet.safetensors"
            
            # Cargar modelo base
            unet = UNet2DConditionModel.from_config(base, subfolder="unet").to("cuda", torch.float16)
            unet.load_state_dict(load_file(hf_hub_download(repo, ckpt), device="cuda"))
            
            # Crear pipeline
            self.pipe = StableDiffusionXLPipeline.from_pretrained(
                base, 
                unet=unet, 
                torch_dtype=torch.float16,
                variant="fp16",
                use_safetensors=True
            ).to("cuda")
            
            # Configurar scheduler para Lightning
            self.pipe.scheduler = EulerDiscreteScheduler.from_config(
                self.pipe.scheduler.config,
                timestep_spacing="trailing",
                prediction_type="epsilon"
            )
            
            # Optimizaciones
            self.pipe.set_progress_bar_config(disable=True)
            
            # XFormers
            try:
                self.pipe.enable_xformers_memory_efficient_attention()
                print("✅ XFormers habilitado")
            except:
                print("⚠️ XFormers no disponible")
            
            # Optimizar VAE
            self.pipe.vae.enable_slicing()
            self.pipe.vae.enable_tiling()
            
            # Pre-calentar
            print("🔥 Pre-calentando...")
            with torch.no_grad():
                _ = self.pipe(
                    prompt="test",
                    num_inference_steps=4,
                    guidance_scale=0,
                    width=512,
                    height=512
                ).images[0]
            
            print("✅ Lightning SDXL listo!")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            # Fallback a LCM
            self.init_lcm_fallback()
    
    def init_lcm_fallback(self):
        """Fallback a LCM si Lightning falla"""
        print("🔄 Cambiando a LCM como fallback...")
        
        from diffusers import LCMScheduler
        
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True
        ).to("cuda")
        
        # Cargar LoRA de LCM
        self.pipe.load_lora_weights("latent-consistency/lcm-lora-sdxl")
        self.pipe.fuse_lora()
        
        # Configurar scheduler LCM
        self.pipe.scheduler = LCMScheduler.from_config(self.pipe.scheduler.config)
        
        print("✅ LCM listo como fallback")
    
    def process_image(self, prompt, num_steps=4):
        """Procesar imagen con manejo de errores"""
        try:
            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    result = self.pipe(
                        prompt=prompt,
                        num_inference_steps=num_steps,
                        guidance_scale=0,
                        width=512,
                        height=512,
                        output_type="pil"
                    ).images[0]
                    return result
        except Exception as e:
            print(f"Error en proceso: {e}")
            torch.cuda.empty_cache()
            return None

# HTML simple y funcional
HTML_CONTENT = """
<!DOCTYPE html>
<html>
<head>
    <title>Lightning Stream</title>
    <style>
        body { margin: 0; background: #111; color: white; font-family: Arial; overflow: hidden; }
        canvas { position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); }
        .controls { position: fixed; bottom: 20px; left: 50%; transform: translateX(-50%); 
                   background: rgba(0,0,0,0.8); padding: 15px; border-radius: 10px; }
        button { padding: 10px 20px; margin: 0 5px; cursor: pointer; }
        .stats { position: fixed; top: 20px; right: 20px; background: rgba(0,0,0,0.8); 
                padding: 10px; border-radius: 5px; font-family: monospace; }
    </style>
</head>
<body>
    <canvas id="canvas"></canvas>
    <div class="stats">
        FPS: <span id="fps">0</span> | 
        Latency: <span id="latency">0</span>ms
    </div>
    <div class="controls">
        <button onclick="start()">Start</button>
        <button onclick="stop()">Stop</button>
        <select id="style">
            <option value="">Original</option>
            <option value="cyberpunk">Cyberpunk</option>
            <option value="anime">Anime</option>
            <option value="oil painting">Oil Painting</option>
        </select>
    </div>
    
    <script>
        const canvas = document.getElementById('canvas');
        const ctx = canvas.getContext('2d');
        canvas.width = 512;
        canvas.height = 512;
        
        let video, ws, streaming = false;
        let frameCount = 0, lastTime = Date.now();
        
        async function start() {
            video = document.createElement('video');
            video.srcObject = await navigator.mediaDevices.getUserMedia({video: true});
            video.play();
            
            ws = new WebSocket(`${location.protocol === 'https:' ? 'wss:' : 'ws:'}//${location.host}/ws`);
            ws.onopen = () => { streaming = true; sendFrame(); };
            ws.onmessage = (e) => {
                const data = JSON.parse(e.data);
                if (data.image) {
                    const img = new Image();
                    img.onload = () => {
                        ctx.drawImage(img, 0, 0, 512, 512);
                        frameCount++;
                        updateStats();
                    };
                    img.src = data.image;
                }
            };
        }
        
        function sendFrame() {
            if (!streaming) return;
            
            const tempCanvas = document.createElement('canvas');
            tempCanvas.width = tempCanvas.height = 512;
            const tempCtx = tempCanvas.getContext('2d');
            tempCtx.drawImage(video, 0, 0, 512, 512);
            
            ws.send(JSON.stringify({
                image: tempCanvas.toDataURL('image/jpeg', 0.7),
                style: document.getElementById('style').value,
                timestamp: Date.now()
            }));
            
            setTimeout(sendFrame, 100); // 10 FPS max
        }
        
        function updateStats() {
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

# Procesador global
processor = None

@app.on_event("startup")
async def startup_event():
    global processor
    processor = LightningProcessor()

@app.get("/")
async def get():
    return HTMLResponse(content=HTML_CONTENT)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("⚡ Cliente conectado")
    
    # Cola para procesamiento asíncrono
    process_queue = asyncio.Queue(maxsize=2)
    
    async def process_loop():
        """Loop de procesamiento en background"""
        while True:
            try:
                data = await process_queue.get()
                if data is None:
                    break
                
                # Procesar
                style = data.get('style', '')
                prompt = f"{style} style" if style else "high quality photo"
                
                result = processor.process_image(prompt, num_steps=4)
                
                if result:
                    # Codificar
                    buffered = io.BytesIO()
                    result.save(buffered, format="JPEG", quality=75)
                    img_str = base64.b64encode(buffered.getvalue()).decode()
                    
                    # Enviar
                    await websocket.send_json({
                        'image': f'data:image/jpeg;base64,{img_str}',
                        'timestamp': data.get('timestamp')
                    })
                    
            except Exception as e:
                print(f"Error en proceso: {e}")
    
    # Iniciar procesador en background
    process_task = asyncio.create_task(process_loop())
    
    try:
        while True:
            data = await websocket.receive_json()
            
            # Agregar a cola (descartando antiguos si está llena)
            try:
                process_queue.put_nowait(data)
            except asyncio.QueueFull:
                # Descartar el más antiguo
                try:
                    process_queue.get_nowait()
                    process_queue.put_nowait(data)
                except:
                    pass
            
    except Exception as e:
        print(f"Error: {e}")
    finally:
        await process_queue.put(None)
        await process_task
        print("Cliente desconectado")

if __name__ == "__main__":
    import uvicorn
    
    print("⚡ Lightning SDXL Server")
    print("📍 http://0.0.0.0:8000")
    print("🎯 4 pasos = máxima velocidad")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="error")
