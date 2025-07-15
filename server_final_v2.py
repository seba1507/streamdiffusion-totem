#!/usr/bin/env python3
import os
import sys
import time
import asyncio
import threading
import queue
from collections import deque
import hashlib

# -------------------------------------------------------------------
#  CONFIGURACIÓN CUDA
# -------------------------------------------------------------------
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

import torch
import numpy as np

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

# -------------------------------------------------------------------
#  FASTAPI
# -------------------------------------------------------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------------------------
#  ULTRA‑FAST SD‑TURBO PIPELINE
# -------------------------------------------------------------------
class UltraFastDiffusion:
    def __init__(self):
        self.pipe = None
        self.device = torch.device("cuda:0")
        self.dtype = torch.float16

        # Cola de frames y resultados
        self.frame_queue  = queue.Queue(maxsize=2)
        self.result_queue = queue.Queue(maxsize=3)
        self.processing   = False

        # Similarity filter
        self.last_image_hash       = None
        self.similarity_threshold  = 0.95
        self.frame_skip_counter    = 0

        # Métricas
        self.total_frames  = 0
        self.skipped_frames = 0

        self.init_model()

    # ---------------------------------------------------------------
    #  MODELO
    # ---------------------------------------------------------------
    def init_model(self):
        print("Cargando SD‑Turbo optimizado…")

        model_id = "stabilityai/sd-turbo"
        self.pipe = AutoPipelineForImage2Image.from_pretrained(
            model_id,
            torch_dtype=self.dtype,
            variant="fp16",
            safety_checker=None,
            requires_safety_checker=False
        ).to(self.device)

        self.pipe.set_progress_bar_config(disable=True)

        # XFormers
        xformers_enabled = False
        try:
            self.pipe.enable_xformers_memory_efficient_attention()
            xformers_enabled = True
            print("✓ XFormers habilitado")
        except Exception as e:
            print(f"⚠ XFormers no disponible: {e}")

        # torch.compile si no hay conflicto
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

        # VAE slicing + tiling
        try:
            self.pipe.vae.enable_slicing()
            self.pipe.vae.enable_tiling()
            print("✓ VAE optimizado (slicing + tiling)")
        except Exception as e:
            print(f"⚠ Optimizaciones VAE fallaron: {e}")

        # Pre‑calentamiento
        print("Pre‑calentando pipeline…")
        dummy_image = Image.new('RGB', (512, 512), color='black')
        try:
            with torch.no_grad(), torch.cuda.amp.autocast():
                _ = self.pipe(
                    "test",
                    image=dummy_image,
                    num_inference_steps=1,
                    strength=1.0,
                    guidance_scale=0.0
                ).images[0]
            print("✓ Pre‑calentamiento exitoso")
        except Exception as e:
            print(f"⚠ Error en pre‑calentamiento: {e}")

        print("✓ Modelo listo!")
        self.start_processing_thread()

    # ---------------------------------------------------------------
    #  HASH & SIMILARITY
    # ---------------------------------------------------------------
    def calculate_image_hash(self, image):
        try:
            small = image.resize((64, 64))
            return hashlib.md5(np.asarray(small).tobytes()).hexdigest()
        except Exception:
            return str(time.time())

    def should_skip_frame(self, image):
        try:
            current = self.calculate_image_hash(image)
            if self.last_image_hash is None:
                self.last_image_hash = current
                return False
            if current == self.last_image_hash:
                self.frame_skip_counter += 1
                return True
            self.last_image_hash = current
            self.frame_skip_counter = 0
            return False
        except Exception:
            return False

    # ---------------------------------------------------------------
    #  THREAD DE PROCESAMIENTO
    # ---------------------------------------------------------------
    def start_processing_thread(self):
        self.processing = True
        threading.Thread(target=self._processing_loop, daemon=True).start()
        print("✓ Thread de procesamiento iniciado")

    def _processing_loop(self):
        while self.processing:
            try:
                frame_data = self.frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error en processing loop: {e}")
                continue

            try:
                # Similarity filter
                if self.should_skip_frame(frame_data['image']):
                    self.skipped_frames += 1
                    # Re‑emitir último resultado para mantener FPS
                    try:
                        last_result = self.result_queue.queue[-1] if self.result_queue.queue else None
                        if last_result:
                            self.result_queue.put_nowait(last_result)
                    except queue.Full:
                        pass
                    continue

                start = time.time()

                with torch.no_grad(), torch.cuda.amp.autocast():
                    result_img = self.pipe(
                        prompt          = frame_data['prompt'],
                        image           = frame_data['image'],
                        num_inference_steps = 1,
                        strength        = frame_data.get('strength', 1.0),
                        guidance_scale  = frame_data.get('guidance_scale', 0.0),
                        generator       = torch.Generator(device=self.device).manual_seed(42)
                    ).images[0]

                elapsed_ms = (time.time() - start) * 1000
                self.total_frames += 1

                result_data = {
                    'image'          : result_img,
                    'processing_time': elapsed_ms,
                    'timestamp'      : frame_data['timestamp'],
                    'stats': {
                        'total_frames' : self.total_frames,
                        'skipped_frames': self.skipped_frames,
                        'skip_rate'    : self.skipped_frames / max(1, self.total_frames) * 100
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

                print(f"Frame procesado: {elapsed_ms:.1f} ms (Skip {self.skipped_frames}/{self.total_frames})")

            except Exception as e:
                print(f"Error procesando frame: {e}")

    # ---------------------------------------------------------------
    #  API
    # ---------------------------------------------------------------
    def add_frame(self, image, prompt, timestamp, strength=1.0, guidance_scale=0.0):
        frame_data = {
            "image"         : image,
            "prompt"        : prompt,
            "timestamp"     : timestamp,
            "strength"      : strength,
            "guidance_scale": guidance_scale
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
        result = None
        while not self.result_queue.empty():
            try:
                result = self.result_queue.get_nowait()
            except queue.Empty:
                break
        return result

# Instancia global
processor = UltraFastDiffusion()

# -------------------------------------------------------------------
#  HTML / JS (con sliders strength & guidance + prompt libre)
# -------------------------------------------------------------------
HTML_CONTENT = """
<!DOCTYPE html>
<html>
<head>
    <title>Ultra Fast SD-Turbo Stream</title>
    <style>
        body{margin:0;background:#000;color:#fff;font-family:Consolas,monospace;overflow:hidden}
        .container{display:flex;height:100vh;align-items:center;justify-content:center;gap:20px}
        video,canvas{width:512px;height:512px;border:2px solid #00ff41;background:#111;border-radius:8px}
        .output-container{position:relative}
        .controls{position:fixed;bottom:20px;left:50%;transform:translateX(-50%);background:rgba(0,0,0,0.9);padding:20px;border-radius:15px;display:flex;gap:15px;align-items:center;border:1px solid #00ff41}
        button{padding:12px 24px;font-size:16px;cursor:pointer;background:linear-gradient(45deg,#00ff41,#00cc33);color:#000;border:none;border-radius:8px;font-weight:bold;transition:all .3s}
        button:hover{transform:scale(1.05);box-shadow:0 0 20px #00ff41}
        button:disabled{background:#333;color:#666;cursor:not-allowed;transform:none;box-shadow:none}
        select,label,input{font-size:16px;border-radius:8px}
        select{padding:12px;background:#222;color:#fff;border:1px solid #00ff41}
        label{display:flex;flex-direction:column;align-items:center;color:#fff}
        input[type="range"]{width:140px}
        .stats{position:fixed;top:20px;right:20px;background:rgba(0,0,0,0.9);padding:20px;border-radius:15px;font-family:Consolas,monospace;font-size:14px;border:1px solid #00ff41;min-width:250px}
        .stat-value{color:#00ff41;font-weight:bold;font-size:16px}
        .stat-row{display:flex;justify-content:space-between;margin:8px 0}
        .status-indicator{position:absolute;top:10px;left:10px;width:20px;height:20px;border-radius:50%;background:#ff4444;transition:all .3s}
        .status-indicator.active{background:#00ff41;box-shadow:0 0 15px #00ff41}
        .title{text-align:center;margin-bottom:10px;color:#00ff41;font-size:18px;font-weight:bold}
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
        <div class="title">SD‑TURBO METRICS</div>
        <div class="stat-row"><span>FPS:</span><span id="fps" class="stat-value">0</span></div>
        <div class="stat-row"><span>Latencia:</span><span id="latency" class="stat-value">0</span> ms</div>
        <div class="stat-row"><span>Total Frames:</span><span id="totalFrames" class="stat-value">0</span></div>
        <div class="stat-row"><span>Skip Rate:</span><span id="skipRate" class="stat-value">0</span>%</div>
    </div>

    <div class="controls">
        <button id="startBtn" onclick="start()">🚀 INICIAR</button>
        <button id="stopBtn" onclick="stop()" disabled>⏹ DETENER</button>

        <select id="style">
            <option value="">Fotorealista</option>
            <option value="cyberpunk futuristic">Cyberpunk</option>
            <option value="anime style">Anime</option>
            <option value="oil painting masterpiece">Óleo</option>
            <option value="watercolor painting">Acuarela</option>
            <option value="digital art">Arte Digital</option>
        </select>

        <label>Strength
            <input type="range" id="strength" min="0" max="1" step="0.05" value="1">
        </label>
        <label>Guidance
            <input type="range" id="guidance" min="0" max="20" step="0.5" value="0">
        </label>

        <input id="customPrompt" type="text" placeholder="Prompt opcional" style="width:250px;padding:12px;border:1px solid #00ff41;background:#222;color:#fff">
    </div>

    <script>
        let ws=null,streaming=false,video=null,canvas=null,ctx=null;
        let frameCount=0,totalFrames=0,lastTime=Date.now();const latencies=[];
        async function start(){
            try{
                video=document.getElementById('video');
                canvas=document.getElementById('output');
                ctx=canvas.getContext('2d');canvas.width=512;canvas.height=512;

                const stream=await navigator.mediaDevices.getUserMedia({video:{width:{ideal:1280},height:{ideal:720},frameRate:{ideal:30}}});
                video.srcObject=stream;await new Promise(r=>video.onloadedmetadata=r);

                const protocol=location.protocol==='https:'?'wss:':'ws:';
                ws=new WebSocket(`${protocol}//${location.host}/ws`);

                ws.onopen=()=>{console.log('WS conectado');streaming=true;
                    document.getElementById('startBtn').disabled=true;
                    document.getElementById('stopBtn').disabled=false;
                    document.getElementById('statusIndicator').classList.add('active');
                    sendFrame();
                };
                ws.onmessage=e=>{
                    const data=JSON.parse(e.data);
                    if(data.image){
                        const img=new Image();
                        img.onload=()=>{ctx.clearRect(0,0,512,512);ctx.drawImage(img,0,0,512,512)};
                        img.src=data.image;
                        updateStats(data);
                    }
                };
                ws.onerror=e=>{console.error('WS error',e);stop()};
                ws.onclose=()=>{console.log('WS cerrado');stop()};
            }catch(err){console.error(err);alert(err.message)}
        }
        function sendFrame(){
            if(!streaming||!ws||ws.readyState!==1)return;
            if(video.videoWidth===0||video.videoHeight===0){return setTimeout(sendFrame,10);}
            const tmp=document.createElement('canvas');tmp.width=512;tmp.height=512;const tctx=tmp.getContext('2d');

            const aspect=video.videoWidth/video.videoHeight;let sx,sy,sw,sh;
            if(aspect>1){sw=video.videoHeight;sh=video.videoHeight;sx=(video.videoWidth-sw)/2;sy=0;}
            else{sw=video.videoWidth;sh=video.videoWidth;sx=0;sy=(video.videoHeight-sh)/2;}
            tctx.drawImage(video,sx,sy,sw,sh,0,0,512,512);

            const style=document.getElementById('style').value.trim();
            const custom=document.getElementById('customPrompt').value.trim();
            const prompt=custom?custom:(style?`${style}, high quality, detailed`:"high quality, detailed, photorealistic");

            const strength=parseFloat(document.getElementById('strength').value);
            const guidance=parseFloat(document.getElementById('guidance').value);

            ws.send(JSON.stringify({
                image: tmp.toDataURL('image/jpeg',0.9),
                prompt: prompt,
                strength: strength,
                guidance_scale: guidance,
                timestamp: Date.now()
            }));
            requestAnimationFrame(sendFrame);
        }
        function updateStats(data){
            frameCount++;totalFrames++;
            if(data.processing_time){latencies.push(data.processing_time);if(latencies.length>50)latencies.shift();}
            const now=Date.now();
            if(now-lastTime>=1000){
                document.getElementById('fps').textContent=frameCount;
                frameCount=0;lastTime=now;
                if(latencies.length){document.getElementById('latency').textContent=Math.round(latencies.reduce((a,b)=>a+b)/latencies.length);}
            }
            if(data.stats){
                document.getElementById('totalFrames').textContent=data.stats.total_frames;
                document.getElementById('skipRate').textContent=data.stats.skip_rate.toFixed(1);
            }
        }
        function stop(){
            streaming=false;
            if(ws){ws.close();ws=null;}
            if(video&&video.srcObject){video.srcObject.getTracks().forEach(t=>t.stop());video.srcObject=null;}
            if(ctx){ctx.clearRect(0,0,512,512);}
            document.getElementById('startBtn').disabled=false;
            document.getElementById('stopBtn').disabled=true;
            document.getElementById('statusIndicator').classList.remove('active');
        }
        window.addEventListener('beforeunload',stop);
    </script>
</body>
</html>
"""

# -------------------------------------------------------------------
#  ENDPOINTS
# -------------------------------------------------------------------
@app.get("/")
async def get():
    return HTMLResponse(content=HTML_CONTENT)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("🚀 Cliente conectado")
    try:
        while True:
            data = await websocket.receive_json()
            if 'image' not in data:
                continue

            try:
                img_data = base64.b64decode(data['image'].split(',')[1])
                inp_img  = Image.open(io.BytesIO(img_data)).convert("RGB")
                inp_img  = inp_img.resize((512,512), Image.Resampling.LANCZOS)
            except Exception as e:
                print(f"❌ Error decodificando imagen: {e}")
                continue

            strength       = float(data.get("strength",1.0))
            guidance_scale = float(data.get("guidance_scale",0.0))

            processor.add_frame(
                inp_img,
                data.get('prompt',"high quality, detailed"),
                data.get('timestamp',time.time()*1000),
                strength=strength,
                guidance_scale=guidance_scale
            )

            result = processor.get_latest_result()
            if result:
                buff = io.BytesIO()
                result['image'].save(buff,format="JPEG",quality=90)
                img_str = base64.b64encode(buff.getvalue()).decode()

                await websocket.send_json({
                    'image'          : f'data:image/jpeg;base64,{img_str}',
                    'processing_time': result['processing_time'],
                    'timestamp'      : result['timestamp'],
                    'stats'          : result['stats']
                })
    except Exception as e:
        print(f"❌ Error WebSocket: {e}")
    finally:
        print("🔌 Cliente desconectado")

# -------------------------------------------------------------------
#  MAIN
# -------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*70)
    print("🚀 SD‑TURBO STREAM SERVER (STR + GUIDANCE + PROMPT)")
    print("="*70)
    print("🌐 URL: http://0.0.0.0:8000\n")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
