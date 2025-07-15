from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import torch
from diffusers import DiffusionPipeline
import base64
from PIL import Image
import io
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pipeline global
pipe = None

def load_model():
    """Cargar modelo más simple y estable"""
    global pipe
    logger.info("Cargando modelo...")
    
    # Usar el modelo LCM que sabemos que funciona
    pipe = DiffusionPipeline.from_pretrained(
        "SimianLuo/LCM_Dreamshaper_v7",
        custom_pipeline="latent_consistency_txt2img",
        custom_revision="main",
        torch_dtype=torch.float16
    )
    
    # Configuración simple sin offload ni compile
    pipe = pipe.to("cuda")
    
    # Solo XFormers si está disponible
    try:
        pipe.enable_xformers_memory_efficient_attention()
        logger.info("XFormers habilitado")
    except:
        logger.info("Usando atención estándar")
    
    logger.info("Modelo cargado!")

# Cargar modelo
load_model()

@app.get("/")
async def get():
    return HTMLResponse(content="""
<!DOCTYPE html>
<html>
<head>
    <title>Stream Diffusion Minimal</title>
    <style>
        body { margin: 0; background: #1a1a1a; color: white; font-family: Arial; }
        .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
        video, canvas { width: 512px; height: 512px; border: 2px solid #333; margin: 10px; }
        .controls { margin: 20px 0; }
        button { padding: 10px 20px; margin: 5px; font-size: 16px; cursor: pointer; }
        .status { margin: 10px 0; padding: 10px; background: #333; border-radius: 5px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎨 Stream Diffusion - Minimal Version</h1>
        
        <div style="display: flex; justify-content: center;">
            <div>
                <h3>📷 Cámara</h3>
                <video id="video" autoplay muted></video>
            </div>
            <div>
                <h3>🎨 Resultado</h3>
                <canvas id="canvas"></canvas>
            </div>
        </div>
        
        <div class="controls">
            <button onclick="start()">▶️ Iniciar</button>
            <button onclick="stop()">⏹️ Detener</button>
            
            <select id="style">
                <option value="">Original</option>
                <option value="cyberpunk">Cyberpunk</option>
                <option value="anime">Anime</option>
                <option value="watercolor">Acuarela</option>
            </select>
            
            <label>
                FPS: <input type="range" id="fps" min="1" max="5" value="2">
                <span id="fpsValue">2</span>
            </label>
        </div>
        
        <div class="status" id="status">
            Estado: Listo
        </div>
    </div>
    
    <script>
        const video = document.getElementById('video');
        const canvas = document.getElementById('canvas');
        const ctx = canvas.getContext('2d');
        const status = document.getElementById('status');
        const fpsSlider = document.getElementById('fps');
        const fpsValue = document.getElementById('fpsValue');
        
        canvas.width = 512;
        canvas.height = 512;
        
        let ws = null;
        let isStreaming = false;
        let frameCount = 0;
        let startTime = Date.now();
        
        fpsSlider.oninput = () => {
            fpsValue.textContent = fpsSlider.value;
        };
        
        async function start() {
            try {
                // Obtener cámara
                const stream = await navigator.mediaDevices.getUserMedia({ 
                    video: { width: 512, height: 512 } 
                });
                video.srcObject = stream;
                
                // Conectar WebSocket
                const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                ws = new WebSocket(`${protocol}//${window.location.host}/ws`);
                
                ws.onopen = () => {
                    status.textContent = 'Estado: Conectado';
                    isStreaming = true;
                    sendFrame();
                };
                
                ws.onmessage = (event) => {
                    const data = JSON.parse(event.data);
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
                
                ws.onerror = (error) => {
                    status.textContent = 'Estado: Error de conexión';
                    console.error('WebSocket error:', error);
                };
                
                ws.onclose = () => {
                    status.textContent = 'Estado: Desconectado';
                    isStreaming = false;
                };
                
            } catch (error) {
                status.textContent = 'Estado: Error - ' + error.message;
                console.error('Error:', error);
            }
        }
        
        function sendFrame() {
            if (!isStreaming || !ws || ws.readyState !== WebSocket.OPEN) return;
            
            // Capturar frame
            const tempCanvas = document.createElement('canvas');
            tempCanvas.width = 512;
            tempCanvas.height = 512;
            const tempCtx = tempCanvas.getContext('2d');
            tempCtx.drawImage(video, 0, 0, 512, 512);
            
            const imageData = tempCanvas.toDataURL('image/jpeg', 0.8);
            const style = document.getElementById('style').value;
            
            ws.send(JSON.stringify({
                image: imageData,
                style: style
            }));
            
            // Siguiente frame según FPS configurado
            const delay = 1000 / parseInt(fpsSlider.value);
            setTimeout(sendFrame, delay);
        }
        
        function updateStats() {
            const elapsed = (Date.now() - startTime) / 1000;
            const fps = frameCount / elapsed;
            status.textContent = `Estado: Streaming - ${fps.toFixed(1)} FPS reales`;
        }
        
        function stop() {
            isStreaming = false;
            if (ws) {
                ws.close();
            }
            if (video.srcObject) {
                video.srcObject.getTracks().forEach(track => track.stop());
            }
            status.textContent = 'Estado: Detenido';
        }
    </script>
</body>
</html>
    """)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("Cliente conectado")
    
    while True:
        try:
            # Recibir datos
            data = await websocket.receive_json()
            
            # Decodificar imagen
            img_data = base64.b64decode(data['image'].split(',')[1])
            image = Image.open(io.BytesIO(img_data)).convert("RGB")
            
            # Si hay estilo, aplicar transformación
            style = data.get('style', '')
            if style:
                # Generar con LCM (4-8 pasos)
                prompt = f"{style} style, high quality"
                with torch.no_grad():
                    result = pipe(
                        prompt=prompt,
                        num_inference_steps=4,
                        guidance_scale=8.0,
                        lcm_origin_steps=50,
                        output_type="pil"
                    ).images[0]
                
                # Mezclar con original para mejor coherencia
                result = Image.blend(image, result, alpha=0.7)
            else:
                # Sin estilo, devolver original
                result = image
            
            # Codificar y enviar
            buffered = io.BytesIO()
            result.save(buffered, format="JPEG", quality=85)
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            await websocket.send_json({
                'image': f'data:image/jpeg;base64,{img_str}'
            })
            
        except Exception as e:
            logger.error(f"Error: {e}")
            break
    
    logger.info("Cliente desconectado")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
