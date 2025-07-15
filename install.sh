#!/bin/bash

echo "🚀 Instalando StreamDiffusion Totem..."

# Actualizar pip
pip install --upgrade pip

# Instalar dependencias
pip install -r requirements.txt

# Descargar modelo en cache
echo "📥 Pre-descargando modelo LCM..."
python -c "
from diffusers import AutoPipelineForImage2Image
import torch
pipe = AutoPipelineForImage2Image.from_pretrained(
    'SimianLuo/LCM_Dreamshaper_v7',
    torch_dtype=torch.float16
)
print('✅ Modelo descargado!')
"

echo "✅ Instalación completa!"
echo "🎯 Ejecuta 'python server.py' para iniciar"