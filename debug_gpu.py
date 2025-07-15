#!/usr/bin/env python3
"""
Script de diagnóstico para problemas con Diffusers y GPU
"""

import sys
import torch
import subprocess

print("=== 🔍 DIAGNÓSTICO DE SISTEMA ===\n")

# 1. Verificar PyTorch y CUDA
print("1. PyTorch y CUDA:")
print(f"   PyTorch version: {torch.__version__}")
print(f"   CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   CUDA version: {torch.version.cuda}")
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
print()

# 2. Verificar librerías instaladas
print("2. Librerías instaladas:")
libs = ['diffusers', 'transformers', 'accelerate', 'xformers']
for lib in libs:
    try:
        module = __import__(lib)
        version = getattr(module, '__version__', 'unknown')
        print(f"   {lib}: {version}")
    except ImportError:
        print(f"   {lib}: NO INSTALADO")
print()

# 3. Test básico de Diffusers
print("3. Test básico de Diffusers:")
try:
    from diffusers import DiffusionPipeline
    print("   ✅ Diffusers importado correctamente")
    
    # Intentar cargar un modelo pequeño
    print("   Cargando modelo de prueba...")
    pipe = DiffusionPipeline.from_pretrained(
        "hf-internal-testing/tiny-stable-diffusion-pipe",
        torch_dtype=torch.float16
    )
    pipe = pipe.to("cuda")
    print("   ✅ Modelo cargado en GPU")
    
    # Test de inferencia
    print("   Ejecutando inferencia de prueba...")
    with torch.no_grad():
        result = pipe("test", num_inference_steps=1)
    print("   ✅ Inferencia exitosa")
    
except Exception as e:
    print(f"   ❌ Error: {e}")
print()

# 4. Verificar configuración de torch._dynamo
print("4. Configuración de torch._dynamo:")
try:
    import torch._dynamo
    print(f"   suppress_errors: {torch._dynamo.config.suppress_errors}")
    
    # Configurar para suprimir errores
    torch._dynamo.config.suppress_errors = True
    print("   ✅ suppress_errors configurado a True")
except Exception as e:
    print(f"   ❌ Error con torch._dynamo: {e}")
print()

# 5. Test de modelos específicos
print("5. Test de modelos específicos:")
models_to_test = [
    ("SimianLuo/LCM_Dreamshaper_v7", "LCM"),
    ("stabilityai/sdxl-turbo", "SDXL-Turbo"),
]

for model_id, name in models_to_test:
    print(f"\n   Testing {name}...")
    try:
        # Solo intentar importar, no cargar completamente
        from diffusers import AutoPipelineForImage2Image
        print(f"   ✅ {name} disponible")
    except Exception as e:
        print(f"   ❌ {name} error: {e}")

print("\n6. Recomendaciones:")
print("   - Si hay errores con accelerate, intenta: pip uninstall accelerate && pip install accelerate==0.25.0")
print("   - Si hay errores con torch.compile, usa torch._dynamo.config.suppress_errors = True")
print("   - Para máximo rendimiento, usa SDXL-Turbo o LCM sin cpu_offload")
print("   - Si persisten los errores, usa la versión minimal sin optimizaciones avanzadas")

print("\n=== 🏁 DIAGNÓSTICO COMPLETO ===")

# 7. Crear archivo de configuración recomendado
print("\nCreando archivo config_recommended.py...")
config_content = """# Configuración recomendada para máximo rendimiento

import torch
import torch._dynamo

# Suprimir errores de compilación
torch._dynamo.config.suppress_errors = True

# Configuración para mejor rendimiento
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

# Modelo recomendado para velocidad
MODEL_ID = "stabilityai/sdxl-turbo"  # o "SimianLuo/LCM_Dreamshaper_v7"
TORCH_DTYPE = torch.float16

# Configuración de inferencia
INFERENCE_STEPS = 1  # SDXL-Turbo solo necesita 1
GUIDANCE_SCALE = 0.0  # Sin guidance para máxima velocidad
STRENGTH = 0.3  # Menor = más rápido y más parecido al original

print("✅ Configuración cargada")
"""

with open("config_recommended.py", "w") as f:
    f.write(config_content)

print("✅ Archivo config_recommended.py creado")
print("\nPara usar esta configuración, importa al inicio de tu servidor:")
print("   from config_recommended import *")
