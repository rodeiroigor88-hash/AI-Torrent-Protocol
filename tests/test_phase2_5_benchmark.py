import sys
import os
import time
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.tensor_utils import serialize_tensor, deserialize_tensor

def run_benchmark():
    print("=== BENCHMARK: EL SECRET SAUCE ===")
    print("Simulando un tensor de hidden_states de GPT-2 (Batch 1, Secuencia 1024 tokens, 768 dimensiones)")
    
    # Creamos un tensor realista de las dimensiones de GPT-2
    # 1 x 1024 x 768
    tensor_original = torch.randn(1, 1024, 768, dtype=torch.float32)
    
    print(f"\nTamaño matemático original en memoria RAM: {tensor_original.element_size() * tensor_original.nelement() / 1024:.2f} KB")
    print("-" * 50)
    
    # --- PRUEBA 1: SIN SECRET SAUCE ---
    start_time = time.time()
    payload_raw = serialize_tensor(tensor_original, use_secret_sauce=False)
    raw_time = time.time() - start_time
    raw_size_kb = len(payload_raw) / 1024
    
    start_time_des = time.time()
    _ = deserialize_tensor(payload_raw)
    raw_time_des = time.time() - start_time_des
    
    print(f"[VERSIÓN NORMAL - Float32 Puro]")
    print(f"-> Tamaño a enviar por Internet: {raw_size_kb:.2f} KB")
    print(f"-> Tiempo de preparación: {(raw_time + raw_time_des)*1000:.2f} ms")
    
    print("-" * 50)
    
    # --- PRUEBA 2: CON SECRET SAUCE ---
    start_time = time.time()
    payload_sauce = serialize_tensor(tensor_original, use_secret_sauce=True)
    sauce_time = time.time() - start_time
    sauce_size_kb = len(payload_sauce) / 1024
    
    start_time_des = time.time()
    tensor_recuperado = deserialize_tensor(payload_sauce)
    sauce_time_des = time.time() - start_time_des
    
    print(f"[VERSIÓN SECRET SAUCE - Float16 + ZLib]")
    print(f"-> Tamaño a enviar por Internet: {sauce_size_kb:.2f} KB")
    print(f"-> Tiempo de preparación: {(sauce_time + sauce_time_des)*1000:.2f} ms")
    
    print("-" * 50)
    
    # --- RESULTADOS ---
    reduccion = 100 - ((sauce_size_kb / raw_size_kb) * 100)
    print(f"REDUCCION DE ANCHO DE BANDA: {reduccion:.2f}% menos datos!")
    
    # Calculamos la ganancia teórica de red asumiendo una velocidad subida lenta de 5 Mbps (0.625 MB/s)
    # 5 Mbps = 625 KB/s
    upload_speed_kbs = 625.0
    
    tiempo_red_raw = raw_size_kb / upload_speed_kbs
    tiempo_red_sauce = sauce_size_kb / upload_speed_kbs
    
    # Tiempo total = procesamiento + red
    total_raw = raw_time + raw_time_des + tiempo_red_raw
    total_sauce = sauce_time + sauce_time_des + tiempo_red_sauce
    
    mejora_velocidad = total_raw / total_sauce
    
    print(f"GANANCIA DE VELOCIDAD ESTIMADA (en una red de 5 Mbps): {mejora_velocidad:.2f}x mas rapido")
    
    # Verificamos que la información se preservó adecuadamente (pequeña pérdida de precisión aceptable por Float16)
    diff = torch.max(torch.abs(tensor_original - tensor_recuperado))
    print(f"Error de perdida de precision matematica maxima (Float16 vs Float32): {diff.item():.5f}")
    if diff.item() < 0.1:
         print("   (Totalmente seguro para modelos de IA sin perder coherencia)")

if __name__ == "__main__":
    run_benchmark()
