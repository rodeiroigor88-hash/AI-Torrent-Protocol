import subprocess
import time
import os
import sys

def run_test():
    print("Iniciando Worker 1 (Capas 8-15)...")
    worker1 = subprocess.Popen([
        sys.executable, "-u", "src/worker.py", 
        "--port", "8001", 
        "--layers", "8-15", 
        "--next-node", "http://127.0.0.1:8002/forward"
    ])
    
    print("Iniciando Worker 2 (Capas 16-23 + Head)...")
    worker2 = subprocess.Popen([
        sys.executable, "-u", "src/worker.py", 
        "--port", "8002", 
        "--layers", "16-23", 
        "--next-node", "http://127.0.0.1:8000/callback",
        "--is-last"
    ])
    
    print("Esperando a que los workers carguen el modelo (30s)...")
    time.sleep(30) # Dar tiempo para que descarguen y carguen en RAM
    
    print("Iniciando Chat Agent...")
    # Le pasamos "salir" después de generar para que termine
    chat_input = "Crea un archivo python que imprima hola mundo. Nómbralo # test_hello.py\ny\nsalir\n"
    
    chat_agent = subprocess.Popen([
        sys.executable, "-u", "src/chat_agent.py",
        "--port", "8000",
        "--next-node", "http://127.0.0.1:8001/forward"
    ], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    
    try:
        stdout, _ = chat_agent.communicate(input=chat_input, timeout=120)
        print(stdout)
    except subprocess.TimeoutExpired:
        chat_agent.kill()
        print("Timeout esperado del agente")
        
    # Limpieza
    worker1.kill()
    worker2.kill()
    
    if os.path.exists("test_hello.py") or os.path.exists("script.py"):
        filename = "test_hello.py" if os.path.exists("test_hello.py") else "script.py"
        print(f"\nEXITO! El agente creo {filename}")
        os.remove(filename)
    else:
        print("\nFALLO: El agente no creo el archivo.")

if __name__ == "__main__":
    run_test()
