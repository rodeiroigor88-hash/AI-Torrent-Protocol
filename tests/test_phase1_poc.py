import asyncio
import logging
import sys
import os

# Asegurarnos de que puede importar src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.p2p_node import P2PNode
import torch
from aiohttp import ClientSession

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(name)s - %(message)s',
    handlers=[
        logging.FileHandler("AI-Torrent.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

async def main():
    # 1. Definimos las URLs de los 3 nodos (y el callback)
    # Flujo: Cliente -> Nodo A -> Nodo B -> Nodo C -> Callback del Cliente
    
    url_callback = "http://127.0.0.1:8000/callback"
    
    # Nodo C: el último, debe enviar al callback
    node_c = P2PNode(host="127.0.0.1", port=8003, next_node_url=url_callback)
    node_c.operation = lambda x: x * 3  # Multiplica por 3
    
    # Nodo B: envía al Nodo C
    node_b = P2PNode(host="127.0.0.1", port=8002, next_node_url="http://127.0.0.1:8003")
    node_b.operation = lambda x: x * 2  # Multiplica por 2
    
    # Nodo A: envía al Nodo B
    node_a = P2PNode(host="127.0.0.1", port=8001, next_node_url="http://127.0.0.1:8002")
    node_a.operation = lambda x: x + 1  # Suma 1 para variar
    
    # Cliente (actuaremos como servidor en el puerto 8000 solo para recibir el callback)
    client_node = P2PNode(host="127.0.0.1", port=8000, next_node_url=None)
    
    # Future para esperar a que termine todo
    loop = asyncio.get_running_loop()
    finished = loop.create_future()
    
    def on_result_received(tensor):
        logger.info(f"=== PRUEBA COMPLETADA ===")
        logger.info(f"El resultado final llegó de vuelta al cliente: {tensor.tolist()}")
        finished.set_result(True)
        
    client_node.callback_handler = on_result_received
    
    # 2. Iniciamos todos los servidores asíncronos
    runners = []
    for node in [client_node, node_a, node_b, node_c]:
        runner = await node.start()
        runners.append(runner)
        
    await asyncio.sleep(1) # Esperar un momento a que los puertos estén escuchando
    
    # 3. Disparamos la petición inicial al Nodo A simulando ser un usuario real
    tensor_inicial = torch.tensor([2.0], dtype=torch.float32)
    logger.info(f"\n---> Enviando Tensor Inicial al Nodo A: {tensor_inicial.tolist()}")
    
    from src.tensor_utils import serialize_tensor
    payload = serialize_tensor(tensor_inicial)
    
    async with ClientSession() as session:
        async with session.post("http://127.0.0.1:8001/forward", data=payload, headers={'Content-Type': 'application/msgpack'}) as resp:
            logger.info(f"Respuesta inicial del Nodo A: {resp.status} {await resp.text()}\n")
            
    # 4. Esperamos a que el callback se ejecute
    try:
        await asyncio.wait_for(finished, timeout=5.0)
    except asyncio.TimeoutError:
        logger.error("Timeout: El resultado nunca llegó al callback.")
        
    # 5. Apagamos todo
    logger.info("Apagando nodos...")
    for runner in runners:
        await runner.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
