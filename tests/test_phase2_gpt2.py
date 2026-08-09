import asyncio
import logging
import sys
import os
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from aiohttp import ClientSession

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.p2p_node import P2PNode
from src.tensor_utils import serialize_tensor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(name)s - %(message)s',
    handlers=[logging.FileHandler("TokenTorrent.log"), logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# --- Clases para encapsular el Sharding del Modelo ---

class NodeBModel(torch.nn.Module):
    def __init__(self, gpt2_model):
        super().__init__()
        # NODO B: Capas 4 a 7
        self.blocks = torch.nn.ModuleList([gpt2_model.transformer.h[i] for i in range(4, 8)])
        
    def forward(self, hidden_states):
        for block in self.blocks:
            # GPT2Block en las versiones nuevas de HF devuelve un tensor, no una tupla si no hay cache
            output = block(hidden_states)
            hidden_states = output[0] if isinstance(output, tuple) else output
        return hidden_states

class NodeCModel(torch.nn.Module):
    def __init__(self, gpt2_model):
        super().__init__()
        # NODO C: Capas 8 a 11 y el LayerNorm Final
        self.blocks = torch.nn.ModuleList([gpt2_model.transformer.h[i] for i in range(8, 12)])
        self.ln_f = gpt2_model.transformer.ln_f
        
    def forward(self, hidden_states):
        for block in self.blocks:
            output = block(hidden_states)
            hidden_states = output[0] if isinstance(output, tuple) else output
        hidden_states = self.ln_f(hidden_states)
        return hidden_states

async def main():
    logger.info("Cargando GPT-2... (Esto puede tardar unos segundos si no está cacheado)")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    model.eval() # Modo inferencia
    logger.info("GPT-2 cargado en memoria.")

    # 1. Instanciamos los modelos divididos
    model_b = NodeBModel(model)
    model_c = NodeCModel(model)
    
    # El Cliente (Nodo A) se queda con la primera parte del modelo
    wte = model.transformer.wte
    wpe = model.transformer.wpe
    drop = model.transformer.drop
    blocks_a = torch.nn.ModuleList([model.transformer.h[i] for i in range(0, 4)])
    lm_head = model.lm_head

    url_callback = "http://127.0.0.1:8000/callback"
    
    # 2. Configurar los Nodos Servidores P2P
    # Nodo C: el último, procesa y envía al callback
    node_c = P2PNode(host="127.0.0.1", port=8003, next_node_url=url_callback)
    node_c.operation = lambda x: model_c(x)
    
    # Nodo B: procesa y envía al Nodo C
    node_b = P2PNode(host="127.0.0.1", port=8002, next_node_url="http://127.0.0.1:8003")
    node_b.operation = lambda x: model_b(x)
    
    # Cliente / Nodo A: donde iniciamos el prompt y recibimos el callback
    client_node = P2PNode(host="127.0.0.1", port=8000, next_node_url=None)
    
    loop = asyncio.get_running_loop()
    finished = loop.create_future()
    
    # Callback para cuando el Nodo C devuelva el tensor final
    def on_result_received(final_hidden_states):
        with torch.no_grad():
            # El Cliente tiene el LM Head para convertir el hidden_state en la palabra predicha
            logits = lm_head(final_hidden_states)
            next_token_logits = logits[0, -1, :]
            next_token_id = torch.argmax(next_token_logits).item()
            next_token_word = tokenizer.decode([next_token_id])
            
            logger.info(f"=== PRUEBA FASE 2 COMPLETADA ===")
            logger.info(f"Siguiente palabra predicha por la red distribuida: '{next_token_word}'")
            finished.set_result(next_token_word)
            
    client_node.callback_handler = on_result_received
    
    # 3. Arrancamos los servidores
    runners = []
    for node in [client_node, node_b, node_c]:
        runner = await node.start()
        runners.append(runner)
        
    await asyncio.sleep(1)
    
    # 4. Fase del Cliente (Nodo A): Procesamos el prompt
    prompt = "The future of artificial intelligence is"
    logger.info(f"\n---> Prompt original: '{prompt}'")
    
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    
    with torch.no_grad():
        # Ejecución Local en el Cliente (Nodo A)
        position_ids = torch.arange(0, input_ids.shape[-1], dtype=torch.long).unsqueeze(0)
        
        inputs_embeds = wte(input_ids)
        position_embeds = wpe(position_ids)
        hidden_states = drop(inputs_embeds + position_embeds)
        
        for block in blocks_a:
            output = block(hidden_states)
            hidden_states = output[0] if isinstance(output, tuple) else output
            
    logger.info(f"[Cliente] Tensor intermedio generado. Tamaño: {hidden_states.shape}. Enviando al Nodo B...")
    
    # 5. Enviamos el tensor al Nodo B para que continúe la cascada
    payload = serialize_tensor(hidden_states)
    async with ClientSession() as session:
        async with session.post("http://127.0.0.1:8002/forward", data=payload, headers={'Content-Type': 'application/msgpack'}) as resp:
            logger.info(f"Respuesta del Nodo B: {resp.status} {await resp.text()}\n")
            
    # 6. Esperamos el resultado de vuelta
    try:
        await asyncio.wait_for(finished, timeout=10.0)
    except asyncio.TimeoutError:
        logger.error("Timeout: El resultado distribuido de GPT-2 nunca llegó.")
        
    logger.info("Apagando nodos...")
    for runner in runners:
        await runner.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
