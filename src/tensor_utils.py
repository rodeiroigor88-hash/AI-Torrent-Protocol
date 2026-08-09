import torch
import msgpack
import zlib
import numpy as np

MAX_COMPRESSED_BYTES = 16 * 1024 * 1024
MAX_DECOMPRESSED_BYTES = 64 * 1024 * 1024
MAX_TENSOR_ELEMENTS = 16 * 1024 * 1024
MAX_TENSOR_DIMENSIONS = 8
# Las listas solo existen en el sobre v2 para transportar la ruta del pipeline
# (ver routing.MAX_ROUTE_HOPS). El tope es deliberadamente pequeno.
MAX_LIST_ITEMS = 64

_TORCH_DTYPES = {
    'float32': torch.float32,
    'float16': torch.float16,
    'float64': torch.float64,
    'int32': torch.int32,
    'int64': torch.int64,
}
_NUMPY_DTYPES = {
    'float32': np.float32,
    'float16': np.float16,
    'float64': np.float64,
    'int32': np.int32,
    'int64': np.int64,
}

def _pack_single_tensor(tensor: torch.Tensor, use_secret_sauce: bool) -> dict:
    if use_secret_sauce and tensor.dtype in [torch.float32, torch.bfloat16]:
        tensor = tensor.to(torch.float16)
    elif tensor.dtype == torch.bfloat16:
        tensor = tensor.to(torch.float32)
        
    return {
        '_is_tensor': True,
        'shape': list(tensor.shape),
        'dtype': str(tensor.dtype).split('.')[-1],
        'data': tensor.detach().cpu().numpy().tobytes(),
        'secret_sauce': use_secret_sauce
    }

def _unpack_single_tensor(tensor_data: dict) -> torch.Tensor:
    dtype_name = tensor_data.get('dtype')
    if dtype_name not in _TORCH_DTYPES:
        raise ValueError(f"Unsupported tensor dtype: {dtype_name}")
    shape = tensor_data.get('shape')
    if not isinstance(shape, list) or len(shape) > MAX_TENSOR_DIMENSIONS:
        raise ValueError("Invalid tensor shape")
    if any(not isinstance(dim, int) or isinstance(dim, bool) or dim < 0 for dim in shape):
        raise ValueError("Invalid tensor dimension")

    element_count = int(np.prod(shape, dtype=np.int64)) if shape else 1
    if element_count > MAX_TENSOR_ELEMENTS:
        raise ValueError("Tensor exceeds the element limit")

    raw_data = tensor_data.get('data')
    if not isinstance(raw_data, bytes):
        raise ValueError("Invalid tensor data")
    dtype = _TORCH_DTYPES[dtype_name]
    np_dtype = _NUMPY_DTYPES[dtype_name]
    expected_bytes = element_count * np.dtype(np_dtype).itemsize
    if len(raw_data) != expected_bytes:
        raise ValueError("Tensor byte length does not match its shape and dtype")

    np_arr = np.frombuffer(raw_data, dtype=np_dtype).copy()

    tensor = torch.from_numpy(np_arr).reshape(shape)
    
    if tensor_data.get('secret_sauce', False) and tensor.dtype == torch.float16:
        tensor = tensor.to(torch.float32)
    else:
        tensor = tensor.to(dtype)
    return tensor

def _pack_value(value, use_secret_sauce: bool):
    if isinstance(value, torch.Tensor):
        return _pack_single_tensor(value, use_secret_sauce)
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("Dictionary keys must be strings")
        return {key: _pack_value(item, use_secret_sauce) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        if len(value) > MAX_LIST_ITEMS:
            raise ValueError("List exceeds the item limit")
        return [_pack_value(item, use_secret_sauce) for item in value]
    if value is None or isinstance(value, (str, bytes, bool, int, float)):
        return value
    raise ValueError(f"Unsupported value type: {type(value).__name__}")

def _unpack_value(value):
    if isinstance(value, dict):
        if value.get('_is_tensor') is True:
            return _unpack_single_tensor(value)
        return {key: _unpack_value(item) for key, item in value.items()}
    if isinstance(value, list):
        if len(value) > MAX_LIST_ITEMS:
            raise ValueError("List exceeds the item limit")
        return [_unpack_value(item) for item in value]
    return value

def serialize_tensor(data, use_secret_sauce: bool = True) -> bytes:
    """
    Convierte un tensor o un diccionario de tensores a una secuencia de bytes usando msgpack y zlib.
    """
    payload_dict = _pack_value(data, use_secret_sauce)
        
    packed = msgpack.packb(payload_dict, use_bin_type=True)
    
    if len(packed) > MAX_DECOMPRESSED_BYTES:
        raise ValueError("Serialized payload exceeds the size limit")
    if use_secret_sauce:
        packed = zlib.compress(packed, level=6)
    if len(packed) > MAX_COMPRESSED_BYTES:
        raise ValueError("Compressed payload exceeds the size limit")
        
    return packed

def deserialize_tensor(payload: bytes):
    """
    Convierte bytes de msgpack de vuelta a tensor o diccionario de tensores.
    """
    if not isinstance(payload, bytes) or len(payload) > MAX_COMPRESSED_BYTES:
        raise ValueError("Payload exceeds the compressed size limit")
    try:
        decompressor = zlib.decompressobj()
        decompressed_payload = decompressor.decompress(payload, MAX_DECOMPRESSED_BYTES + 1)
        if decompressor.unconsumed_tail or len(decompressed_payload) > MAX_DECOMPRESSED_BYTES:
            raise ValueError("Payload exceeds the decompressed size limit")
        decompressed_payload += decompressor.flush(MAX_DECOMPRESSED_BYTES + 1 - len(decompressed_payload))
        if len(decompressed_payload) > MAX_DECOMPRESSED_BYTES:
            raise ValueError("Payload exceeds the decompressed size limit")
    except zlib.error:
        decompressed_payload = payload
    if len(decompressed_payload) > MAX_DECOMPRESSED_BYTES:
        raise ValueError("Payload exceeds the decompressed size limit")
        
    unpacker = msgpack.Unpacker(
        raw=False,
        strict_map_key=True,
        max_buffer_size=MAX_DECOMPRESSED_BYTES,
        # Los unicos arrays del formato son `shape` (<= 8 elementos) y la ruta
        # del sobre v2, asi que el tope puede ser muy bajo: un array gigante de
        # enteros pequenos seria una via de agotamiento de memoria.
        max_array_len=MAX_LIST_ITEMS,
        max_map_len=128,
        max_str_len=1024 * 1024,
        max_bin_len=MAX_DECOMPRESSED_BYTES,
    )
    unpacker.feed(decompressed_payload)
    objects = list(unpacker)
    if len(objects) != 1:
        raise ValueError("Payload must contain exactly one MessagePack object")
    tensor_data = objects[0]
    result = _unpack_value(tensor_data)
    if not isinstance(result, (torch.Tensor, dict)):
        raise ValueError("Unrecognized payload format")
    return result
