from collections.abc import Callable
_REGISTRY: dict[str, Callable] = {}
def register_tool(name: str):
    def decorator(func: Callable): _REGISTRY[name] = func; return func
    return decorator
def get_tool(name: str) -> Callable:
    if name not in _REGISTRY: raise KeyError(f'Tool not registered: {name}')
    return _REGISTRY[name]
def list_tools() -> list[str]: return sorted(_REGISTRY)
