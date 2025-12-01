# 模型注册字典
_model_registry = {}

# 定义 register_model 装饰器
def register_model(fn):
    """Decorator to register a model function in the global model registry."""
    model_name = fn.__name__  # 获取函数名（即模型名称）
    _model_registry[model_name] = fn  # 存入全局字典
    return fn  # 返回原函数，保持装饰器功能

# 获取注册模型（一般在 timm.create_model() 里会调用）
def get_registered_model(name):
    """Retrieve a registered model by name."""
    if name in _model_registry:
        return _model_registry[name]
    else:
        raise ValueError(f"Model '{name}' not found in registry!")
