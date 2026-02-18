import importlib
import pkgutil
import inspect
from .base import Feature
import src.features

def load_features():
    features = {}
    path = src.features.__path__
    prefix = src.features.__name__ + "."

    for _, name, _ in pkgutil.iter_modules(path, prefix):
        try:
            module = importlib.import_module(name)
            for _, obj in inspect.getmembers(module):
                if inspect.isclass(obj) and issubclass(obj, Feature) and obj is not Feature:
                    instance = obj()
                    features[instance.name] = instance
        except Exception as e:
            print(f"Error loading feature {name}: {e}")
            
    return features
