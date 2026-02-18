import json
import os
from typing import Dict, Any

class FeatureSet:
    def __init__(self, name: str, features: Dict[str, Dict[str, Any]] = None):
        self.name = name
        self.features = features or {}

    def add_feature(self, feat_name: str, params: Dict[str, Any]):
        self.features[feat_name] = params

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "features": self.features
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FeatureSet':
        return cls(name=data["name"], features=data["features"])

    def save(self, directory: str = "feature_sets"):
        os.makedirs(directory, exist_ok=True)
        file_path = os.path.join(directory, f"{self.name}.json")
        with open(file_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=4)

    @classmethod
    def load(cls, name: str, directory: str = "feature_sets") -> 'FeatureSet':
        file_path = os.path.join(directory, f"{name}.json")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Feature set {name} not found.")
        
        with open(file_path, 'r') as f:
            data = json.load(f)
            return cls.from_dict(data)

    @staticmethod
    def list_available(directory: str = "feature_sets"):
        if not os.path.exists(directory):
            return []
        return [f.replace(".json", "") for f in os.listdir(directory) if f.endswith(".json")]
