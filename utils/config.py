"""Simple config system.

We keep it tiny on purpose: YAML file -> nested dict -> a thin wrapper
that supports attribute access (`cfg.model.width`) and still behaves like
a dict (for checkpoint serialization).
"""
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Union
import copy
import yaml


class Config(dict):
    """Dict with attribute-style access. Nested dicts become Configs too."""

    def __init__(self, data: Dict[str, Any] | None = None) -> None:
        super().__init__()
        if data:
            for k, v in data.items():
                self[k] = self._wrap(v)

    @staticmethod
    def _wrap(value: Any) -> Any:
        if isinstance(value, dict):
            return Config(value)
        if isinstance(value, list):
            return [Config._wrap(v) for v in value]
        return value

    def __getattr__(self, item: str) -> Any:
        try:
            return self[item]
        except KeyError as e:
            raise AttributeError(item) from e

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = self._wrap(value)

    def get(self, key: str, default: Any = None) -> Any:  # type: ignore[override]
        return self[key] if key in self else default

    def to_dict(self) -> Dict[str, Any]:
        """Turn it back into a plain dict (for YAML dumping / checkpoints)."""
        out: Dict[str, Any] = {}
        for k, v in self.items():
            if isinstance(v, Config):
                out[k] = v.to_dict()
            elif isinstance(v, list):
                out[k] = [x.to_dict() if isinstance(x, Config) else x for x in v]
            else:
                out[k] = v
        return out

    def merge(self, overrides: Dict[str, Any]) -> "Config":
        """Return a new Config with `overrides` deep-merged on top."""
        merged = copy.deepcopy(self.to_dict())
        _deep_update(merged, overrides)
        return Config(merged)


def _deep_update(base: Dict[str, Any], new: Dict[str, Any]) -> None:
    for k, v in new.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_update(base[k], v)
        else:
            base[k] = v


def load_config(path: Union[str, Path]) -> Config:
    """Load a YAML file as a Config."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}
    return Config(data)
