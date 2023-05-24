import collections.abc
import os
import os.path
import typing

from ..tf import MaiaNet
from .stockfish_model import Stockfish, _sf_default_path


def list_maia_paths() -> typing.Dict[str, str]:
    models_dict = {}
    for p in os.listdir(os.path.join(os.path.dirname(__file__), "maia")):
        if p.endswith(".pb.gz"):
            models_dict[p.replace(".pb.gz", "")] = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "maia", p)
            )
    return models_dict


def list_maias() -> typing.List[str]:
    return sorted(list_maia_paths().keys())


def stockfish_path() -> str:
    return _sf_default_path


class _Models(collections.abc.Mapping):
    def __init__(self):
        self.model_list = list_maias() + ["stockfish"]
        self.model_paths = list_maia_paths()
        self.model_paths["stockfish"] = stockfish_path()
        self.loaded_models: typing.Dict[str, MaiaNet] = {}
        self.stockfish = Stockfish()
        self.load_gpu: typing.Union[None, int] = None

    def _get_memoized_model(self, name) -> MaiaNet:
        if name in self.model_list:
            try:
                return self.loaded_models[name]
            except:
                return self.loaded_models.setdefault(
                    name, MaiaNet(self.model_paths[name], gpu_id=self.load_gpu)
                )
        else:
            raise KeyError(f"{name} not found")

    def __getattr__(self, name):
        try:
            return self._get_memoized_model(name)
        except KeyError:
            raise AttributeError(
                f"{name} not a known model, known models are: {', '.join(self.model_list)}"
            )

    def __dir__(self) -> typing.List[str]:
        return self.model_list + list(super().__dir__())

    def __getitem__(self, key) -> MaiaNet:
        try:
            return self._get_memoized_model(key)
        except KeyError:
            raise KeyError(
                f"{key} not a known model, known models are: {', '.join(self.model_list)}"
            ) from None

    def __iter__(self) -> typing.Iterator[str]:
        for name in self.model_list:
            yield name

    def __len__(self) -> int:
        return len(self.model_list)


models = _Models()
