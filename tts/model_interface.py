from abc import ABC, abstractmethod


class Model(ABC):
    @abstractmethod
    def synthesize(self, text: str):
        pass
