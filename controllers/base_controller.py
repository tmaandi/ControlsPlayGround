from abc import ABC, abstractmethod

class BaseController(ABC):
    @abstractmethod
    def update(self, *args, **kwargs):
        pass

    @abstractmethod
    def reset(self):
        pass
