from abc import ABC, abstractmethod

class BasePlant(ABC):
    @abstractmethod
    def get_state(self):
        pass

    @abstractmethod
    def set_state(self, state):
        pass

    @abstractmethod
    def update(self, input, dt):
        pass
