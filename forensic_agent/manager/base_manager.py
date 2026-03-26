from abc import ABC, abstractmethod


class BaseManager(ABC):
    def run(self, *args, **kwargs):
        pass
