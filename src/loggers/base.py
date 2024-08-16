from abc import ABC


class Logger(ABC):
    def init_logger(self):
        pass

    def log_params(self):
        pass

    def init_experiment(self):
        pass

    def log_artifact(self):
        pass

    def log_metrics(self):
        pass

    def set_tags(self):
        pass

    def __repr__(self) -> str:
        return self.__class__.__name__
