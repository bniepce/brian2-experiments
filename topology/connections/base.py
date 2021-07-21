from abc import ABC, abstractmethod

class AbstractConnection(ABC):
    
    def __init__(self, source, target, name, weights = None, connect_prob = None):
        super().__init__()
        self.source         = source.neurons
        self.target         = target.neurons
        self.weights        = weights
        self.connect_prob   = connect_prob
        
        self.name           = name
        self.var_list       = []
        
    @abstractmethod
    def init_connection(self):
        raise NotImplementedError