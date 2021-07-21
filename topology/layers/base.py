from abc import ABC, abstractmethod


class AbstractLayer(ABC):
    '''
    Abstract base class for neuron building.
    '''
    def __init__(self, n_neurons : int, name : str):
        self.name           = name
        self.n_neurons      = n_neurons
        self.var_list       = []
        
    @abstractmethod
    def init_layer(self):
        '''
        Set neuron random parameters within typical ranges.
        '''
        raise NotImplementedError