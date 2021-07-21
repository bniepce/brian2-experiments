import brian2 as b2
from .base import AbstractConnection


class ForwardConnection(AbstractConnection):

    def __init__(self, source, target, name):
        
        super().__init__(source, target, name)
        self.connection_type    = 'Forward'
        self.init_connection()

    def init_connection(self):
        '''
        Initialize all STDP self.parameters.
        Weights are randomly initialized
        '''
        self.synapses = b2.Synapses(source=self.source, target=self.target, 
                                model='''w : volt''',
                                on_pre='v += w')
        self.synapses.connect()