import brian2 as b2
from .base import AbstractLayer
from utils import spike_train_to_times
import numpy as np
import random

class PoissonInput(AbstractLayer):
    '''
    Creates Poisson distributed spike trains
    '''
    def __init__(self, n_neurons : int, name : str, freq : int):
        super().__init__(n_neurons, name)
        self.freq       = freq * b2.Hz
        self.init_layer()
        
    def init_layer(self):
        self.neurons = b2.PoissonGroup(self.n_neurons, rates=self.freq)
        
        
class ManualSpikeInput(AbstractLayer):
    '''
    Uses a spike train array to get spike times and build a SpikeGeneratorGroup.
    '''
    def __init__(self, spike_trains, dt, n_neurons : int, name : str):
        super().__init__(n_neurons, name)
        self.dt             = dt
        self.spike_trains   = spike_trains
        self.init_layer()

    def get_input_stream(self, spike_times):
        '''Returns a brian2 SpikeGeneratorGroup representing input spike times.
        
        Parameters
        ----------
        n_neurons : int
            Number of neurons in the group
        spike_times : np.array
            Array containing spike times. If quantity is not specified the times are set in ms.

        '''
        indices = [self.n_neurons - 1 for i in range(len(spike_times))]
        G = b2.SpikeGeneratorGroup(self.n_neurons, indices, spike_times)
        return G
            
    def init_layer(self):
        self.spike_times = spike_train_to_times(self.spike_trains, dt=self.dt)
        self.neurons = self.get_input_stream(self.spike_times)

        
class CurrentBasedLIF(AbstractLayer):

    def __init__(self, input_current : np.array, n_neurons : int, name : str):
        super().__init__(n_neurons, name)
        self.neuron_type    = 'CurrentBasedLIF'
        self.var_list       = ['v', 'vth', 'I']
        self.input_current  = b2.TimedArray(input_current, dt=1*b2.ms)
        self.parameters = {
            'v_rest' : 0 * b2.mV,
            'tau_m' : 20 * b2.ms,
            'tau_ref' : 5 * b2.ms,
            'R' : random.randint(2, 15) * b2.Mohm
        }
        self.init_layer()

    def init_layer(self):
        '''
        Initialize all LIF parameters as well as model equation.
        The membrane threshold and potential are both initialized randomly for
        each neuron in the group.
        '''
        
        lif_equation = '''
            dv/dt = (g * (v - v_rest) + R * I) / tau_m : volt
            dg/dt = - g / tau_m : 1
            I = input_current(t) : amp
            vth = 30 * mV * i : volt
        '''
        
        self.neurons = b2.NeuronGroup(self.n_neurons, model = lif_equation, 
                                    reset = 'v = v_rest', threshold = 'v > vth',
                                    refractory = self.parameters['tau_ref'], method = 'euler')
        self.neurons.namespace['input_current'] = self.input_current
        self.neurons.namespace.update(self.parameters)
        self.neurons.v = 'v_rest + rand() * mV'