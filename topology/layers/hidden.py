from .base import AbstractLayer
import brian2 as b2
import random
import numpy as np


class HiddenLayer(AbstractLayer):
    '''
    Abstract base class for neuron building.
    '''
    def __init__(self, n_neurons : int, name : str):
        super().__init__(n_neurons, name)
        self.var_list       = ['v']
        
    def init_layer(self):
        '''
        Set neuron random parameters within typical ranges.
        '''
        raise NotImplementedError
    

class LIF(HiddenLayer):
    '''
    Implementation of the LIF neuron model
    '''
    def __init__(self, n_neurons : int, name : str):
        super().__init__(n_neurons, name)
        self.neuron_type = 'LIF'
        # self.parameters = {
        #     'v_rest' : (-55. + random.randint(0, 15)) * b2.mV,
        #     'v_reset' : None,
        #     'v_th' : random.randint(-50, +5) * b2.mV,
        #     'R' : random.randint(2, 15) * b2.Mohm,
        #     'tau_m' : random.randint(2, 30) * b2.ms,
        #     'tau_ref' : random.randint(1, 7) * b2.ms
        # }
        # self.parameters['v_reset'] = self.parameters['v_rest'] + random.randint(-10, +10) * b2.mV
        
        self.parameters = {
            'v_rest': -70 * b2.mV,
            'v_reset' : -65 * b2.mV,
            'v_th' : -50 * b2.mV,
            'R' : 2 * b2.Mohm,
            'tau_m' : 10 * b2.ms,
            'tau_ref' : 5 * b2.ms
            
        }
        
        self.init_layer()

    def init_layer(self):
        '''
        Initialize all LIF parameters as well as model equation.
        The membrane threshold and potential are both initialized randomly for
        each neuron in the group.
        '''
        
        lif_equation = '''
            dv/dt = (ge * (v - v_rest)) / tau_m : volt (unless refractory)
            dge/dt = - ge / tau_m : 1
            vth : volt
        '''
        
        self.neurons = b2.NeuronGroup(self.n_neurons, model = lif_equation, 
                                    reset = 'v = v_reset', threshold = 'v > vth',
                                    refractory = self.parameters['tau_ref'], method = 'euler')
        self.neurons.namespace.update(self.parameters)
        self.neurons.v = 'v_rest + rand() * (v_th - v_rest)'
        self.neurons.vth = 'v_th + rand()*5*mV'
        
class Izhikevich(HiddenLayer):
    '''
    Implementation of the Izhikevich neuron model

    TODO : Thalamic input
    '''
    def __init__(self, name : str, n_neurons : int = 10, model_type : str = 'RS'):
        super().__init__(n_neurons, name)
        self.neuron_type = 'Izhikevich'
        self.model_type = model_type
        self.parameters = {
            'RS' : {'a': 0.02 / b2.ms, 'b': 0.20 / b2.ms, 'c': -65.0 * b2.mV, 'd': 8.00 * (b2.mV / b2.ms)},
            'IB' : {'a': 0.02 / b2.ms, 'b': 0.20 / b2.ms, 'c': -55.0 * b2.mV, 'd': 4.00 * (b2.mV / b2.ms)},
            'CH' : {'a': 0.02 / b2.ms, 'b': 0.20 / b2.ms, 'c': -50.0 * b2.mV, 'd': 2.00 * (b2.mV / b2.ms)},
            'FS' : {'a': 0.10 / b2.ms, 'b': 0.20 / b2.ms, 'c': -65.0 * b2.mV, 'd': 2.00 * (b2.mV / b2.ms)},
            'TC' : {'a': 0.02 / b2.ms, 'b': 0.25 / b2.ms, 'c': -65.0 * b2.mV, 'd': 0.05 * (b2.mV / b2.ms)},
            'RZ' : {'a': 0.10 / b2.ms, 'b': 0.25 / b2.ms, 'c': -65.0 * b2.mV, 'd': 2.00 * (b2.mV / b2.ms)},
            'LTS' : {'a': 0.02 / b2.ms, 'b': 0.25 / b2.ms, 'c': -65.0 * b2.mV, 'd': 2.00 * (b2.mV / b2.ms)}
        }[self.model_type]
        self.init_layer()
        
    def init_layer(self):
        '''
        Initialize all LIF parameters as well as model equation.
        The membrane threshold and potential are both initialized randomly for
        each neuron in the group.
        '''
            
        eqs = '''
            dv/dt = (0.04/ms/mV) * v**2 + (5/ms) * v + 140*mV/ms - u + I: volt
            du/dt = a*(b*v-u) : volt/second
            I : volt/second
        '''
        
        reset ='''
            v = c
            u += d
        '''
        
        self.neurons = b2.NeuronGroup(self.n_neurons, model = eqs, 
                                    reset = reset, threshold = 'v > 30 * mV', method = 'euler')
        self.neurons.namespace.update(self.parameters)
        self.neurons.v = self.parameters['c']
        self.neurons.u = self.parameters['b'] * self.neurons.v
        self.neurons.I = 0 * b2.mV / b2.ms
        #self.neurons.vth = '-55*mV + rand()*15*mV'
        
        
class Diehl_and_Cook_LIF(HiddenLayer):
    '''
    Implementation of the LIF neuron model used in Diehl and Cook's paper.
    '''
    def __init__(self, n_neurons : int, name : str, mode : str = 'train', weight_path = None):
        super().__init__(n_neurons, name)
        self.neuron_type    = 'LIF'
        self.mode           = mode
        self.weight_path    = weight_path
    
        self.parameters     = {
            'v_rest' : -65. * b2.mV,
            'v_reset' : -65. * b2.mV,
            'refrac' : 5. * b2.ms,
            'offset' : 20.0 * b2.mV,
            'vth' : -52. * b2.mV
        }
        self.init_layer()

    def init_layer(self):
        '''
        Initialize all LIF parameters as well as model equation.
        The membrane threshold and potential are both initialized randomly for
        each neuron in the group.
        '''
        
        lif_equation = '''
            dv/dt = ((v_rest - v) + I_syn / nS) / (100*ms)  : volt (unless refractory)
            I_syn = ge * nS * -v                            : amp
            dge/dt = -ge/(1.0*ms)                           : 1
            dtimer/dt = 0.05                                : second
        '''

        if self.mode == 'train':
            lif_equation += '\n  dtheta/dt = -theta / (tc_theta)  : volt'
            self.parameters['tc_theta'] = 1e7 * b2.ms
            self.parameters['theta_plus'] = 0.05 * b2.mV
            self.parameters['scr'] = 'v = v_reset; theta += theta_plus; timer = 0*ms'
        else:
            lif_equation += '\n  theta :volt'
            self.parameters['scr'] = 'v = v_reset; timer = 0*ms'
        
        v_th = '(v>(theta - offset + vth)) and (timer>refrac)'
        self.neurons = b2.NeuronGroup(self.n_neurons, model = lif_equation, 
                                    threshold = v_th, 
                                    refractory = self.parameters['refrac'],
                                    reset = self.parameters['scr'],
                                    method = 'euler')         
        self.neurons.namespace.update(self.parameters)
    
        if self.mode == 'test' or self.weight_path:
            self.neurons.theta = np.load(weight_path)
        else:
            self.neurons.theta = np.ones((self.n_neurons)) * 20.0 * b2.mV
        
        self.neurons.v = self.parameters['v_rest']