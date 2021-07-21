import brian2 as b2
from .base import AbstractConnection

class STDPConnection(AbstractConnection):

    def __init__(self, source, target, weights, name, connect_prob = None, parameters = None):
        super().__init__(source, target, name, weights, connect_prob)
        self.connection_type    = 'STDP'
        self.var_list           = ['w']
        if parameters:
            self.parameters     = parameters
        else:
            self.parameters     = {
                'gmax' : 1.0,
                'dApre' : 1.0,
                'taum' : 10 * b2.ms,
                'taupre' : 20 * b2.ms,
                'taupost' : 20 * b2.ms
            }
        
        self.init_connection()

    def init_connection(self):
        '''
        Initialize all STDP self.parameters.
        Weights are randomly initialized
        '''
        self.parameters['dApost'] = - self.parameters['dApre'] * self.parameters['taupre'] / self.parameters['taupost'] * 1.05
        self.parameters['dApost'] = self.parameters['dApost'] * self.parameters['gmax']
        self.parameters['dApre'] = self.parameters['dApre'] * self.parameters['gmax']
        

        synaptic_model = '''
                w : 1
                dApre/dt = - Apre / taupre : 1 (event-driven)
                dApost/dt = - Apost / taupost : 1 (event-driven)
        '''
        on_pre = '''
            ge_post += w
            Apre += dApre
            w = clip(w + Apost, 0, gmax)
        '''
        on_post = '''
            Apost += dApost
            w = clip(w + Apre, 0, gmax)
        '''
        
        self.synapses = b2.Synapses(source=self.source, target=self.target,
                                    model = synaptic_model, on_pre = on_pre,
                                    on_post = on_post)
        self.synapses.namespace.update(self.parameters)
        self.synapses.connect()
        self.synapses.w = 'rand() * gmax'
        
        
class DA_STDP(AbstractConnection):
    
    def __init__(self, source, target, name, weights, connect_prob, parameters = None):
        super().__init__(source, target, name, weights, connect_prob, )
        self.connection_type    = 'DA_STDP'
        self.weights            = weights
        self.var_list           = ['s', 'c', 'd']
        
        if parameters:
            self.parameters = parameters
        else:
            self.parameters = {
                'gmax' : .01,
                'dApre' : 0.01,
                'taum' : 10 * b2.ms,
                'taupre' : 20 * b2.ms,
                'taupost' : 20 * b2.ms,
                'tauc' : 1000 * b2.ms,
                'taud' : 200 * b2.ms,
                'taus' : 1 * b2.ms,
                'epsilon_dopa' : 5e-3
            }
            
            self.parameters['dApost'] = - self.parameters['dApre'] * self.parameters['taupre'] / self.parameters['taupost'] * 1.05
            self.parameters['dApost'] *= self.parameters['gmax']
            self.parameters['dApre'] *= self.parameters['gmax']
            
        
        self.init_connection()

    def init_connection(self):
         
        synaptic_model = '''
            mode: 1
            dc/dt = -c / tauc : 1 (clock-driven)
            dd/dt = -d / taud : 1 (clock-driven)
            ds/dt = mode * c * d / taus : 1 (clock-driven)
            dApre/dt = -Apre / taupre : 1 (event-driven)
            dApost/dt = -Apost / taupost : 1 (event-driven)
        '''
        
        on_pre='''
            ge += s
            Apre += dApre
            c = clip(c + mode * Apost, -gmax, gmax)
            s = clip(s + (1-mode) * Apost, -gmax, gmax)
        '''
        
        on_post='''
            Apost += dApost
            c = clip(c + mode * Apre, -gmax, gmax)
            s = clip(s + (1-mode) * Apre, -gmax, gmax)
        '''
        
        self.synapses = b2.Synapses(source=self.source, target=self.target,
                                    model = synaptic_model, on_pre = on_pre,
                                    on_post = on_post, method = 'euler')
        self.synapses.namespace.update(self.parameters)
        self.synapses.mode  = 0
        self.synapses.s     = 1e-10
        self.synapses.c     = 1e-10
        self.synapses.d     = 0
        
        
        self.synapses.mode  = 1
        


    
        

class Diehl_and_Cook_STDP(AbstractConnection):
    
    def __init__(self, source, target, weights, name, connect_prob = None, parameters = None):
        super().__init__(source, target, name, weights, connect_prob)
        self.connection_type    = 'STDP'
        self.var_list           = ['w']
        self.weights            = weights
        exp_ee_pre              = 0.2
        
        if parameters:
            self.parameters = parameters
        else:
            self.parameters = {
                'tc_pre_ee' : 20*b2.ms,
                'tc_post_1_ee' : 20*b2.ms,
                'tc_post_2_ee' : 40*b2.ms,
                'nu_ee_pre' :  0.0001,
                'nu_ee_post' : 0.01,
                'wmax_ee' : 1.0,
                'exp_ee_post' : exp_ee_pre
            }
            
        self.init_connection()
        
    def init_connection(self):
        '''
        Initialize all STDP self.parameters.
        Weights are randomly initialized
        '''
        
        synaptic_model = '''
            w                                      : 1
            post2before                            : 1
            dpre/dt   =   -pre/(tc_pre_ee)         : 1 (event-driven)
            dpost1/dt  = -post1/(tc_post_1_ee)     : 1 (event-driven)
            dpost2/dt  = -post2/(tc_post_2_ee)     : 1 (event-driven)
        '''
        
        on_pre = '''
            ge_post += w
            pre = 1. 
            w = clip(w + nu_ee_pre * post1, 0, wmax_ee)
        '''
        
        on_post = '''
            post2before = post2
            w = clip(w + nu_ee_post * pre * post2before, 0, wmax_ee)
            post1 = 1.
            post2 = 1.
        '''
        
        self.synapses = b2.Synapses(source=self.source, target=self.target,
                                    model = synaptic_model, on_pre = on_pre,
                                    on_post = on_post)
        self.synapses.namespace.update(self.parameters)
        self.synapses.connect()
        self.synapses.w = self.weights[self.synapses.i, self.synapses.j]
