class Equation:

    def __init__(self, params):
        # init stuff
        self.params = params

    def rhs(self, state, control):
        pass
        # calculate linear and non-linear terms


class SplitEquation(Equation):
    def __init__(self, params):
        super().__init__(params)
    
    def linear_terms(self, state, control): 
        pass

    def nonlinear_terms(self, state, control): 
        pass
    
    def rhs(self, state, control):
        return self.linear_terms(state, control) + self.nonlinear_terms(state, control)
    
    def forcing(self):
        pass
        
class IMEXEquation(SplitEquation):
    def __init__(self, params):
        super().__init__(params)

    def implicit_timestep(self, state):
        raise NotImplementedError

