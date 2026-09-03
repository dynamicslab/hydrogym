class Equation:
    """Base class for a PDE right-hand-side defined on a state vector.

    Subclasses implement the full right-hand side in :meth:`rhs`, either
    directly or by splitting it into linear and nonlinear parts (see
    :class:`SplitEquation`).
    """

    def __init__(self, params):
        """Store the equation parameters.

        Args:
            params: Solver/equation parameters (interpreted by subclasses).
        """
        # init stuff
        self.params = params

    def rhs(self, state, control):
        """Evaluate the right-hand side of the equation.

        Args:
            state: Current state vector.
            control: Control (actuation) input applied to the flow.

        Returns:
            Time derivative of the state. Not implemented in the base class.
        """
        pass
        # calculate linear and non-linear terms


class SplitEquation(Equation):
    """Equation whose right-hand side is split into linear and nonlinear terms.

    The split exists so implicit-explicit (IMEX) time integrators can treat
    the linear (stiff, typically viscous) term implicitly via
    :meth:`implicit_timestep` in subclasses, and the nonlinear term explicitly.
    """

    def __init__(self, params):
        """See :class:`Equation`."""
        super().__init__(params)

    def linear_terms(self, state, control):
        """Evaluate the linear (implicitly treated) term of the equation.

        Args:
            state: Current state vector.
            control: Control (actuation) input.

        Returns:
            Linear contribution to the state derivative. Not implemented here.
        """
        pass

    def nonlinear_terms(self, state, control):
        """Evaluate the nonlinear (explicitly treated) term of the equation.

        Args:
            state: Current state vector.
            control: Control (actuation) input.

        Returns:
            Nonlinear contribution to the state derivative. Not implemented here.
        """
        pass

    def rhs(self, state, control):
        """Evaluate the full right-hand side as linear + nonlinear terms.

        Args:
            state: Current state vector.
            control: Control (actuation) input.

        Returns:
            Sum of the linear and nonlinear contributions.
        """
        return self.linear_terms(state, control) + self.nonlinear_terms(state, control)

    def forcing(self):
        """Evaluate any external forcing added to the right-hand side.

        Not implemented in the base class.
        """
        pass


class IMEXEquation(SplitEquation):
    """Split equation supporting an implicit linear timestep.

    Intended for IMEX schemes: the linear term is advanced implicitly via
    :meth:`implicit_timestep` while the nonlinear term is handled explicitly
    by the integrator.
    """

    def __init__(self, params):
        """See :class:`SplitEquation`."""
        super().__init__(params)

    def implicit_timestep(self, state):
        """Advance the linear part of the equation implicitly by one timestep.

        Args:
            state: Current state vector.

        Raises:
            NotImplementedError: Always; subclasses must define the implicit solve.
        """
        raise NotImplementedError
