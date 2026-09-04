from __future__ import annotations

import abc
import dataclasses
from typing import TYPE_CHECKING

import firedrake as fd
from ufl import div, dot, dx, inner, nabla_grad

if TYPE_CHECKING:
    from ..flow import FlowConfig

__all__ = ["SUPG", "GLS", "ns_stabilization"]


@dataclasses.dataclass
class NavierStokesStabilization(metaclass=abc.ABCMeta):
    """Base class (and "no stabilization" implementation) for Navier-Stokes stabilization schemes.

    Bundles everything the residual-based stabilization terms need from the
    calling solver; subclasses build the actual UFL stabilization forms in
    :meth:`stabilize`. This base class implements the identity map, i.e. the
    unstabilized Galerkin formulation (the ``"none"``/``"linearized_none"``
    options in ``ns_stabilization``).

    Attributes:
        flow: Flow configuration (mesh, viscosity, stress/strain operators).
        q_trial: ``(u, p)`` pair of trial functions on the mixed space.
        q_test: ``(v, s)`` pair of test functions on the mixed space.
        wind: Velocity field used as the convective wind (the extrapolated
            velocity for the nonlinear solvers, or the base flow for the
            linearized ones).
        dt: Timestep (float or ``fd.Constant``); enters the stabilization
            parameter ``tau_M`` when given.
        u_t: UFL expression for the time derivative estimate (the BDF term),
            included in the strong-form residual when given.
        f: Body forcing, subtracted from the strong-form residual when given.
    """

    flow: FlowConfig
    q_trial: tuple[fd.Function, fd.Function]
    q_test: tuple[fd.Function, fd.Function]
    wind: fd.Function
    dt: float | fd.Constant = None
    u_t: fd.Function = None
    f: fd.Function = None

    def stabilize(self, weak_form):
        """Return the weak form unchanged (no stabilization terms).

        Args:
            weak_form: The unstabilized UFL weak form.

        Returns:
            The same weak form, unmodified.
        """
        # By default, no stabilization
        return weak_form


class UpwindNSStabilization(NavierStokesStabilization):
    """Residual-based upwind stabilization for the (unlinearized) Navier-Stokes equations.

    Shared machinery for SUPG and GLS: computes the strong-form momentum
    residual ``Lu`` with the given wind, the stabilization parameters
    ``tau_M``/``tau_C`` from the element size, wind magnitude, viscosity and
    (optionally) timestep, and adds ``tau_M <Lu, Lv>`` plus a least-squares
    incompressibility (LSIC) term ``tau_C <div u, div v>`` to the weak form.
    Subclasses define ``Lv``, the residual operator applied to the test
    functions.

    Attributes (inherited from :class:`NavierStokesStabilization`) supply the
    trial/test functions, wind, timestep, time derivative and forcing.
    """

    @property
    def h(self):
        """Mesh cell size (UFL ``CellSize``), the length scale in ``tau_M``."""
        return fd.CellSize(self.flow.mesh)

    @property
    def Lu(self):
        """Strong-form momentum residual of the trial functions.

        ``w . grad(u) - div(sigma(u,p))``, plus the time-derivative estimate
        ``u_t`` and minus the forcing ``f`` when those are supplied.

        Returns:
            UFL expression for ``Lu`` (the momentum residual).
        """
        (u, p) = self.q_trial

        w = self.wind
        sigma = self.flow.sigma

        Lu = dot(w, nabla_grad(u)) - div(sigma(u, p))

        if self.u_t is not None:
            Lu += self.u_t

        if self.f is not None:
            Lu -= self.f

        return Lu

    @abc.abstractproperty
    def Lv(self):
        """Residual operator applied to the test functions (subclass-specific).

        Returns:
            UFL expression for ``Lv``, paired with ``Lu`` in the
            stabilization inner product.
        """
        # Test function form for the stabilization term
        pass

    @property
    def tau_M(self):
        """Stabilization parameter for the momentum residual.

        ``tau_M = (4|w|^2/h^2 + 9 (4 nu / h^2)^2 [+ 4/dt^2])^(-1/2)``, i.e.
        the inverse squared "elemental" advective/diffusive/temporal rates.
        Based on:
        https://github.com/florianwechsung/alfi/blob/master/alfi/stabilisation.py

        Returns:
            UFL expression for ``tau_M``.
        """
        # Stabilization constant for momentum residual
        #
        # Based on:
        # https://github.com/florianwechsung/alfi/blob/master/alfi/stabilisation.py

        w = self.wind
        h = self.h
        nu = self.flow.nu

        denom_sq = 4.0 * dot(w, w) / (h**2) + 9.0 * (4.0 * nu / h**2) ** 2

        if self.dt is not None:
            denom_sq += 4.0 / (self.dt**2)

        return denom_sq ** (-0.5)

    @property
    def tau_C(self):
        """Stabilization parameter for the continuity residual, ``h^2 / tau_M``.

        Returns:
            UFL expression for ``tau_C``.
        """
        # Stabilization constant for continuity residual
        h = self.h
        return h**2 / self.tau_M

    @property
    def momentum_stabilization(self):
        """Momentum stabilization term ``tau_M <Lu, Lv> dx``.

        Returns:
            UFL form to be added to the weak form.
        """
        return self.tau_M * inner(self.Lu, self.Lv) * dx

    @property
    def lsic_stabilization(self):
        """Least-squares incompressibility (LSIC) term ``tau_C <div u, div v> dx``.

        Returns:
            UFL form to be added to the weak form.
        """
        (u, _) = self.q_trial
        (v, _) = self.q_test
        return self.tau_C * inner(div(u), div(v)) * dx

    def stabilize(self, weak_form):
        """Add the momentum and LSIC stabilization terms to a weak form.

        Args:
            weak_form: The unstabilized UFL weak form.

        Returns:
            The weak form with ``momentum_stabilization`` and
            ``lsic_stabilization`` appended.
        """
        weak_form += self.momentum_stabilization
        weak_form += self.lsic_stabilization
        return weak_form


class SUPG(UpwindNSStabilization):
    """Streamline-Upwind/Petrov-Galerkin stabilization.

    Uses only the streamline derivative ``w . grad(v)`` of the velocity test
    function as the residual operator ``Lv``, so the stabilization acts
    along the flow direction.
    """

    @property
    def Lv(self):
        """Streamline derivative of the velocity test function, ``w . grad(v)``.

        Returns:
            UFL expression for ``Lv``.
        """
        (v, _) = self.q_test
        w = self.wind
        return dot(w, nabla_grad(v))


class GLS(UpwindNSStabilization):
    """Galerkin least-squares stabilization.

    Uses the full momentum operator applied to the test functions,
    ``Lv = w . grad(v) - div(sigma(v, s))``, so the least-squares term
    minimizes the complete momentum residual (not just its streamline
    component, as SUPG does).
    """

    @property
    def Lv(self):
        """Full momentum operator applied to the test functions.

        ``w . grad(v) - div(sigma(v, s))``.

        Returns:
            UFL expression for ``Lv``.
        """
        (v, s) = self.q_test
        w = self.wind
        sigma = self.flow.sigma
        return dot(w, nabla_grad(v)) - div(sigma(v, s))


class LinearizedNSStabilization(UpwindNSStabilization):
    """Residual-based stabilization for the Navier-Stokes equations linearized about a base flow.

    Identical machinery to :class:`UpwindNSStabilization`, except the
    strong-form momentum residual ``Lu`` uses the linearized convective
    operator ``uB . grad(u) + u . grad(uB)``, where the base flow ``uB`` is
    supplied as the ``wind``.
    """

    @property
    def Lu(self):
        """Strong-form momentum residual of the linearized operator.

        ``uB . grad(u) + u . grad(uB) - div(sigma(u,p))``, plus the
        time-derivative estimate ``u_t`` and minus the forcing ``f`` when
        those are supplied.

        Returns:
            UFL expression for the linearized momentum residual ``Lu``.
        """
        (u, p) = self.q_trial

        uB = self.wind
        sigma = self.flow.sigma

        Lu = dot(uB, nabla_grad(u)) + dot(u, nabla_grad(uB)) - div(sigma(u, p))

        if self.u_t is not None:
            Lu += self.u_t

        if self.f is not None:
            Lu -= self.f

        return Lu


class LinearizedSUPG(LinearizedNSStabilization):
    """Streamline-upwind stabilization of the base-flow-linearized equations.

    The SUPG residual operator about the base flow: both linearized
    convective terms applied to the test function.
    """

    @property
    def Lv(self):
        """Linearized streamline derivative of the test function.

        ``uB . grad(v) + v . grad(uB)``.

        Returns:
            UFL expression for ``Lv``.
        """
        (v, _) = self.q_test
        uB = self.wind
        return dot(uB, nabla_grad(v)) + dot(v, nabla_grad(uB))


class LinearizedGLS(LinearizedNSStabilization):
    """Galerkin least-squares stabilization of the base-flow-linearized equations.

    The full linearized momentum operator applied to the test functions
    (linearized convective terms plus the viscous/pressure operator).
    """

    @property
    def Lv(self):
        """Full linearized momentum operator applied to the test functions.

        ``uB . grad(v) + v . grad(uB) - div(sigma(v, s))``.

        Returns:
            UFL expression for ``Lv``.
        """
        (v, s) = self.q_test
        uB = self.wind
        sigma = self.flow.sigma
        return dot(uB, nabla_grad(v)) + dot(v, nabla_grad(uB)) - div(sigma(v, s))


ns_stabilization = {
    "none": NavierStokesStabilization,
    "supg": SUPG,
    "gls": GLS,
    "linearized_none": NavierStokesStabilization,
    "linearized_supg": LinearizedSUPG,
    "linearized_gls": LinearizedGLS,
}
