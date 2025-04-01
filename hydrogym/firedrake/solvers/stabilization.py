from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import TYPE_CHECKING

from hydrogym.utils import DependencyNotInstalled

try:
    import firedrake as fd
    from ufl import div, dot, dx, inner, nabla_grad
except ImportError as e:
    raise DependencyNotInstalled(
        "Firedrake is not installed, consult `https://www.firedrakeproject.org/install.html` for installation instructions."  # noqa: E501
    ) from e

if TYPE_CHECKING:
    from hydrogym.firedrake.flow import FlowConfig

__all__ = ["SUPG", "GLS", "ns_stabilization"]


@dataclass
class NavierStokesStabilization(metaclass=abc.ABCMeta):
    flow: FlowConfig
    q_trial: tuple[fd.Function, fd.Function]
    q_test: tuple[fd.Function, fd.Function]
    wind: fd.Function
    dt: float | fd.Constant = None
    u_t: fd.Function = None
    f: fd.Function = None

    def stabilize(self, weak_form):
        # By default, no stabilization
        return weak_form


class UpwindNSStabilization(NavierStokesStabilization):
    @property
    def h(self):
        return fd.CellSize(self.flow.mesh)

    @property
    def Lu(self):
        (u, p) = self.q_trial

        w = self.wind
        sigma = self.flow.sigma

        Lu = dot(w, nabla_grad(u)) - div(sigma(u, p))

        if self.u_t is not None:
            Lu += self.u_t

        if self.f is not None:
            Lu -= self.f

        return Lu

    @property
    @abc.abstractmethod
    def Lv(self):
        # Test function form for the stabilization term
        pass

    @property
    def tau_M(self):
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
        # Stabilization constant for continuity residual
        h = self.h
        return h**2 / self.tau_M

    @property
    def momentum_stabilization(self):
        return self.tau_M * inner(self.Lu, self.Lv) * dx

    @property
    def lsic_stabilization(self):
        (u, _) = self.q_trial
        (v, _) = self.q_test
        return self.tau_C * inner(div(u), div(v)) * dx

    def stabilize(self, weak_form):
        weak_form += self.momentum_stabilization
        weak_form += self.lsic_stabilization
        return weak_form


class SUPG(UpwindNSStabilization):
    @property
    def Lv(self):
        (v, _) = self.q_test
        w = self.wind
        return dot(w, nabla_grad(v))


class GLS(UpwindNSStabilization):
    @property
    def Lv(self):
        (v, s) = self.q_test
        w = self.wind
        sigma = self.flow.sigma
        return dot(w, nabla_grad(v)) - div(sigma(v, s))


class LinearizedNSStabilization(UpwindNSStabilization):
    @property
    def Lu(self):
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
    @property
    def Lv(self):
        (v, _) = self.q_test
        uB = self.wind
        return dot(uB, nabla_grad(v)) + dot(v, nabla_grad(uB))


class LinearizedGLS(LinearizedNSStabilization):
    @property
    def Lv(self):
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
