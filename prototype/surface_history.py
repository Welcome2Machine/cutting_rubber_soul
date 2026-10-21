"""Diagnostic smooth material-surface chart; NOT a native IPC replacement.

One open C2 spline master and one positive-weight slave quadrature point.
Persistent active contact can cross spline knots without resetting its material
coordinate. Signed improved-max combinations, self-contact, endpoints, new
contact/recontact and a complete FE tangent are deliberately NOT implemented.
Reference parameters must be identical in both configurations. Spline geometry
differs from P1 boundary geometry and requires independent qualification.
"""
import numpy as np
from scipy.interpolate import CubicSpline
from numpy.polynomial import polynomial as poly


class SurfaceChart:
    def __init__(self, parameters):
        self.parameters = np.array(parameters, dtype=float, copy=True)
        if (self.parameters.ndim != 1 or len(self.parameters) < 3
                or not np.isfinite(self.parameters).all()
                or np.any(np.diff(self.parameters) <= 0)):
            raise ValueError('At least three finite ordered material parameters required')
        self.basis = CubicSpline(self.parameters, np.eye(len(self.parameters)),
                                 axis=0, bc_type='natural', extrapolate=False)

    def geometry(self, nodes):
        nodes = np.asarray(nodes, dtype=float)
        if nodes.shape != (len(self.parameters), 2) or not np.isfinite(nodes).all():
            raise ValueError('Finite master nodes matching the material chart required')
        return CubicSpline(self.parameters, nodes, axis=0,
                           bc_type='natural', extrapolate=False)

    def project(self, point, nodes, *, side=1):
        """Enumerate stationary projections of ALL cubic segments; fail on ties.

        Local Newton alone is insufficient when two projection branches coexist.
        Endpoints/degenerate minima are not silently interpreted as EV contact.
        """
        point = np.asarray(point, dtype=float)
        if point.shape != (2,) or not np.isfinite(point).all() or side not in (-1, 1):
            raise ValueError('Finite point and side +1/-1 required')
        curve = self.geometry(nodes)
        # Interior knots of a C2 chart are not constrained minima. Including
        # every knot would falsely classify a nearby stationary minimum as a
        # tied second branch when their distances differ only quadratically.
        candidates = [self.parameters[0], self.parameters[-1]]
        tol = 1e-10*max(1., float(np.ptp(self.parameters)))
        for i, h in enumerate(np.diff(self.parameters)):
            c = curve.c[:, i, :][::-1].copy()
            c[0] -= point
            polynomial = poly.polyadd(poly.polymul(c[:, 0], poly.polyder(c[:, 0])),
                                      poly.polymul(c[:, 1], poly.polyder(c[:, 1])))
            roots = poly.polyroots(polynomial)
            for root in roots:
                if abs(root.imag) < 1e-9*max(1., abs(root.real)) and -tol <= root.real <= h+tol:
                    candidates.append(float(self.parameters[i]+np.clip(root.real,0,h)))
        candidates = np.sort(np.array(candidates))
        candidates = candidates[np.r_[True, np.diff(candidates) > tol]]
        distance = np.linalg.norm(point-curve(candidates), axis=1)
        order = np.argsort(distance)
        u = float(candidates[order[0]])
        # Distinct equally close stationary branches make material identity
        # ambiguous; do not arbitrarily select a smaller parameter.
        close = order[np.abs(distance[order]-distance[order[0]]) < 1e-11]
        if len(close) > 1:
            raise ValueError('Ambiguous closest material projection')
        if u <= self.parameters[0]+tol or u >= self.parameters[-1]-tol:
            raise ValueError('Endpoint projection is outside persistent interior scope')
        c, v, a = curve(u), curve(u, 1), curve(u, 2)
        length = float(np.linalg.norm(v));r = point-c
        D = float(v@v-r@a)
        if length < 1e-12 or D <= 1e-10*length**2:
            raise ValueError('Degenerate/nonunique projection differential')
        if abs(r@v) > 1e-9*max(1., length):
            raise ValueError('Projection stationarity failed')
        tangent = v/length
        normal = side*np.array([-tangent[1], tangent[0]])
        N, B = self.basis(u), self.basis(u, 1)
        du = np.vstack((v, B[:, None]*r-N[:, None]*v))/D
        dg = np.vstack((normal, -N[:, None]*normal))
        return dict(parameter=u, gap=float(r@normal), projection=c, normal=normal,
                    tangent=tangent, speed=length, parameter_gradient=du,
                    gap_gradient=dg, projection_denominator=D, curve=curve)

    def arc_increment(self, nodes, start, end, *, order=12):
        """Signed CURRENT arc length between two persistent material labels.

        Returns the derivative with respect to master nodes at FIXED labels.
        Integration is split at knots, with positive Gauss weights per segment.
        """
        if not self.parameters[0] <= min(start,end) <= max(start,end) <= self.parameters[-1]:
            raise ValueError('Material arc lies outside chart')
        if order not in (8,12,20):raise ValueError('Qualified quadrature orders are 8,12,20')
        gradient = np.zeros_like(nodes, dtype=float)
        if start == end:return 0., gradient
        curve = self.geometry(nodes);sign = 1. if end > start else -1.
        lo,hi = min(start,end),max(start,end)
        boundaries = np.r_[lo,self.parameters[(self.parameters>lo)&(self.parameters<hi)],hi]
        xi,w = np.polynomial.legendre.leggauss(order);value = 0.
        for a,b in zip(boundaries[:-1],boundaries[1:]):
            q=(a+b)/2+(b-a)*xi/2;weights=sign*(b-a)*w/2
            v=curve(q,1);L=np.linalg.norm(v,axis=1)
            if np.min(L)<1e-12:raise ValueError('Degenerate surface arc')
            value+=float(weights@L)
            gradient+=np.einsum('q,qn,qi->ni',weights,self.basis(q,1),v/L[:,None])
        return value,gradient

    def differential(self, point, nodes, *, side=1):
        """Exact current projection Hessian, gap Hessian and transport metric.

        At a spline knot the third derivative is one-sided. The force is
        continuous, but a classical two-sided tangent need not exist there.
        This is not a finite-difference Jacobian or a PSD solver metric.
        """
        q=self.project(point,nodes,side=side);u=q['parameter'];curve=q['curve']
        v,a,j=curve(u,1),curve(u,2),curve(u,3);r=np.asarray(point)-curve(u)
        N,B,C=(self.basis(u,k) for k in (0,1,2));d=2*(len(nodes)+1)
        E=np.hstack([np.eye(2)]+[-z*np.eye(2) for z in N])
        V=np.hstack([np.zeros((2,2))]+[z*np.eye(2) for z in B])
        Acc=np.hstack([np.zeros((2,2))]+[z*np.eye(2) for z in C])
        du=q['parameter_gradient'].ravel();D=q['projection_denominator'];L=q['speed']
        fzu=E.T@a+Acc.T@r-2*V.T@v
        hu=(E.T@V+V.T@E+np.outer(fzu,du)+np.outer(du,fzu)
            +(r@j-3*v@a)*np.outer(du,du))/D
        dv=V+np.outer(a,du);dr=E-np.outer(v,du);da=Acc+np.outer(j,du)
        t=q['tangent'];n=q['normal'];rot=np.array([[0.,-1.],[1.,0.]])
        dn=side*rot@(np.eye(2)-np.outer(t,t))@dv/L
        hg=E.T@dn-np.outer(V.T@n,du)
        metric=D/L;dm=(2*v@dv-a@dr-r@da)/L-D/L**2*(t@dv)
        beta=metric*du;dbeta=metric*hu+np.outer(du,dm)
        q.update(parameter_hessian=hu,gap_hessian=hg,transport_metric=metric,
            metric_gradient=dm,transport_covector=beta,transport_jacobian=dbeta)
        return q


def surface_contact(chart, current, previous, *, weight, dhat, stiffness, mu, epsilon,
                    side=1, order=12):
    """Objective, work-conjugate chart diagnostic with analytic FORCE gradient.

    DOF order: slave point, all master nodes. Normal capacity uses the current
    barrier. Friction acts conjugately to surface arc migration, so at finite
    curved gaps it is not identical to a pointwise tangential Coulomb force.
    This distinction is reported, not concealed as an equivalent native kernel.
    No approximate Jacobian is offered to a production FE solver.
    """
    from .curve_contact import physical_barrier, friction_mollifier
    x,old=np.asarray(current,float),np.asarray(previous,float)
    if x.shape != old.shape or x.shape != (len(chart.parameters)+1,2):
        raise ValueError('Matching slave/master configurations required')
    if not all(np.isfinite(v) for v in (weight,dhat,stiffness,mu,epsilon)) or min(
            weight,dhat,stiffness,epsilon)<=0 or mu<0:
        raise ValueError('Positive weight/scales and nonnegative friction required')
    now=chart.project(x[0],x[1:],side=side)
    prior=chart.project(old[0],old[1:],side=side)
    if not (0<now['gap']<dhat and 0<prior['gap']<dhat):
        raise ValueError('Activation/detachment/recontact requires event history')
    slip,master_gradient=chart.arc_increment(x[1:],prior['parameter'],now['parameter'],order=order)
    ds=now['speed']*now['parameter_gradient'];ds[1:]+=master_gradient
    dg=now['gap_gradient']
    b,bp,_=(float(z[0]) for z in physical_barrier(np.array([now['gap']]),dhat,stiffness))
    f0,f1,_=(float(z[0]) for z in friction_mollifier(np.array([slip]),epsilon))
    N=-bp;ng=weight*bp*dg;fg=weight*mu*N*f1*ds
    return dict(slip_mm=float(slip),old_parameter=prior['parameter'],parameter=now['parameter'],
        gap_mm=now['gap'],normal_capacity_n=weight*N,gradient=ng+fg,
        normal_gradient=ng,friction_gradient=fg,slip_gradient=ds,
        gap_gradient=dg,normal_energy_nmm=weight*b,
        friction_potential_unit=f0,friction_dissipation_nmm=float(weight*mu*N*f1*slip),
        slave_tangential_metric=float(ds[0]@now['tangent']),
        scope='Persistent positive-weight smooth-chart diagnostic; no complete FE tangent, native equivalence, event handling or self-contact qualification')
