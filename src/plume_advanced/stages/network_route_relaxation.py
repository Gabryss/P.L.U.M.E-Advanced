"""Move coarse route vertices in the continuous routing cost field.

This bounded elastic-path proposal removes some lattice bias without adding
decorative noise. Junctions and topology stay fixed. Dense host checks screen
the proposal here; full network inspection still decides whether it survives.
"""

import numpy as np
from scipy.optimize import minimize


def sample_cost(field, x, y, points):
    """Bilinear value and exact within-cell XY derivative on a regular grid."""
    dx, dy = x[1] - x[0], y[1] - y[0]
    u = np.clip((points[:, 0] - x[0]) / dx, 0, len(x) - 1)
    v = np.clip((points[:, 1] - y[0]) / dy, 0, len(y) - 1)
    i = np.minimum(u.astype(int), len(x) - 2)
    j = np.minimum(v.astype(int), len(y) - 2)
    a, b = u - i, v - j
    f00, f10 = field[j, i], field[j, i + 1]
    f01, f11 = field[j + 1, i], field[j + 1, i + 1]
    value = (1-a)*(1-b)*f00 + a*(1-b)*f10 + (1-a)*b*f01 + a*b*f11
    gradient = np.column_stack((((1-b)*(f10-f00) + b*(f11-f01)) / dx,
                                ((1-a)*(f01-f00) + a*(f11-f10)) / dy))
    return value, gradient


def route_energy(points, field, x, y, stiffness):
    """Cost integral plus second-difference energy, with analytic gradient."""
    cost, derivative = sample_cost(field, x, y, points)
    delta = np.diff(points, axis=0)
    length = np.maximum(np.linalg.norm(delta, axis=1), 1e-9)
    mean = (cost[:-1] + cost[1:]) / 2
    gradient = np.zeros_like(points)
    tension = delta * (mean / length)[:, None]
    gradient[:-1] -= tension
    gradient[1:] += tension
    gradient[:-1] += .5 * length[:, None] * derivative[:-1]
    gradient[1:] += .5 * length[:, None] * derivative[1:]
    bends = np.diff(points, n=2, axis=0)
    gradient[:-2] += 2 * stiffness * bends
    gradient[1:-1] -= 4 * stiffness * bends
    gradient[2:] += 2 * stiffness * bends
    return float(length @ mean + stiffness * np.sum(bends*bends)), gradient


def relax_route(planner, points, layer):
    """At most 40 optimizer iterations and a capped line search, within 2 widths."""
    reference = np.asarray(points, float)
    if len(reference) < 4:
        return reference, dict(accepted=False, reason="short_route", iterations=0)
    field = planner.cost_fields[layer]
    finite = np.isfinite(field) & (field > 0)
    if not finite.any():
        return reference, dict(accepted=False, reason="invalid_cost_field", iterations=0)
    # A blocked host cell can carry NaN cost. Give it a finite penalty during
    # optimization; dense physical checks still prohibit entering that cell.
    scale = max(float(np.median(field[finite])), 1e-9)
    field = np.where(finite, field / scale, 100 * float(np.max(field[finite])) / scale)
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(reference, axis=0), axis=1))]
    step = max(float(np.mean(np.diff(arc))), 1e-9)
    stiffness = 2 * planner.width**2 / step**3
    # A tapered envelope keeps exact anchors and limits junction perturbation.
    reach = np.minimum(arc, arc[-1]-arc) / (4 * planner.width)
    bound = 2 * planner.width / np.sqrt(2) * np.minimum(reach, 1)
    lower = np.maximum(reference - bound[:, None], [planner.x[0], planner.y[0]])
    upper = np.minimum(reference + bound[:, None], [planner.x[-1], planner.y[-1]])

    def objective(flat):
        xy = reference.copy()
        xy[1:-1] = flat.reshape(-1, 2)
        value, gradient = route_energy(xy, field, planner.x, planner.y, stiffness)
        return value, gradient[1:-1].ravel()

    before = objective(reference[1:-1].ravel())[0]
    result = minimize(objective, reference[1:-1].ravel(), jac=True, method="L-BFGS-B",
                      bounds=list(zip(lower[1:-1].ravel(), upper[1:-1].ravel())),
                      options=dict(maxiter=40, maxfun=100, maxls=10, ftol=1e-8))
    candidate = reference.copy()
    candidate[1:-1] = result.x.reshape(-1, 2)
    after = objective(result.x)[0]
    report = dict(accepted=False, reason="no_improvement", iterations=int(result.nit),
                  evaluations=int(result.nfev), energy_before=before, energy_after=after)
    if not np.isfinite(candidate).all() or not np.isfinite(after) or after >= before - 1e-8:
        return reference, report
    # Sample every edge: moving viable vertices alone can cut through a thin
    # invalid strip. Keep the old route if any intermediate host test fails.
    dense = np.concatenate([np.linspace(a, b, max(2, int(np.ceil(np.linalg.norm(b-a)/.5))+1))[:-1]
                            for a, b in zip(candidate, candidate[1:])] + [candidate[-1:]])
    samples = planner.sample_host(dense)
    lengths = np.linalg.norm(np.diff(dense, axis=0), axis=1)
    if (not np.isfinite(samples).all() or np.any(samples[:, 1:] <= 0)
            or np.any(np.diff(samples[:, 0]) > planner.config.quality.maximum_uphill_grade * lengths + 1e-9)):
        report["reason"] = "host_constraint"
        return reference, report
    report.update(accepted=True, reason="lower_routing_energy",
                  maximum_displacement_m=float(np.linalg.norm(candidate-reference, axis=1).max()))
    return candidate, report
