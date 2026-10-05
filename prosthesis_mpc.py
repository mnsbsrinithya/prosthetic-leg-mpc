"""
Model Predictive Control of a Two-Link Robotic Prosthetic Leg (Lagrangian dynamics)
==================================================================================
Based on: Ma, Zhang, Xu, "Robotic Leg Prosthesis: A Survey From Dynamic Model to
Adaptive Control for Gait Coordination", IEEE TNSRE vol. 32, 2024.

What is implemented (paper -> code)
-----------------------------------
 Eq.(3)-(8)  Lagrangian model   M(q)q'' + C(q,q')q' + N(q) = u + J^T Fc      -> class Leg
 Eq.(7)      spring-damper GRF  Fc = k*delta + c*delta'                      -> Leg.contact_torque
 Eq.(9)      stochastic model   Wiener-process disturbance + random GRF      -> Environment
 Eq.(10)     state constraints  q_inf <= q <= q_sup (soft in the MPC)        -> MPC (slack variables)
 Eq.(13)     gait coordination  e(t) = q_p(t) - q_L(t - Gamma),  Gamma=Tg/2  -> HealthyGait + metrics

Model: planar two-link leg (q1 = hip/thigh angle from vertical, q2 = knee angle relative to thigh,
q2 <= 0 means knee flexion). Both joints are torque-actuated by the prosthesis.
The "healthy limb" gait is a synthetic periodic reference; the prosthesis must track it with a
half-cycle phase lag.

Controllers compared: MPC (LTV, constrained)  vs  Computed-Torque PD  vs  PD + gravity compensation.
The controllers only know the NOMINAL model; the simulated plant has parameter mismatch,
joint friction, uneven terrain contact and random torque noise.

Run:  python prosthesis_mpc.py            (all controllers, saves plots in ./outputs)
      python prosthesis_mpc.py --help
"""
import argparse
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize

GRAV = 9.81
DEG = 180.0 / np.pi


# =============================================================================
# 1. LAGRANGIAN DYNAMICS  (paper Eq. 8)
# =============================================================================
class Leg:
    """Two-link planar leg. State q=[q1,q2]; angles measured from downward vertical."""

    def __init__(self, m1=7.0, m2=4.0, l1=0.45, l2=0.45, lc1=0.20, lc2=0.20, b=(0.0, 0.0)):
        self.m1, self.m2, self.l1, self.l2, self.lc1, self.lc2 = m1, m2, l1, l2, lc1, lc2
        self.I1 = m1 * l1 ** 2 / 12.0
        self.I2 = m2 * l2 ** 2 / 12.0
        self.b = np.array(b, float)           # viscous joint friction (unknown to controller)

    def M(self, q):
        c2 = np.cos(q[1])
        a = self.m2 * self.l1 * self.lc2 * c2
        m11 = self.m1 * self.lc1 ** 2 + self.m2 * (self.l1 ** 2 + self.lc2 ** 2) + 2 * a + self.I1 + self.I2
        m12 = self.m2 * self.lc2 ** 2 + a + self.I2
        m22 = self.m2 * self.lc2 ** 2 + self.I2
        return np.array([[m11, m12], [m12, m22]])

    def Cqd(self, q, qd):                     # Coriolis/centrifugal vector  C(q,qd)*qd
        h = -self.m2 * self.l1 * self.lc2 * np.sin(q[1])
        return np.array([h * (2 * qd[0] * qd[1] + qd[1] ** 2), -h * qd[0] ** 2])

    def N(self, q):                           # gravity vector
        s12 = np.sin(q[0] + q[1])
        return np.array([(self.m1 * self.lc1 + self.m2 * self.l1) * GRAV * np.sin(q[0]) + self.m2 * self.lc2 * GRAV * s12,
                         self.m2 * self.lc2 * GRAV * s12])

    # --- kinematics (used for ground contact, Eq. 6-7) ---
    def foot(self, q):
        x = self.l1 * np.sin(q[0]) + self.l2 * np.sin(q[0] + q[1])
        y = -(self.l1 * np.cos(q[0]) + self.l2 * np.cos(q[0] + q[1]))
        return np.array([x, y])

    def jac(self, q):
        c1, c12 = np.cos(q[0]), np.cos(q[0] + q[1])
        s1, s12 = np.sin(q[0]), np.sin(q[0] + q[1])
        return np.array([[self.l1 * c1 + self.l2 * c12, self.l2 * c12],
                         [self.l1 * s1 + self.l2 * s12, self.l2 * s12]])

    def inverse_dynamics(self, q, qd, qdd):
        return self.M(q) @ qdd + self.Cqd(q, qd) + self.N(q)

    def accel(self, q, qd, tau, tau_ext=0.0):
        rhs = tau + tau_ext - self.Cqd(q, qd) - self.N(q) - self.b * qd
        return np.linalg.solve(self.M(q), rhs)

    def f(self, x, tau, tau_ext=0.0):
        return np.concatenate([x[2:], self.accel(x[:2], x[2:], tau, tau_ext)])

    def rk4(self, x, tau, dt, tau_ext=0.0):
        k1 = self.f(x, tau, tau_ext)
        k2 = self.f(x + 0.5 * dt * k1, tau, tau_ext)
        k3 = self.f(x + 0.5 * dt * k2, tau, tau_ext)
        k4 = self.f(x + dt * k3, tau, tau_ext)
        return x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)


# =============================================================================
# 2. HEALTHY-LIMB GAIT REFERENCE  and  coordination delay (paper Eq. 13)
# =============================================================================
class HealthyGait:
    """Periodic healthy hip/knee trajectories q_L(t). Prosthesis reference = q_L(t - Gamma)."""

    def __init__(self, Tg=1.1, n=2200, h0=0.05, hA=0.30, kc=0.15, kp=1.00):
        self.Tg = Tg
        ph = np.linspace(0, 1, n + 1)

        def bump(p, c, w):
            d = (p - c + 0.5) % 1.0 - 0.5
            return np.exp(-0.5 * (d / w) ** 2)

        hip = h0 + hA * np.cos(2 * np.pi * (ph - 0.03))
        knee = 0.02 + kc * bump(ph, 0.15, 0.06) + kp * bump(ph, 0.72, 0.10)
        Y = np.stack([hip, -knee], axis=1)      # q2 <= 0  : knee flexion
        Y[-1] = Y[0]
        self.spl = CubicSpline(ph * Tg, Y, bc_type="periodic")

    def ref(self, t, shift=0.0):
        """returns q, qd, qdd of healthy limb evaluated at (t - shift)."""
        te = (t - shift) % self.Tg
        return self.spl(te), self.spl(te, 1), self.spl(te, 2)


# =============================================================================
# 3. ENVIRONMENT: stochastically uneven terrain + random torque (paper Eq. 7, 9)
# =============================================================================
class Environment:
    def __init__(self, plant, gait, shift, T, dt, seed=0, uneven=0.002, torque_noise=0.3,
                 contact=True, k=2.0e4, c=100.0, ct=50.0):
        rng = np.random.default_rng(seed)
        n = int(round(T / dt)) + 2
        self.dt, self.contact, self.k, self.c, self.ct = dt, contact, k, c, ct
        # ground level: just above the lowest foot point of the reference cycle
        ts = np.linspace(0, gait.Tg, 400)
        ymin = min(plant.foot(gait.ref(t, shift)[0])[1] for t in ts)
        self.ground0 = ymin + 0.004
        # Ornstein-Uhlenbeck ground roughness (stochastically uneven terrain)
        tau_c = 0.2
        a = np.exp(-dt / tau_c)
        r = np.zeros(n)
        for i in range(1, n):
            r[i] = a * r[i - 1] + uneven * np.sqrt(1 - a * a) * rng.standard_normal()
        self.rough = r
        # white torque disturbance (discretised Wiener increment dW/dt)
        self.tau_noise = torque_noise * rng.standard_normal((n, 2)) / np.sqrt(dt)

    def contact_torque(self, plant, x, i):
        if not self.contact:
            return np.zeros(2)
        q, qd = x[:2], x[2:]
        p = plant.foot(q)
        J = plant.jac(q)
        v = J @ qd
        pen = self.ground0 + self.rough[i] - p[1]          # delta : foot-ground penetration
        if pen <= 0:
            return np.zeros(2)
        Fy = max(0.0, self.k * pen + self.c * (-v[1]))    # Fc = k*delta + c*delta'   (Eq. 7)
        Fx = -self.ct * v[0]
        return J.T @ np.array([Fx, Fy])                   # Fe = J^T Fc               (Eq. 6)

    def disturbance(self, plant, x, i):
        return self.contact_torque(plant, x, i) + self.tau_noise[i]


# =============================================================================
# 4. CONTROLLERS
# =============================================================================
class PDGravity:
    name = "PD + gravity comp."

    def __init__(self, nominal, gait, shift, umax, Kp=(120.0, 60.0), Kd=(12.0, 3.5)):
        self.nom, self.gait, self.shift, self.umax = nominal, gait, shift, umax
        self.Kp, self.Kd = np.array(Kp), np.array(Kd)

    def __call__(self, t, x):
        qr, qdr, _ = self.gait.ref(t, self.shift)
        u = self.nom.N(x[:2]) + self.Kp * (qr - x[:2]) + self.Kd * (qdr - x[2:])
        return np.clip(u, -self.umax, self.umax)


class ComputedTorque:
    name = "Computed-torque PD"

    def __init__(self, nominal, gait, shift, umax, wn=(25.0, 25.0), zeta=1.0):
        self.nom, self.gait, self.shift, self.umax = nominal, gait, shift, umax
        self.Kp = np.array(wn) ** 2
        self.Kd = 2 * zeta * np.array(wn)

    def __call__(self, t, x):
        qr, qdr, qddr = self.gait.ref(t, self.shift)
        e, ed = qr - x[:2], qdr - x[2:]
        a = qddr + self.Kd * ed + self.Kp * e
        u = self.nom.inverse_dynamics(x[:2], x[2:], a)
        return np.clip(u, -self.umax, self.umax)


class MPC:
    """
    Linear time-varying MPC around the gait reference.
      error state  e = x - x_ref(t),  x=[q1,q2,qd1,qd2]
      model        e_{k+1} = A_k e_k + B_k du_k + c_k      (RK4-discretised nominal Lagrangian model,
                                                            linearised by finite differences around
                                                            (x_ref, u_ff))
      input        u = u_ff + du,  u_ff = nominal inverse dynamics of the reference
      cost         sum e'Qe + du'R du + e_N'Qf e_N + rho*(bound violation)^2
      constraints  |u| <= umax (hard),  q_inf <= q <= q_sup (soft penalty, paper Eq. 10)
    The condensed QP (box-constrained, soft state constraints) is solved with a projected Newton method (numpy only).
      An optional disturbance estimator (offset-free MPC) compensates unmodelled torques.
    """
    name = "MPC"

    def __init__(self, nominal, gait, shift, umax, qmin, qmax, N=15, Ts=0.02,
                 Q=(2000.0, 2000.0, 20.0, 20.0), R=(0.002, 0.002), Qf_scale=5.0, rho=1e5, dist_est=True, alpha=0.4):
        self.nom, self.gait, self.shift = nominal, gait, shift
        self.umax, self.qmin, self.qmax = np.asarray(umax, float), np.asarray(qmin, float), np.asarray(qmax, float)
        self.N, self.Ts, self.rho = N, Ts, rho
        self.Q, self.R = np.diag(Q), np.diag(R)
        self.Qf = Qf_scale * self.Q
        self.nx, self.nu = 4, 2
        self._z_prev = None
        self.dist_est, self.alpha = dist_est, alpha
        self.dhat = np.zeros(2)                   # estimated lumped torque disturbance [Nm]
        self._x_prev = self._u_prev = None
        self.solve_times, self.fail_count = [], 0

        # constant parts of the QP
        N_, nx, nu = N, self.nx, self.nu
        self.Qb = np.zeros((N_ * nx, N_ * nx))
        for k in range(N_):
            self.Qb[k * nx:(k + 1) * nx, k * nx:(k + 1) * nx] = self.Qf if k == N_ - 1 else self.Q
        self.Rb = np.kron(np.eye(N_), self.R)
        self.P = np.zeros((N_ * 2, N_ * nx))          # selects joint positions from stacked error
        for k in range(N_):
            self.P[k * 2:(k + 1) * 2, k * nx:k * nx + 2] = np.eye(2)

    # discrete nominal model with zero-order hold over Ts
    def F(self, x, u):
        xn, sub = x, 2
        for _ in range(sub):
            xn = self.nom.rk4(xn, u + self.dhat, self.Ts / sub)    # u + estimated disturbance
        return xn

    def update_disturbance(self, x):
        """Offset-free MPC: estimate unmodelled torque (contact, friction, mismatch) from the
        one-step prediction residual and low-pass it (alpha = filter gain)."""
        if not self.dist_est or self._x_prev is None:
            return
        pred = self.F(self._x_prev, self._u_prev)
        r = x[2:] - pred[2:]                                       # velocity residual
        self.dhat = np.clip(self.dhat + self.alpha * (self.nom.M(x[:2]) @ r) / self.Ts, -40.0, 40.0)

    def linearize(self, xb, ub, eps=1e-6):
        A = np.zeros((4, 4))
        B = np.zeros((4, 2))
        f0 = self.F(xb, ub)
        for i in range(4):
            d = np.zeros(4)
            d[i] = eps
            A[:, i] = (self.F(xb + d, ub) - self.F(xb - d, ub)) / (2 * eps)
        for j in range(2):
            d = np.zeros(2)
            d[j] = eps
            B[:, j] = (self.F(xb, ub + d) - self.F(xb, ub - d)) / (2 * eps)
        return A, B, f0

    def reference_x(self, t):
        q, qd, qdd = self.gait.ref(t, self.shift)
        return np.concatenate([q, qd]), self.nom.inverse_dynamics(q, qd, qdd)

    @staticmethod
    def _projected_newton(fun, H, PSu, lo, hi, rho, z, zl, zu, iters=25, tol=1e-8):
        """Projected Newton for  min f(z)  s.t. zl<=z<=zu  with piecewise-quadratic f
        (QP + hinge-squared soft state constraints). Converges in a handful of iterations."""
        f, gr = fun(z)
        for _ in range(iters):
            at_lo, at_hi = (z <= zl + 1e-10) & (gr > 0), (z >= zu - 1e-10) & (gr < 0)
            free = ~(at_lo | at_hi)
            if np.linalg.norm(gr[free]) < tol * max(1.0, abs(f)):
                return z, True
            p = PSu @ z
            act = ((p < lo) | (p > hi)).astype(float)
            Hs = H + rho * (PSu.T * act) @ PSu
            d = np.zeros_like(z)
            d[free] = -np.linalg.solve(Hs[np.ix_(free, free)] + 1e-9 * np.eye(free.sum()), gr[free])
            step, ok = 1.0, False
            for _ in range(30):                                  # backtracking line search
                zn = np.clip(z + step * d, zl, zu)
                fn, gn = fun(zn)
                if fn <= f + 1e-4 * gr @ (zn - z):
                    ok = True
                    break
                step *= 0.5
            if not ok:
                return z, False
            z, f, gr = zn, fn, gn
        return z, False

    def __call__(self, t, x):
        N, nx, nu = self.N, self.nx, self.nu
        self.update_disturbance(x)
        # ---- build prediction model along the reference over the horizon
        xb, ub = [], []
        for k in range(N + 1):
            xr, ur = self.reference_x(t + k * self.Ts)
            xb.append(xr)
            ub.append(np.clip(ur - self.dhat, -self.umax, self.umax))   # feedforward minus estimated disturbance
        A, B, c = [], [], []
        for k in range(N):
            Ak, Bk, fk = self.linearize(xb[k], ub[k])
            A.append(Ak); B.append(Bk); c.append(fk - xb[k + 1])

        Sx = np.zeros((N * nx, nx)); Su = np.zeros((N * nx, N * nu)); Sc = np.zeros(N * nx)
        pSx, pSu, pSc = np.eye(nx), np.zeros((nx, N * nu)), np.zeros(nx)
        for k in range(N):
            cSx = A[k] @ pSx
            cSu = A[k] @ pSu
            cSu[:, k * nu:(k + 1) * nu] += B[k]
            cSc = A[k] @ pSc + c[k]
            Sx[k * nx:(k + 1) * nx] = cSx; Su[k * nx:(k + 1) * nx] = cSu; Sc[k * nx:(k + 1) * nx] = cSc
            pSx, pSu, pSc = cSx, cSu, cSc

        e0 = x - xb[0]
        free = Sx @ e0 + Sc                                     # error prediction with du = 0
        H = Su.T @ self.Qb @ Su + self.Rb
        g = Su.T @ self.Qb @ free
        # soft state constraints  qmin <= q <= qmax  (quadratic hinge penalty, paper Eq. 10)
        qb = np.concatenate([xb[k + 1][:2] for k in range(N)])
        PSu, Pfree = self.P @ Su, self.P @ free
        lo = np.tile(self.qmin, N) - qb - Pfree          # need  PSu du >= lo
        hi = np.tile(self.qmax, N) - qb - Pfree          # need  PSu du <= hi
        rho = self.rho

        def fun(du):
            p = PSu @ du
            vl, vh = np.maximum(0.0, lo - p), np.maximum(0.0, p - hi)
            val = 0.5 * du @ H @ du + g @ du + 0.5 * rho * (vl @ vl + vh @ vh)
            grad = H @ du + g + rho * (PSu.T @ (vh - vl))
            return val, grad

        ub_arr = np.concatenate([ub[k] for k in range(N)])
        du_lo = np.tile(-self.umax, N) - ub_arr          # hard torque limits |u| <= umax
        du_hi = np.tile(self.umax, N) - ub_arr
        z0 = np.zeros(N * nu) if self._z_prev is None else self._z_prev
        z0 = np.clip(z0, du_lo, du_hi)
        t0 = time.perf_counter()
        z, converged = self._projected_newton(fun, H, PSu, lo, hi, rho, z0, du_lo, du_hi)
        self.solve_times.append(time.perf_counter() - t0)
        if not converged:
            self.fail_count += 1
        # warm start: shift the input sequence by one step
        zs = z.copy()
        zs[:(N - 1) * nu] = z[nu:N * nu]
        self._z_prev = zs
        u = ub[0] + z[:nu]
        u = np.clip(u, -self.umax, self.umax)
        self._x_prev, self._u_prev = x.copy(), u.copy()
        return u


# =============================================================================
# 5. SIMULATION  &  METRICS
# =============================================================================
def simulate(controller, plant, env, gait, shift, T, Ts, dt):
    steps = int(round(T / Ts))
    sub = int(round(Ts / dt))
    q0, qd0, _ = gait.ref(0.0, shift)
    x = np.concatenate([q0 + np.array([0.04, -0.04]), qd0])      # small initial offset
    log = {k: [] for k in ("t", "q", "qd", "qref", "u", "dist")}
    i = 0
    for s in range(steps):
        t = s * Ts
        u = controller(t, x)
        qr, _, _ = gait.ref(t, shift)
        log["t"].append(t); log["q"].append(x[:2].copy()); log["qd"].append(x[2:].copy())
        log["qref"].append(qr.copy()); log["u"].append(u.copy())
        log["dist"].append(env.contact_torque(plant, x, i).copy())
        for _ in range(sub):
            x = plant.rk4(x, u, dt, env.disturbance(plant, x, i))
            i += 1
    return {k: np.array(v) for k, v in log.items()}


def metrics(log, Tg, qmin, qmax, umax):
    m = log["t"] > Tg                                           # discard first cycle (transient)
    e = log["q"][m] - log["qref"][m]
    rmse = np.sqrt(np.mean(e ** 2, axis=0)) * DEG
    coord = np.linalg.norm(e, axis=1) * DEG                      # ||q_p(t) - q_L(t-Gamma)||  (Eq. 13)
    u = log["u"][m]
    power = np.maximum(0, np.sum(u * log["qd"][m], axis=1))
    viol = np.mean(np.any((log["q"][m] < qmin) | (log["q"][m] > qmax), axis=1)) * 100
    return {"RMSE hip [deg]": rmse[0], "RMSE knee [deg]": rmse[1],
            "Coord. error mean [deg]": coord.mean(), "Coord. error max [deg]": coord.max(),
            "Torque RMS hip [Nm]": np.sqrt(np.mean(u[:, 0] ** 2)),
            "Torque RMS knee [Nm]": np.sqrt(np.mean(u[:, 1] ** 2)),
            "Mean positive power [W]": power.mean(),
            "Sat. time [%]": np.mean(np.any(np.abs(u) >= umax - 1e-6, axis=1)) * 100,
            "State-bound violation [%]": viol}


# =============================================================================
# 6. PLOTS & ANIMATION
# =============================================================================
COLORS = {"MPC": "tab:red", "Computed-torque PD": "tab:blue", "PD + gravity comp.": "tab:green"}


def make_plots(logs, mets, Tg, outdir):
    names = list(logs)
    t = logs[names[0]]["t"]
    fig, ax = plt.subplots(4, 1, figsize=(11, 11), sharex=True)
    for j, jn in enumerate(["Hip (q1)", "Knee (q2)"]):
        ax[j].plot(t, logs[names[0]]["qref"][:, j] * DEG, "k--", lw=2, label="Healthy limb delayed q_L(t-Γ)")
        for n in names:
            ax[j].plot(t, logs[n]["q"][:, j] * DEG, color=COLORS.get(n), lw=1.3, label=n)
        ax[j].set_ylabel(jn + " [deg]"); ax[j].grid(alpha=.3)
    ax[0].legend(ncol=4, fontsize=8, loc="upper right")
    for n in names:
        e = np.linalg.norm(logs[n]["q"] - logs[n]["qref"], axis=1) * DEG
        ax[2].plot(t, e, color=COLORS.get(n), label=n)
        ax[3].plot(t, logs[n]["u"][:, 1], color=COLORS.get(n), label=n)
    ax[2].set_ylabel("‖e(t)‖ [deg]\n(coordination error)"); ax[2].grid(alpha=.3)
    ax[3].set_ylabel("Knee torque [Nm]"); ax[3].set_xlabel("time [s]"); ax[3].grid(alpha=.3)
    plt.suptitle("Gait coordination of a two-link prosthetic leg")
    plt.tight_layout(); plt.savefig(f"{outdir}/tracking.png", dpi=150); plt.close()

    # bar chart comparison
    keys = ["RMSE hip [deg]", "RMSE knee [deg]", "Coord. error max [deg]", "Mean positive power [W]"]
    fig, ax = plt.subplots(1, len(keys), figsize=(4 * len(keys), 3.6))
    for a, k in zip(ax, keys):
        a.bar(range(len(names)), [mets[n][k] for n in names], color=[COLORS.get(n) for n in names])
        a.set_xticks(range(len(names))); a.set_xticklabels([n.replace(" ", "\n", 1) for n in names], fontsize=7)
        a.set_title(k, fontsize=9)
    plt.tight_layout(); plt.savefig(f"{outdir}/comparison.png", dpi=150); plt.close()

    # phase portrait hip-knee (gait cycle shape)
    plt.figure(figsize=(5, 5))
    plt.plot(logs[names[0]]["qref"][:, 0] * DEG, logs[names[0]]["qref"][:, 1] * DEG, "k--", lw=2, label="reference")
    for n in names:
        plt.plot(logs[n]["q"][:, 0] * DEG, logs[n]["q"][:, 1] * DEG, color=COLORS.get(n), lw=1, label=n)
    plt.xlabel("hip q1 [deg]"); plt.ylabel("knee q2 [deg]"); plt.legend(fontsize=7); plt.grid(alpha=.3)
    plt.title("Hip-knee cyclogram"); plt.tight_layout(); plt.savefig(f"{outdir}/cyclogram.png", dpi=150); plt.close()


def make_animation(log, gait, plant, shift, env, outdir, fps=25):
    from matplotlib.animation import FuncAnimation, PillowWriter
    t = log["t"]
    fig, ax = plt.subplots(figsize=(4, 5))
    ax.set_xlim(-0.7, 0.7); ax.set_ylim(-1.0, 0.15); ax.set_aspect("equal"); ax.axis("off")
    ax.axhline(env.ground0, color="saddlebrown", lw=3)
    (hl,) = ax.plot([], [], "o-", color="tab:blue", lw=6, alpha=.45, label="healthy limb")
    (pl,) = ax.plot([], [], "o-", color="tab:red", lw=4, label="prosthesis (MPC)")
    ttl = ax.text(0, 0.08, "", ha="center"); ax.legend(loc="lower center", fontsize=7)

    def pts(q):
        k = np.array([plant.l1 * np.sin(q[0]), -plant.l1 * np.cos(q[0])])
        f = plant.foot(q)
        return [0, k[0], f[0]], [0, k[1], f[1]]

    def upd(i):
        qh, _, _ = gait.ref(t[i], 0.0)
        hl.set_data(*pts(qh)); pl.set_data(*pts(log["q"][i])); ttl.set_text(f"t = {t[i]:.2f} s")
        return hl, pl, ttl

    FuncAnimation(fig, upd, frames=range(0, len(t), 2), blit=False).save(
        f"{outdir}/animation.gif", writer=PillowWriter(fps=fps))
    plt.close()


# =============================================================================
def main():
    ap = argparse.ArgumentParser(description="MPC for a two-link robotic prosthetic leg")
    ap.add_argument("--controller", default="all", choices=["all", "mpc", "ctc", "pd"])
    ap.add_argument("--duration", type=float, default=6.0, help="simulation time [s]")
    ap.add_argument("--horizon", type=int, default=15, help="MPC prediction horizon (steps of 20 ms)")
    ap.add_argument("--gait-period", type=float, default=1.1, help="gait cycle Tg [s]")
    ap.add_argument("--delay-ratio", type=float, default=0.5, help="Gamma = ratio*Tg  (paper: 0.5)")
    ap.add_argument("--mismatch", type=float, default=0.25, help="plant mass error vs nominal model (0.25 = +25%%)")
    ap.add_argument("--friction", type=float, default=1.5, help="unmodelled viscous joint friction [Nms/rad]")
    ap.add_argument("--torque-noise", type=float, default=0.3, help="random torque disturbance intensity")
    ap.add_argument("--terrain", type=float, default=0.002, help="terrain roughness std [m] (stochastic)")
    ap.add_argument("--no-contact", action="store_true", help="disable foot-ground contact")
    ap.add_argument("--hip-limit", type=float, default=100.0)
    ap.add_argument("--knee-limit", type=float, default=70.0)
    ap.add_argument("--no-dist-est", action="store_true", help="disable MPC disturbance estimator")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--animate", action="store_true", help="also write animation.gif")
    ap.add_argument("--out", default="outputs")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    Ts, dt = 0.02, 0.005
    gait = HealthyGait(Tg=a.gait_period)
    shift = a.delay_ratio * a.gait_period                       # Gamma
    nominal = Leg()
    plant = Leg(m1=nominal.m1 * (1 + a.mismatch), m2=nominal.m2 * (1 + a.mismatch), b=(a.friction, a.friction))
    umax = np.array([a.hip_limit, a.knee_limit])
    qmin, qmax = np.array([-0.60, -1.45]), np.array([0.80, 0.03])  # state constraints (Eq. 10)
    print(f"Gait period Tg={a.gait_period}s, coordination delay Gamma={shift:.3f}s, "
          f"mass mismatch {a.mismatch*100:.0f}%, friction {a.friction}, contact={not a.no_contact}")

    ctrls = {"pd": PDGravity(nominal, gait, shift, umax),
             "ctc": ComputedTorque(nominal, gait, shift, umax),
             "mpc": MPC(nominal, gait, shift, umax, qmin, qmax, N=a.horizon, Ts=Ts, dist_est=not a.no_dist_est)}
    chosen = list(ctrls) if a.controller == "all" else [a.controller]

    logs, mets = {}, {}
    for key in chosen:
        c = ctrls[key]
        env = Environment(plant, gait, shift, a.duration, dt, seed=a.seed, uneven=a.terrain,
                          torque_noise=a.torque_noise, contact=not a.no_contact)   # same noise for all
        t0 = time.perf_counter()
        logs[c.name] = simulate(c, plant, env, gait, shift, a.duration, Ts, dt)
        wall = time.perf_counter() - t0
        mets[c.name] = metrics(logs[c.name], gait.Tg, qmin, qmax, umax)
        extra = ""
        if key == "mpc":
            st = np.array(c.solve_times) * 1000
            extra = f" | MPC solve time mean {st.mean():.1f} ms, max {st.max():.1f} ms, solver warnings {c.fail_count}"
        print(f"  [{c.name}] simulated in {wall:.1f}s{extra}")

    import pandas as pd
    table = pd.DataFrame(mets).T.round(3)
    print("\n=== Results (after first gait cycle) ===")
    print(table.to_string())
    table.to_csv(f"{a.out}/metrics.csv")

    make_plots(logs, mets, gait.Tg, a.out)
    if a.animate and "MPC" in logs:
        env = Environment(plant, gait, shift, a.duration, dt, seed=a.seed, uneven=a.terrain,
                          torque_noise=a.torque_noise, contact=not a.no_contact)
        make_animation(logs["MPC"], gait, plant, shift, env, a.out)
    print(f"\nSaved figures/metrics in ./{a.out}")


if __name__ == "__main__":
    main()
