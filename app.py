"""
Streamlit dashboard: MPC for a two-link robotic prosthetic leg.
Run:  streamlit run app.py          (keep this file next to prosthesis_mpc.py)
"""
import json
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from plotly.subplots import make_subplots

import prosthesis_mpc as pm

st.set_page_config(page_title="Prosthesis MPC", layout="wide")
COL = {"PD + gravity comp.": "#2ca02c", "Computed-torque PD": "#1f77b4",
       "MPC": "#d62728", "MPC (no estimator)": "#ff7f0e"}
DEG = 180 / np.pi
ACT = {"walking": dict(n="Walking", f=0.90, h0=0.05, hA=0.30, kc=0.15, kp=1.00),
       "jogging": dict(n="Jogging", f=1.15, h0=0.12, hA=0.40, kc=0.25, kp=1.35),
       "running": dict(n="Running", f=1.40, h0=0.15, hA=0.50, kc=0.35, kp=1.60),
       "stairsup": dict(n="Stairs Up", f=0.70, h0=0.45, hA=0.35, kc=0.20, kp=1.50),
       "stairsdown": dict(n="Stairs Down", f=0.70, h0=0.10, hA=0.30, kc=0.50, kp=1.10)}


# ----------------------------------------------------------------------------- simulation
@st.cache_data(show_spinner=False)
def run_sim(duration, horizon, delay_ratio, activity, mismatch, friction, noise, terrain,
            contact, hip_lim, knee_lim, seed, with_noest):
    Ts, dt = 0.02, 0.005
    ac = ACT[activity]
    Tg = 1.0 / ac["f"]
    gait = pm.HealthyGait(Tg=Tg, h0=ac["h0"], hA=ac["hA"], kc=ac["kc"], kp=ac["kp"])
    shift = delay_ratio * Tg
    nominal = pm.Leg()
    plant = pm.Leg(m1=nominal.m1 * (1 + mismatch), m2=nominal.m2 * (1 + mismatch), b=(friction, friction))
    umax = np.array([hip_lim, knee_lim])
    qmin = np.array([ac["h0"] - ac["hA"] - 0.2, -(ac["kp"] + 0.35)])
    qmax = np.array([ac["h0"] + ac["hA"] + 0.25, 0.03])
    mk = lambda est: pm.MPC(nominal, gait, shift, umax, qmin, qmax, N=horizon, Ts=Ts, dist_est=est)
    ctrls = [("PD + gravity comp.", pm.PDGravity(nominal, gait, shift, umax)),
             ("Computed-torque PD", pm.ComputedTorque(nominal, gait, shift, umax)),
             ("MPC", mk(True))]
    if with_noest:
        ctrls.append(("MPC (no estimator)", mk(False)))
    logs, rows, solve = {}, {}, None
    for name, c in ctrls:
        env = pm.Environment(plant, gait, shift, duration, dt, seed=seed, uneven=terrain,
                             torque_noise=noise, contact=contact)
        logs[name] = pm.simulate(c, plant, env, gait, shift, duration, Ts, dt)
        rows[name] = pm.metrics(logs[name], gait.Tg, qmin, qmax, umax)
        if name == "MPC":
            solve = float(np.mean(c.solve_times) * 1000)
    t = logs["MPC"]["t"]
    healthy = np.array([gait.ref(tt, 0.0)[0] for tt in t])
    return logs, pd.DataFrame(rows).T.round(3), healthy, dict(
        Ts=Ts, ground=float(env.ground0), qmin=qmin.tolist(), qmax=qmax.tolist(), Tg=Tg, act=activity, l1=plant.l1, l2=plant.l2, shift=shift, solve_ms=solve)


# ----------------------------------------------------------------------------- plots
def tracking_fig(logs, healthy_delayed):
    t = logs["MPC"]["t"]
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.04,
                        subplot_titles=("Hip angle q1 [deg]", "Knee angle q2 [deg]",
                                        "Coordination error ‖q_p(t) − q_L(t−Γ)‖ [deg]", "Knee torque [Nm]"))
    for j in range(2):
        fig.add_trace(go.Scatter(x=t, y=healthy_delayed[:, j] * DEG, name="Reference q_L(t−Γ)",
                                 line=dict(color="black", dash="dash", width=2),
                                 showlegend=(j == 0), legendgroup="ref"), row=j + 1, col=1)
    for n, L in logs.items():
        kw = dict(line=dict(color=COL[n], width=1.8), legendgroup=n)
        fig.add_trace(go.Scatter(x=t, y=L["q"][:, 0] * DEG, name=n, **kw), row=1, col=1)
        fig.add_trace(go.Scatter(x=t, y=L["q"][:, 1] * DEG, showlegend=False, **kw), row=2, col=1)
        fig.add_trace(go.Scatter(x=t, y=np.linalg.norm(L["q"] - L["qref"], axis=1) * DEG,
                                 showlegend=False, **kw), row=3, col=1)
        fig.add_trace(go.Scatter(x=t, y=L["u"][:, 1], showlegend=False, **kw), row=4, col=1)
    fig.update_xaxes(title_text="time [s]", row=4, col=1)
    fig.update_layout(height=900, hovermode="x unified", legend=dict(orientation="h", y=1.06),
                      margin=dict(t=70, b=30))
    return fig


def cyclogram_fig(logs, ref):
    fig = go.Figure(go.Scatter(x=ref[:, 0] * DEG, y=ref[:, 1] * DEG, name="Reference",
                               line=dict(color="black", dash="dash", width=3)))
    for n, L in logs.items():
        fig.add_trace(go.Scatter(x=L["q"][:, 0] * DEG, y=L["q"][:, 1] * DEG, name=n,
                                 line=dict(color=COL[n], width=1.5)))
    fig.update_layout(height=500, xaxis_title="hip q1 [deg]", yaxis_title="knee q2 [deg]",
                      title="Hip-knee cyclogram")
    return fig


# ----------------------------------------------------------------------------- animation (HTML5 canvas + JS)
ANIM_HTML = """
<div style="font-family:Arial,sans-serif;color:#222">
 <div style="display:flex;gap:14px;align-items:center;margin-bottom:8px">
  <button id="pp" style="padding:6px 16px;border-radius:6px;border:1px solid #888;cursor:pointer">Pause</button>
  <label>Speed <select id="sp"><option value="0.25">0.25x</option><option value="0.5">0.5x</option>
   <option value="1" selected>1x</option><option value="2">2x</option></select></label>
  <input type="range" id="sl" min="0" max="0" value="0" style="flex:1"><span id="tl" style="width:70px"></span>
 </div>
 <canvas id="cv" style="width:100%;height:440px;border:1px solid #ddd;border-radius:10px;background:#fff"></canvas>
</div>
<script>
const D = __DATA__, N = D.t.length, H = 440, R = 180/Math.PI;
const cv = document.getElementById('cv'), ctx = cv.getContext('2d');
const sl = document.getElementById('sl'), pp = document.getElementById('pp'), sp = document.getElementById('sp');
sl.max = N-1;
let idx = 0, last = null, playing = true;
function size(){ const d = window.devicePixelRatio||1; cv.width = cv.clientWidth*d; cv.height = H*d; }
window.addEventListener('resize', size); size();
function lerpQ(a, i){ const i0 = Math.floor(i)%N, i1 = (i0+1)%N, f = i-Math.floor(i);
  return [a[i0][0]+(a[i1][0]-a[i0][0])*f, a[i0][1]+(a[i1][1]-a[i0][1])*f]; }
function pts(q, sc, hx, hy){
  const kx = D.l1*Math.sin(q[0]), ky = -D.l1*Math.cos(q[0]);
  const fx = kx + D.l2*Math.sin(q[0]+q[1]), fy = ky - D.l2*Math.cos(q[0]+q[1]);
  return [[hx,hy],[hx+kx*sc,hy-ky*sc],[hx+fx*sc,hy-fy*sc]]; }
function leg(p, color, w, alpha){
  ctx.globalAlpha = alpha; ctx.strokeStyle = color; ctx.fillStyle = color; ctx.lineWidth = w;
  ctx.lineCap = 'round'; ctx.lineJoin = 'round';
  ctx.beginPath(); ctx.moveTo(p[0][0],p[0][1]); ctx.lineTo(p[1][0],p[1][1]); ctx.lineTo(p[2][0],p[2][1]); ctx.stroke();
  for (const c of p){ ctx.beginPath(); ctx.arc(c[0],c[1],w*0.65,0,6.3); ctx.fill(); }
  ctx.globalAlpha = 1; }
function draw(i){
  const d = window.devicePixelRatio||1; ctx.setTransform(d,0,0,d,0,0);
  const W = cv.clientWidth, n = D.names.length, pw = W/n, sc = (H-120)/1.0, hy = 70;
  ctx.clearRect(0,0,W,H);
  const qh = lerpQ(D.healthy, i), qr = lerpQ(D.qref, i);
  D.names.forEach((name,k)=>{
    const x0 = k*pw, hx = x0+pw/2;
    ctx.save(); ctx.beginPath(); ctx.rect(x0,0,pw,H); ctx.clip();
    if (k>0){ ctx.strokeStyle='#ddd'; ctx.lineWidth=1; ctx.beginPath(); ctx.moveTo(x0,0); ctx.lineTo(x0,H); ctx.stroke(); }
    const gy = hy - D.ground*sc;                       // ground
    ctx.fillStyle='#d9c7a8'; ctx.fillRect(x0,gy,pw,H-gy); ctx.strokeStyle='#8b6b3e'; ctx.lineWidth=3;
    ctx.beginPath(); ctx.moveTo(x0,gy); ctx.lineTo(x0+pw,gy); ctx.stroke();
    ctx.fillStyle='#8b6b3e'; ctx.fillRect(hx-26,hy-9,52,12);          // pelvis
    ctx.beginPath(); ctx.strokeStyle = D.colors[k]; ctx.globalAlpha = .35; ctx.lineWidth = 2;   // foot trail
    for (let j=0;j<45;j++){ const ii=(Math.floor(i)-45+j+N*2)%N; const f=pts(D.q[name][ii],sc,hx,hy)[2];
      if(j==0) ctx.moveTo(f[0],f[1]); else ctx.lineTo(f[0],f[1]); }
    ctx.stroke(); ctx.globalAlpha = 1;
    leg(pts(qh,sc,hx,hy), '#1f77b4', 16, .28);                        // healthy limb (ghost)
    const q = lerpQ(D.q[name], i);
    leg(pts(q,sc,hx,hy), D.colors[k], 8, 1);                          // prosthesis
    const err = Math.hypot(q[0]-qr[0], q[1]-qr[1])*R;
    ctx.fillStyle='#222'; ctx.font='bold 15px Arial'; ctx.textAlign='center'; ctx.fillText(name, hx, 24);
    ctx.font='13px Arial'; ctx.fillStyle = err>8?'#d62728':'#2a7a2a';
    ctx.fillText('coordination error: '+err.toFixed(1)+'°', hx, 44);
    ctx.restore(); });
  ctx.font='12px Arial'; ctx.textAlign='left'; ctx.fillStyle='#1f77b4'; ctx.fillText('■ healthy limb q_L(t)', 10, H-10);
  ctx.fillStyle='#555'; ctx.fillText('■ prosthesis (follows with delay Γ = '+D.shift.toFixed(2)+' s)', 160, H-10);
  document.getElementById('tl').textContent = 't = '+(i*D.Ts).toFixed(2)+' s'; }
function frame(ts){
  if (playing && last!==null){ idx += (ts-last)/1000*parseFloat(sp.value)/D.Ts; if (idx>=N) idx -= N; }
  last = ts; draw(idx); sl.value = Math.floor(idx); requestAnimationFrame(frame); }
pp.onclick = ()=>{ playing=!playing; pp.textContent = playing?'Pause':'Play'; };
sl.oninput = ()=>{ idx = +sl.value; draw(idx); };
requestAnimationFrame(frame);
</script>
"""


def animation_html(logs, healthy, info):
    names = list(logs)
    data = dict(names=names, colors=[COL[n] for n in names], Ts=info["Ts"], ground=info["ground"],
                l1=info["l1"], l2=info["l2"], shift=info["shift"],
                t=np.round(logs[names[0]]["t"], 3).tolist(),
                healthy=np.round(healthy, 4).tolist(),
                qref=np.round(logs[names[0]]["qref"], 4).tolist(),
                q={n: np.round(logs[n]["q"], 4).tolist() for n in names})
    return ANIM_HTML.replace("__DATA__", json.dumps(data))


# ----------------------------------------------------------------------------- UI
st.title("🦿 MPC for a Two-Link Robotic Prosthetic Leg")
st.caption("Lagrangian dynamics · gait coordination q_p(t) → q_L(t − Γ) · Ma, Zhang & Xu, IEEE TNSRE 2024")

with st.sidebar.form("params"):
    st.header("Simulation settings")
    duration = st.slider("Duration [s]", 3.0, 12.0, 6.0, 0.5)
    activity = st.selectbox("Activity", list(ACT), format_func=lambda k: ACT[k]["n"])
    delay_ratio = st.slider("Delay ratio Γ / Tg", 0.0, 1.0, 0.5, 0.05)
    horizon = st.slider("MPC horizon [steps × 20 ms]", 5, 30, 15)
    st.subheader("Uncertainty")
    mismatch = st.slider("Mass mismatch", 0.0, 0.6, 0.25, 0.05)
    friction = st.slider("Joint friction [Nms/rad]", 0.0, 4.0, 1.5, 0.25)
    noise = st.slider("Random torque noise", 0.0, 1.0, 0.3, 0.05)
    terrain = st.slider("Terrain roughness [m]", 0.0, 0.006, 0.002, 0.0005, format="%.4f")
    contact = st.checkbox("Foot-ground contact", True)
    st.subheader("Limits & options")
    hip_lim = st.slider("Hip torque limit [Nm]", 30, 300, 150, 5)
    knee_lim = st.slider("Knee torque limit [Nm]", 30, 250, 120, 5)
    with_noest = st.checkbox("Also run MPC without disturbance estimator", True)
    seed = st.number_input("Random seed", 0, 999, 1)
    st.form_submit_button("▶ Run simulation", use_container_width=True)

with st.spinner("Simulating controllers (about 10 s)…"):
    logs, table, healthy, info = run_sim(duration, horizon, delay_ratio, activity, mismatch, friction, noise,
                                         terrain, contact, hip_lim, knee_lim, int(seed), with_noest)
ref_delayed = logs["MPC"]["qref"]

c1, c2, c3, c4 = st.columns(4)
m, ct, pd_ = table.loc["MPC"], table.loc["Computed-torque PD"], table.loc["PD + gravity comp."]
c1.metric("MPC knee RMSE", f"{m['RMSE knee [deg]']:.2f}°", f"{m['RMSE knee [deg]'] - ct['RMSE knee [deg]']:+.2f}° vs CTC", delta_color="inverse")
c2.metric("MPC hip RMSE", f"{m['RMSE hip [deg]']:.2f}°", f"{m['RMSE hip [deg]'] - ct['RMSE hip [deg]']:+.2f}° vs CTC", delta_color="inverse")
c3.metric("MPC max coord. error", f"{m['Coord. error max [deg]']:.1f}°", f"{m['Coord. error max [deg]'] - pd_['Coord. error max [deg]']:+.1f}° vs PD", delta_color="inverse")
c4.metric("MPC solve time", f"{info['solve_ms']:.2f} ms", "limit: 20 ms")

tab_3d, tab_anim, tab_plots, tab_metrics, tab_model = st.tabs(["🧍 3D Simulator", "🎬 Quick animation", "📈 Plots", "📊 Metrics", "📐 Model"])

with tab_anim:
    st.markdown("Blue ghost = **healthy limb** q_L(t). Coloured leg = **prosthesis** following with a half-cycle lag. "
                "The text shows the live coordination error.")
    components.html(animation_html(logs, healthy, info), height=540)

with tab_plots:
    st.plotly_chart(tracking_fig(logs, ref_delayed), use_container_width=True)
    st.plotly_chart(cyclogram_fig(logs, ref_delayed), use_container_width=True)

with tab_metrics:
    st.dataframe(table, use_container_width=True)
    keys = ["RMSE hip [deg]", "RMSE knee [deg]", "Coord. error max [deg]", "Mean positive power [W]"]
    fig = go.Figure([go.Bar(name=n, x=keys, y=[table.loc[n, k] for k in keys], marker_color=COL[n])
                     for n in table.index])
    fig.update_layout(barmode="group", height=420)
    st.plotly_chart(fig, use_container_width=True)
    st.download_button("Download metrics CSV", table.to_csv().encode(), "metrics.csv", "text/csv")

with tab_model:
    st.subheader("Lagrangian model (paper Eq. 8)")
    st.latex(r"M(q)\ddot q + C(q,\dot q)\dot q + N(q) = u + J^T F_c")
    st.subheader("Ground contact (Eq. 7)")
    st.latex(r"F_{c} = k\,\delta + c\,\dot\delta")
    st.subheader("Gait coordination objective (Eq. 13)")
    st.latex(r"e(t) = q_p(t) - q_L(t-\Gamma),\quad \Gamma = T_g/2,\quad \lim_{t\to\infty}\|e(t)\| = 0")
    st.subheader("MPC problem (each 20 ms)")
    st.latex(r"\min_{\delta u}\ \sum_{k=1}^{N} e_k^T Q e_k + \delta u_k^T R\,\delta u_k"
             r"\quad \text{s.t. } e_{k+1}=A_k e_k + B_k\delta u_k + c_k,\ |u|\le u_{max},\ q_{inf}\le q\le q_{sup}")
    st.markdown("Input: `u = u_ff + δu`, where `u_ff` is the inverse dynamics of the reference and a disturbance "
                "estimator removes unmodelled torque (friction, contact, mass error).")


# ----------------------------------------------------------------------------- 3D simulator fed by the Python results
def simulator_html(logs, healthy, info):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prosthetic_leg_simulator.html")
    with open(path, encoding="utf-8") as f:
        html = f.read()
    names = list(logs)
    r4 = lambda x: np.round(x, 4).tolist()
    data = dict(act=info["act"], Tg=info["Tg"], Ts=info["Ts"], n=len(logs[names[0]]["t"]), names=names,
                qmin=info["qmin"], qmax=info["qmax"], healthy=r4(healthy), qref=r4(logs[names[0]]["qref"]),
                d={n: dict(q=r4(logs[n]["q"]), qd=r4(logs[n]["qd"]), u=r4(logs[n]["u"])) for n in names})
    tag = '<script src="https://cdnjs'
    return html.replace(tag, "<script>window.SIM_DATA=" + json.dumps(data) + ";</script>" + tag, 1)


with tab_3d:
    st.caption("The 3D figure plays back the exact joint angles, torques and constraint results from the Python "
               "simulation above. Change the activity or any setting in the sidebar and press Run simulation.")
    components.html(simulator_html(logs, healthy, info), height=1450, scrolling=True)
