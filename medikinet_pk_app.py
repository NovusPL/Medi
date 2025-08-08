
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Medikinet CR – PK sandbox", layout="wide")
st.title("Medikinet CR – PK sandbox")
st.caption("Toy model for educational tinkering – not medical advice.")

# --- Modeling helpers ---
def gaussian_peak(t, t_peak, sigma, amplitude):
    return amplitude * np.exp(-((t - t_peak) ** 2) / (2 * (sigma ** 2)))

def dose_profile(hours_from_start: np.ndarray, dose_mg: float, t0_hours: float, fed: bool):
    ir_mg = 0.5 * dose_mg
    er_mg = 0.5 * dose_mg
    if fed:
        ir_tmax = 1.5
        er_tmax = 4.5
    else:
        ir_tmax = 1.0
        er_tmax = 3.0
    ir_sigma = 0.5
    er_sigma = 1.2
    ir_amp = ir_mg * 0.8
    er_amp = er_mg * 0.6
    ir = gaussian_peak(hours_from_start, t0_hours + ir_tmax, ir_sigma, ir_amp)
    er = gaussian_peak(hours_from_start, t0_hours + er_tmax, er_sigma, er_amp)
    return ir, er, ir + er

# --- UI state ---
if "doses" not in st.session_state:
    st.session_state.doses = []

with st.sidebar:
    st.header("Add a dose")
    col1, col2 = st.columns([1,1])
    with col1:
        h = st.number_input("Hour", min_value=0, max_value=23, value=8, step=1)
    with col2:
        m = st.number_input("Min", min_value=0, max_value=59, value=0, step=5)
    dose_mg = st.number_input("Dose (mg)", min_value=5.0, max_value=80.0, value=20.0, step=5.0)
    fed = st.checkbox("After food (slower)", value=False)
    if st.button("➕ Add dose"):
        t_str = f"{int(h):02d}:{int(m):02d}"
        st.session_state.doses.append({"time_str": t_str, "mg": float(dose_mg), "fed": bool(fed)})
    st.divider()
    if st.button("Clear all"):
        st.session_state.doses = []

st.subheader("Your doses")
if st.session_state.doses:
    for idx, d in enumerate(st.session_state.doses):
        colA, colB, colC, colD = st.columns([2,2,2,1])
        colA.write(f"**Time:** {d['time_str']}")
        colB.write(f"**Dose:** {d['mg']:.0f} mg")
        colC.write("**With food:** " + ("Yes" if d['fed'] else "No"))
        if colD.button("🗑️ Remove", key=f"rm_{idx}"):
            st.session_state.doses.pop(idx)
            st.experimental_rerun()
else:
    st.info("No doses yet – add some from the left panel.")

start_hour = st.number_input("Plot start hour", 0, 23, 8)
duration_h = st.slider("Duration (hours)", 6, 24, 12)

t = np.linspace(0, duration_h, duration_h * 60)
def parse_time_to_hours(t_str: str) -> float:
    hh, mm = map(int, t_str.split(":"))
    rel = (hh + mm/60) - start_hour
    if rel < 0:
        rel += 24
    return rel

total_all = np.zeros_like(t)
components = []
for d in st.session_state.doses:
    t0 = parse_time_to_hours(d["time_str"])
    ir, er, tot = dose_profile(t, d["mg"], t0, d["fed"])
    total_all += tot
    components.append((f"IR {d['mg']:.0f}mg @ {d['time_str']}" + (" (fed)" if d["fed"] else " (fasted)"), ir))
    components.append((f"ER {d['mg']:.0f}mg @ {d['time_str']}" + (" (fed)" if d["fed"] else " (fasted)"), er))

fig = plt.figure(figsize=(10,5))
plt.plot(start_hour + t, total_all, label="Total concentration")
if st.checkbox("Show IR/ER components", value=True):
    for label, y in components:
        plt.plot(start_hour + t, y, linestyle="--", label=label)
plt.xlabel("Hour of day")
plt.ylabel("Conc. (arbitrary units)")
plt.title("Medikinet CR – toy PK model")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
st.pyplot(fig)
