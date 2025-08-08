
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# ====== App config ======
st.set_page_config(page_title="Medikinet CR – All-in-One", layout="wide")
st.title("Medikinet CR – All-in-One")
st.caption("Very simplified IR+ER Gaussian toy model for educational tinkering only. Not medical advice.")

# ====== Modeling helpers ======
def gaussian_peak(t, t_peak, sigma, amplitude):
    return amplitude * np.exp(-((t - t_peak) ** 2) / (2 * (sigma ** 2)))

def dose_profile(hours_from_start, dose_mg, t0_hours, fed):
    # Split 50/50 between IR and ER (toy assumption)
    ir_mg = 0.5 * dose_mg
    er_mg = 0.5 * dose_mg

    # Tmax depends on fed/fasted
    if fed:
        ir_tmax = 1.5
        er_tmax = 4.5
    else:
        ir_tmax = 1.0
        er_tmax = 3.0

    # Peak widths
    ir_sigma = 0.5
    er_sigma = 1.2

    # Amplitudes proportional to sub-dose
    ir_amp = ir_mg * 0.8
    er_amp = er_mg * 0.6

    ir = gaussian_peak(hours_from_start, t0_hours + ir_tmax, ir_sigma, ir_amp)
    er = gaussian_peak(hours_from_start, t0_hours + er_tmax, er_sigma, er_amp)
    return ir, er, ir + er

def parse_time_to_hours(t_str, start_hour):
    # Map "HH:MM" to hours after start_hour (wrap to next day if needed)
    hh, mm = map(int, t_str.split(":"))
    rel = (hh + mm/60) - start_hour
    if rel < 0:
        rel += 24
    return rel

def simulate_total(t_axis, doses, start_hour):
    # Sum total concentration for a list of doses
    total = np.zeros_like(t_axis)
    parts = []
    for d in doses:
        t0 = parse_time_to_hours(d["time_str"], start_hour)
        ir, er, tot = dose_profile(t_axis, d["mg"], t0, d["fed"])
        total += tot
        parts.append((f"IR {d['mg']:.0f}mg @ {d['time_str']}" + (" (fed)" if d["fed"] else " (fasted)"), ir))
        parts.append((f"ER {d['mg']:.0f}mg @ {d['time_str']}" + (" (fed)" if d["fed"] else " (fasted)"), er))
    return total, parts

# ====== Shared controls ======
with st.sidebar:
    mode = st.selectbox("Mode", ["Simulator", "Optimizer"])
    st.markdown("---")
    start_hour = st.number_input("Plot start hour", 0, 23, 8)
    duration_h = st.slider("Duration (hours)", 6, 24, 12)
    show_parts_default = True if mode == "Simulator" else False

t = np.linspace(0, duration_h, duration_h * 60)  # 1-min resolution

# ====== Simulator UI ======
def simulator_ui():
    st.subheader("Simulator")
    if "sim_doses" not in st.session_state:
        st.session_state.sim_doses = []

    with st.expander("Add dose"):
        col1, col2, col3, col4 = st.columns([1,1,1,1])
        with col1:
            add_h = st.number_input("Hour", 0, 23, 8, key="sim_add_h")
        with col2:
            add_m = st.number_input("Min", 0, 59, 0, step=5, key="sim_add_m")
        with col3:
            add_mg = st.selectbox("Dose", options=[10, 20], index=1, key="sim_add_mg")
        with col4:
            add_fed = st.selectbox("With food?", options=["Fasted", "Fed"], index=0, key="sim_add_fed")
        if st.button("➕ Add dose", key="sim_btn_add"):
            t_str = f"{int(add_h):02d}:{int(add_m):02d}"
            st.session_state.sim_doses.append({"time_str": t_str, "mg": int(add_mg), "fed": add_fed == 'Fed'})
            st.success(f"Added: {add_mg} mg at {t_str} ({'with food' if add_fed == 'Fed' else 'fasted'})")

    if st.session_state.sim_doses:
        st.write("Current doses:")
        for i, d in enumerate(st.session_state.sim_doses):
            c1, c2, c3, c4 = st.columns([2,2,2,1])
            c1.write(f"Time: **{d['time_str']}**")
            c2.write(f"Dose: **{d['mg']} mg**")
            c3.write("With food: **" + ("Yes" if d['fed'] else "No") + "**")
            if c4.button("🗑️", key=f"sim_rm_{i}"):
                st.session_state.sim_doses.pop(i)
                st.experimental_rerun()
    else:
        st.info("No doses yet. Add some above.")

    total_all, components = simulate_total(t, st.session_state.sim_doses, start_hour)

    fig = plt.figure(figsize=(10,5))
    plt.plot(start_hour + t, total_all, label="Total concentration")
    if st.checkbox("Show IR/ER components", value=show_parts_default, key="sim_show_parts"):
        for label, y in components:
            plt.plot(start_hour + t, y, linestyle="--", label=label)

    plt.xlabel("Hour of day")
    plt.ylabel("Conc. (arbitrary units)")
    plt.title("Medikinet CR – Simulator")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    st.pyplot(fig)

# ====== Optimizer UI ======
def optimizer_ui():
    st.subheader("Optimizer")
    if "opt_doses" not in st.session_state:
        st.session_state.opt_doses = []

    with st.expander("Manual doses (optional)"):
        col1, col2, col3, col4 = st.columns([1,1,1,1])
        with col1:
            add_h = st.number_input("Hour", 0, 23, 8, key="opt_add_h")
        with col2:
            add_m = st.number_input("Min", 0, 59, 0, step=5, key="opt_add_m")
        with col3:
            add_mg = st.selectbox("Dose", options=[10, 20], index=1, key="opt_add_mg")
        with col4:
            add_fed = st.selectbox("With food?", options=["Fasted", "Fed"], index=0, key="opt_add_fed")
        if st.button("➕ Add dose", key="opt_btn_add"):
            t_str = f"{int(add_h):02d}:{int(add_m):02d}"
            st.session_state.opt_doses.append({"time_str": t_str, "mg": int(add_mg), "fed": add_fed == 'Fed'})
            st.success(f"Added: {add_mg} mg at {t_str} ({'with food' if add_fed == 'Fed' else 'fasted'})")

    if st.session_state.opt_doses:
        st.write("Current doses:")
        for i, d in enumerate(st.session_state.opt_doses):
            c1, c2, c3, c4 = st.columns([2,2,2,1])
            c1.write(f"Time: **{d['time_str']}**")
            c2.write(f"Dose: **{d['mg']} mg**")
            c3.write("With food: **" + ("Yes" if d['fed'] else "No") + "**")
            if c4.button("🗑️", key=f"opt_rm_{i}"):
                st.session_state.opt_doses.pop(i)
                st.experimental_rerun()
    else:
        st.info("No manual doses yet. You can also just use the optimizer.")

    st.markdown("---")
    st.header("Optimization settings")

    colw1, colw2, colw3 = st.columns([2,2,3])
    with colw1:
        target_start = st.number_input("Target window start (hour)", 0, 23, 9, key="opt_wstart")
    with colw2:
        target_end = st.number_input("Target window end (hour)", 0, 23, 19, key="opt_wend")
    with colw3:
        daily_mg_limit = st.number_input("Daily mg limit (10–120 mg)", min_value=10, max_value=120, value=40, step=10, key="opt_limit")

    opt_fed = st.checkbox("Assume doses with food (slower) for optimization", value=False, key="opt_fed")
    time_step_min = st.selectbox("Time granularity", options=[15, 30, 60], index=1, key="opt_step")

    lambda_rough = st.slider("Smoothness penalty (λ)", 0.0, 0.5, 0.05, 0.01, key="opt_lrough")
    lambda_outside = st.slider("Penalty outside window", 0.0, 1.0, 0.2, 0.05, key="opt_lout")

    def objective(total_curve, t_axis):
        # Higher is better: area inside window minus penalties
        dt = (t_axis[1] - t_axis[0])
        hours = start_hour + t_axis
        if target_end >= target_start:
            inside = (hours >= target_start) & (hours <= target_end)
        else:
            inside = (hours >= target_start) | (hours <= target_end)
        area_inside = float(np.sum(total_curve[inside]) * dt)
        area_outside = float(np.sum(total_curve[~inside]) * dt)
        rough = float(np.sum(np.diff(total_curve)**2))
        return area_inside - lambda_outside*area_outside - lambda_rough*rough

    def greedy_optimize(start_hour, duration_h, mg_limit, fed, time_step_minutes):
        # Greedy addition of 10/20 mg doses on a discrete time grid
        t_axis = np.linspace(0, duration_h, duration_h * 60)
        current = []
        current_total, _ = simulate_total(t_axis, current, start_hour)
        used_mg = 0
        improved = True
        step_h = time_step_minutes / 60.0
        times = np.arange(start_hour, start_hour + duration_h + 1e-9, step_h) % 24
        times_str = [f"{int(hh)%24:02d}:{int(round((hh%1)*60))%60:02d}" for hh in times]

        while improved:
            improved = False
            best_gain = 0.0
            best_add = None
            best_total = None
            for tstr in times_str:
                for dose in (10, 20):
                    if used_mg + dose > mg_limit:
                        continue
                    trial = current + [{"time_str": tstr, "mg": dose, "fed": fed}]
                    trial_total, _ = simulate_total(t_axis, trial, start_hour)
                    gain = objective(trial_total, t_axis) - objective(current_total, t_axis)
                    if gain > best_gain + 1e-9:
                        best_gain = gain
                        best_add = {"time_str": tstr, "mg": dose, "fed": fed}
                        best_total = trial_total
            if best_add is not None:
                current.append(best_add)
                current_total = best_total
                used_mg += best_add["mg"]
                improved = True
            else:
                break
        return current, current_total, t_axis

    if st.button("Optimize"):
        opt_doses, opt_curve, t_axis = greedy_optimize(start_hour, duration_h, int(daily_mg_limit), opt_fed, int(time_step_min))
        if not opt_doses:
            st.warning("No feasible schedule found under the given mg limit and window. Try increasing the limit or widening the window.")
        else:
            total_mg = sum(d['mg'] for d in opt_doses)
            schedule_str = ", ".join([f"{int(d['mg'])}mg @{d['time_str']}" for d in opt_doses])
            st.success("Optimized schedule (" + str(int(total_mg)) + " mg): " + schedule_str)
            if st.button("Load into manual doses"):
                st.session_state.opt_doses = opt_doses
                st.experimental_rerun()

    # Plot current (manual or loaded) schedule
    total_all, components = simulate_total(t, st.session_state.opt_doses, start_hour)
    fig = plt.figure(figsize=(10,5))
    plt.plot(start_hour + t, total_all, label="Total concentration")
    if st.checkbox("Show IR/ER components", value=show_parts_default, key="opt_show_parts"):
        for label, y in components:
            plt.plot(start_hour + t, y, linestyle="--", label=label)

    # Visualize target window
    if target_end >= target_start:
        plt.axvspan(target_start, target_end, alpha=0.12, label="Target window")
    else:
        plt.axvspan(target_start, start_hour + duration_h, alpha=0.12, label="Target window")
        plt.axvspan(start_hour, target_end, alpha=0.12)

    plt.xlabel("Hour of day")
    plt.ylabel("Conc. (arbitrary units)")
    plt.title("Medikinet CR – Optimizer")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    st.pyplot(fig)

# ====== Router ======
if mode == "Simulator":
    simulator_ui()
else:
    optimizer_ui()
