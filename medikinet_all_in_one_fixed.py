
import json
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# ===== App config =====
st.set_page_config(page_title="Medikinet CR – All‑in‑One (fixed)", layout="wide")
st.title("Medikinet CR – All‑in‑One (fixed)")
st.caption("Simplified IR+ER Gaussian toy model. For educational tinkering only — not medical advice.")

# ===== Core model =====
def gaussian_peak(t, t_peak, sigma, amplitude):
    return amplitude * np.exp(-((t - t_peak) ** 2) / (2 * (sigma ** 2)))

def dose_profile(hours_from_start, dose_mg, t0_hours, fed):
    # split 50/50 (toy)
    ir_mg = 0.5 * dose_mg
    er_mg = 0.5 * dose_mg
    # tmax
    ir_tmax, er_tmax = (1.5, 4.5) if fed else (1.0, 3.0)
    # widths
    ir_sigma, er_sigma = 0.5, 1.2
    # amplitudes
    ir_amp = ir_mg * 0.8
    er_amp = er_mg * 0.6
    ir = gaussian_peak(hours_from_start, t0_hours + ir_tmax, ir_sigma, ir_amp)
    er = gaussian_peak(hours_from_start, t0_hours + er_tmax, er_sigma, er_amp)
    return ir, er, ir + er

def parse_time_to_hours(t_str, start_hour):
    hh, mm = map(int, t_str.split(":"))
    # Relative hours from plot start; allow negative values (earlier than start)
    rel = (hh + mm/60) - start_hour
    return rel

def simulate_total(t_axis, doses, start_hour):
    total = np.zeros_like(t_axis)
    parts = []
    for d in doses:
        t0 = parse_time_to_hours(d["time_str"], start_hour)
        ir, er, tot = dose_profile(t_axis, d["mg"], t0, d["fed"])
        total += tot
        parts.append((f"IR {d['mg']}mg @ {d['time_str']}" + (" (fed)" if d["fed"] else " (fasted)"), ir))
        parts.append((f"ER {d['mg']}mg @ {d['time_str']}" + (" (fed)" if d["fed"] else " (fasted)"), er))
    return total, parts


def compute_t_end(total_curve, t_axis, start_hour):
    """Return latest t (hours-from-start) to plot.
    Stop at 23:00 or when curve ~0 (1% of peak), whichever is earlier.
    """
    import numpy as _np
    if total_curve is None or _np.allclose(total_curve, 0):
        return 0.0
    peak = float(_np.max(total_curve))
    if peak <= 1e-12:
        return 0.0
    eps = 0.01 * peak
    idx = _np.where(total_curve > eps)[0]
    if idx.size == 0:
        return 0.0
    # small margin to show the tail
    last_t = float(t_axis[int(idx[-1])]) + 0.25
    # cap by 23:00
    return min(last_t, 23.0 - start_hour)

def compute_t_min(doses, start_hour):
    """Return earliest time (possibly negative) to include on the t-axis
    so that we see peaks from doses taken before start_hour.
    """
    t_min = 0.0
    for d in doses or []:
        # relative time of taking the dose
        hh, mm = map(int, d["time_str"].split(":"))
        t0 = (hh + mm/60) - start_hour  # can be negative
        # tmax and sigmas
        ir_tmax, er_tmax = (1.5, 4.5) if d.get("fed", False) else (1.0, 3.0)
        ir_sigma, er_sigma = 0.5, 1.2
        # earliest relevant times ~ (center - 3*sigma)
        cand_ir = t0 + ir_tmax - 3*ir_sigma
        cand_er = t0 + er_tmax - 3*er_sigma
        t_min = min(t_min, cand_ir, cand_er)
    # clamp to a sensible floor to avoid huge canvases
    return max(t_min, -12.0)

def safe_trapz(y, x):
    y = np.asarray(y); x = np.asarray(x)
    if y.size < 2 or x.size < 2: return 0.0
    return float(np.trapz(y, x))

# ===== Shared controls =====
with st.sidebar:
    mode = st.selectbox("Mode", ["Simulator", "Optimizer"], index=1)
    start_hour = st.number_input("Plot start hour", 0, 23, 8)
    duration_h = st.slider("Duration (hours)", 6, 24, 12)

t = np.linspace(0, duration_h, int(duration_h * 60))  # minute grid

# ===== Simulator =====
def simulator_ui():
    st.subheader("Simulator")
    if "sim_doses" not in st.session_state:
        st.session_state.sim_doses = []
    with st.expander("Add dose"):
        c1,c2,c3,c4 = st.columns(4)
        with c1: h = st.number_input("Hour", 0, 23, 8, key="sim_h")
        with c2: m = st.number_input("Min", 0, 59, 0, step=5, key="sim_m")
        with c3: mg = st.selectbox("Dose", [10,20], index=1, key="sim_mg")
        with c4: fed = st.selectbox("With food?", ["Fasted","Fed"], index=0, key="sim_fed")
        if st.button("➕ Add dose", key="sim_add"):
            st.session_state.sim_doses.append({"time_str": f"{int(h):02d}:{int(m):02d}", "mg": int(mg), "fed": fed=="Fed"})
    # Show and manage current doses
    if st.session_state.sim_doses:
        st.write("Current doses:")
        for i, d in enumerate(list(st.session_state.sim_doses)):
            c1, c2, c3, c4 = st.columns([2,2,2,1])
            c1.write(f"Time: **{d['time_str']}**")
            c2.write(f"Dose: **{d['mg']} mg**")
            c3.write("With food: **" + ("Yes" if d['fed'] else "No") + "**")
            if c4.button("🗑 Remove", key=f"sim_rm_{i}"):
                st.session_state.sim_doses.pop(i)
                try:
                    st.rerun()
                except Exception:
                    st.experimental_rerun()
        # dynamic time axis to include pre-start doses
        t_min = compute_t_min(st.session_state.sim_doses, start_hour)
        t = np.linspace(t_min, duration_h, int((duration_h - t_min)*60))
        total, parts = simulate_total(t, st.session_state.sim_doses, start_hour)
        t_end = compute_t_end(total, t, start_hour)
        t = np.linspace(t_min, t_end, max(2, int((t_end - t_min)*60)))
        total, parts = simulate_total(t, st.session_state.sim_doses, start_hour)
    else:
        # no doses; default axis
        t = np.linspace(0, duration_h, int(duration_h*60))
        total, parts = np.zeros_like(t), []
        st.info("No doses yet.")
    fig = plt.figure(figsize=(8,4))
    plt.plot(start_hour+t, total, label="Total concentration")
    if st.checkbox("Show IR/ER components", True, key="sim_show"):
        for lbl, y in parts:
            plt.plot(start_hour+t, y, "--", label=lbl)
    plt.xlabel("Hour of day"); plt.ylabel("Conc. (arb.)"); plt.title("Simulator")
    plt.grid(True, linestyle="--", alpha=0.5); plt.legend()
    if not st.session_state.get("sim_show", True) and st.session_state.sim_doses:
        for d in st.session_state.sim_doses:
            hh, mm = map(int, d["time_str"].split(":"))
            x = hh + mm/60.0
            plt.axvline(x, alpha=0.2, linestyle=":")
            if (start_hour + t[0]) <= x <= (start_hour + t[-1]):
                import numpy as _np
                y = _np.interp(x - start_hour, t, total)
                plt.scatter([x], [y], marker="o")
    st.pyplot(fig, use_container_width=True)

# ===== Optimizer =====
def objective(total_curve, t_axis, target_start, target_end, lam_out, lam_rough, lam_peak, start_hour):
    hours = start_hour + t_axis
    if target_end >= target_start:
        inside = (hours >= target_start) & (hours <= target_end)
    else:
        inside = (hours >= target_start) | (hours <= target_end)
    area_in  = safe_trapz(total_curve[inside],  t_axis[inside])
    area_out = safe_trapz(total_curve[~inside], t_axis[~inside])
    dcdt = np.gradient(total_curve, t_axis)
    rough = float(np.trapz(dcdt**2, t_axis))
    peak = float(np.max(total_curve))
    return area_in - lam_out*area_out - lam_rough*rough - lam_peak*peak

def times_in_window_grid(start_hour, duration_h, step_min, target_start, target_end, buffer_h):
    step_h = step_min/60.0
    grid = (np.arange(start_hour, start_hour+duration_h+1e-9, step_h) % 24)
    def in_expanded(h):
        s, e = target_start%24, target_end%24
        if e >= s: return (h >= s-buffer_h) and (h <= e+buffer_h)
        return (h >= s-buffer_h) or (h <= e+buffer_h)
    cands = [h for h in grid if in_expanded(h)]
    return [f"{int(h)%24:02d}:{int(round((h%1)*60))%60:02d}" for h in cands]

def greedy_optimize(start_hour, duration_h, mg_limit, fed, step_min,
                    lam_out, lam_rough, lam_peak,
                    target_start, target_end, buffer_h, min_gap_min):
    t_axis = np.linspace(0, duration_h, int(duration_h * 60))
    current = []
    current_total, _ = simulate_total(t_axis, current, start_hour)
    used_mg = 0
    cand_times = times_in_window_grid(start_hour, duration_h, step_min, target_start, target_end, buffer_h)

    def violates_gap(tstr, picks):
        # require min gap between dose TAKEN times
        def parse(tstr): 
            hh, mm = map(int, tstr.split(":")); return hh + mm/60.0
        for d in picks:
            if abs((parse(tstr) - parse(d['time_str']) + 12) % 24 - 12) < (min_gap_min/60.0):
                return True
        return False

    def score(curve):
        return objective(curve, t_axis, target_start, target_end, lam_out, lam_rough, lam_peak, start_hour)

    while True:
        base = score(current_total)
        best_gain, best = 0.0, None
        for tstr in cand_times:
            for dose in (10,20):
                if used_mg + dose > mg_limit: continue
                if violates_gap(tstr, current): continue
                trial = current + [{"time_str": tstr, "mg": dose, "fed": fed}]
                trial_total, _ = simulate_total(t_axis, trial, start_hour)
                gain = score(trial_total) - base
                if gain > best_gain + 1e-12:
                    best_gain, best = gain, {"time_str": tstr, "mg": dose, "fed": fed, "_total": trial_total}
        if best is None: break
        current.append({k:v for k,v in best.items() if k!='_total'})
        current_total = best["_total"]
        used_mg += best["mg"]
        if used_mg + 10 > mg_limit: break
    return current, current_total, t_axis

def trim_to_mg_limit(doses, t_axis, start_hour, mg_limit,
                        lam_out, lam_rough, lam_peak, target_start, target_end):
    """Ensure sum(mg) <= mg_limit by removing the least harmful doses first."""
    def score(curve):
        return objective(curve, t_axis, target_start, target_end, lam_out, lam_rough, lam_peak, start_hour)
    doses = doses[:]
    if not doses: 
        return doses
    base_curve, _ = simulate_total(t_axis, doses, start_hour)
    base_score = score(base_curve)
    while sum(d['mg'] for d in doses) > mg_limit and doses:
        best_idx = None
        best_drop = float('inf')
        # pick the dose whose removal hurts the score the least
        for i in range(len(doses)):
            trial = doses[:i] + doses[i+1:]
            trial_curve, _ = simulate_total(t_axis, trial, start_hour)
            s = score(trial_curve)
            drop = base_score - s
            if drop < best_drop - 1e-12:
                best_drop = drop
                best_idx = i
                best_trial_curve = trial_curve
                best_trial_score = s
        if best_idx is None:
            break
        doses.pop(best_idx)
        base_curve = best_trial_curve
        base_score = best_trial_score
    return doses

def refine_split_twenty(doses, t_axis, start_hour, step_min,
                        lam_out, lam_rough, lam_peak, target_start, target_end, min_gap_min):
    if not doses: return doses, simulate_total(t_axis, doses, start_hour)[0]
    def score(curve): 
        return objective(curve, t_axis, target_start, target_end, lam_out, lam_rough, lam_peak, start_hour)
    grid = times_in_window_grid(start_hour, int(t_axis[-1]), step_min, target_start, target_end, 0.0)
    def violates_gap(tstr, picks):
        def parse(tstr): hh, mm = map(int, tstr.split(":")); return hh + mm/60.0
        for d in picks:
            if abs((parse(tstr)-parse(d['time_str'])+12)%24-12) < (min_gap_min/60.0):
                return True
        return False
    best = doses[:]
    best_curve, _ = simulate_total(t_axis, best, start_hour); best_score = score(best_curve)
    improved = True
    while improved:
        improved = False
        for i, d in enumerate(list(best)):
            if d["mg"] != 20: continue
            base = best[:i] + best[i+1:]
            for t1 in grid:
                if violates_gap(t1, base): continue
                for t2 in grid:
                    if violates_gap(t2, base + [{"time_str": t1,"mg":10,"fed":d["fed"]}]): continue
                    trial = base + [{"time_str": t1, "mg": 10, "fed": d["fed"]}, {"time_str": t2, "mg": 10, "fed": d["fed"]}]
                    trial_curve, _ = simulate_total(t_axis, trial, start_hour)
                    s = score(trial_curve)
                    if s > best_score + 1e-12:
                        best, best_curve, best_score = trial, trial_curve, s
                        improved = True
                        break
                if improved: break
    return best, best_curve


def optimizer_ui():
    st.subheader("Optimizer")
    # inputs
    c1,c2,c3 = st.columns(3)
    with c1:
        target_start = st.number_input("Target start", 0, 23, 9)
        lambda_out  = st.slider("Penalty outside window", 0.0, 10.0, 0.5, 0.1)
    with c2:
        target_end   = st.number_input("Target end", 0, 23, 19)
        lambda_rough = st.slider("Smoothness penalty (λ)", 0.0, 1.0, 0.0, 0.01,
                                  help="Penalizes rapid fluctuations (wiggles). 0 = off, 1 = strong.")
    with c3:
        daily_limit  = st.number_input("Daily mg limit", 10, 120, 40, 10)
        lambda_peak  = st.slider("Peak penalty (λ_peak)", 0.0, 5.0, 1.0, 0.1)
    c4,c5,c6 = st.columns(3)
    with c4:
        step_min = st.selectbox("Time granularity (min)", [15,30,60], index=1)
    with c5:
        cand_buffer_h = st.slider("Candidate buffer ±h", 0.0, 4.0, 2.0, 0.5,
                                 help="How far OUTSIDE your target window to consider dose times. Example: window 9–19 with buffer 2h allows candidates from 7 to 21. Useful to place a dose just before the window to ramp up coverage.")
    with c6:
        min_gap_min = st.slider("Min gap between doses (min)", 0, 240, 60, 15)
    fed = st.checkbox("Assume doses with food (slower)", False)
    debug = st.checkbox("Show debug info", False)
    auto_opt = st.checkbox("Auto‑optimize on change", True)
    run_click = st.button("Optimize")

    # Decide whether to run the optimizer
    should_run = auto_opt or run_click

    if should_run:
        # run greedy + refine
        opt_doses, opt_curve, t_axis = greedy_optimize(start_hour, duration_h, int(daily_limit), fed, int(step_min),
                                                       lambda_out, lambda_rough, lambda_peak,
                                                       target_start, target_end, cand_buffer_h, int(min_gap_min))
        opt_doses, opt_curve = refine_split_twenty(
        opt_doses, t_axis, start_hour, int(step_min),
        lambda_out, lambda_rough, lambda_peak,
        target_start, target_end, int(min_gap_min)
    )
    # Hard cap: never exceed mg limit
    if sum(d['mg'] for d in opt_doses) > int(daily_limit):
        opt_doses = trim_to_mg_limit(opt_doses, t_axis, start_hour, int(daily_limit),
                                     lambda_out, lambda_rough, lambda_peak, target_start, target_end)
        # recompute curve after trimming
        opt_curve, _ = simulate_total(t_axis, opt_doses, start_hour)
        st.session_state["_opt_last"] = {
            "doses": opt_doses,
            "curve": opt_curve,
            "t_axis": t_axis,
            "params": {
                "start_hour": start_hour, "duration_h": duration_h
            }
        }
    else:
        # reuse last result if available
        last = st.session_state.get("_opt_last", None)
        if last is not None:
            opt_doses = last["doses"]
            opt_curve = last["curve"]
            t_axis    = last["t_axis"]
        else:
            opt_doses, opt_curve, t_axis = [], np.zeros_like(np.linspace(0, duration_h, int(duration_h*60))), np.linspace(0, duration_h, int(duration_h*60))

    # show result string
    total_mg = sum(d["mg"] for d in opt_doses)
    if total_mg > int(daily_limit):
        opt_doses = trim_to_mg_limit(opt_doses, np.linspace(0, duration_h, int(duration_h*60)), start_hour, int(daily_limit),
                                     lambda_out, lambda_rough, lambda_peak, target_start, target_end)
        total_mg = sum(d["mg"] for d in opt_doses)
    if opt_doses:
        schedule_str = ", ".join([f"{d['mg']}mg @{d['time_str']}" for d in opt_doses])
        st.success(f"Optimized schedule ({total_mg} / {int(daily_limit)} mg): {schedule_str}")
    else:
        st.warning("No positive-gain schedule under current penalties. Tip: lower λ_smooth/λ_peak or widen the window/buffer.")

    # chart (always render something)
    t_min = compute_t_min(opt_doses, start_hour) if opt_doses else 0.0
    # first pass to estimate end
    t_coarse = np.linspace(t_min, duration_h, int((duration_h - t_min)*60))
    total_coarse, _ = simulate_total(t_coarse, opt_doses, start_hour) if opt_doses else (np.zeros_like(t_coarse), [])
    t_end = compute_t_end(total_coarse, t_coarse, start_hour) if opt_doses else min(duration_h, 23.0 - start_hour)
    t_plot = np.linspace(t_min, t_end, max(2, int((t_end - t_min)*60)))
    total_all, components = simulate_total(t_plot, opt_doses, start_hour) if opt_doses else (np.zeros_like(t_plot), [])
    fig = plt.figure(figsize=(8,4))
    plt.plot(start_hour+t_plot, total_all, label="Total concentration")
    show_parts = st.checkbox("Show IR/ER components", True, key="opt_show")
    if show_parts and components:
        for lbl, y in components:
            plt.plot(start_hour+t_plot, y, "--", label=lbl)
    if (not show_parts) and opt_doses:
        for d in opt_doses:
            hh, mm = map(int, d["time_str"].split(":"))
            x = hh + mm/60.0
            plt.axvline(x, alpha=0.2, linestyle=":")
            if (start_hour + t_plot[0]) <= x <= (start_hour + t_plot[-1]):
                import numpy as _np
                y = _np.interp(x - start_hour, t_plot, total_all)
                plt.scatter([x], [y], marker="o")

    if target_end >= target_start:
        plt.axvspan(target_start, target_end, alpha=0.12, label="Target window")
    else:
        plt.axvspan(target_start, start_hour+duration_h, alpha=0.12, label="Target window")
        plt.axvspan(start_hour, target_end, alpha=0.12)

    plt.xlabel("Hour of day"); plt.ylabel("Conc. (arb.)"); plt.title("Optimizer")
    plt.grid(True, linestyle="--", alpha=0.5); plt.legend()

    # --- Dynamic x-limit: end at 23:00 or earlier if curve ~ 0 ---
    try:
        x_start = start_hour + t_plot[0] if 't_plot' in locals() else start_hour + t[0]
        x_time  = start_hour + (t_plot if 't_plot' in locals() else t)
        y_total = total_all if 'total_all' in locals() else total
        max_y = float(np.max(y_total)) if y_total.size else 0.0
        thr = max(1e-6, 0.02 * max_y)
        last_idx = None
        if y_total.size:
            nz = np.where(y_total > thr)[0]
            if nz.size:
                last_idx = nz[-1]
        if last_idx is not None:
            x_end_curve = float(x_time[int(last_idx)]) + 0.5  # small padding
        else:
            x_end_curve = float(x_time[-1]) if y_total.size else (start_hour + (duration_h if 'duration_h' in locals() else 12))
        x_end_clock = 23.0
        x_end = min(x_end_curve, x_end_clock)
        if x_end <= x_start + 0.5:
            x_end = min(x_end_clock, x_start + 1.0)  # ensure some width
        plt.xlim(x_start, x_end)
    except Exception:
        pass

    # --- Dose markers when components are hidden ---

    show_parts = st.session_state.get("opt_show", True)
    if not show_parts and opt_doses:
        for d in opt_doses:
            hh, mm = map(int, d["time_str"].split(":"))
            t0 = (hh + mm/60) - start_hour
            plt.axvline(start_hour + t0, linestyle=":", alpha=0.7)
            idx = int(np.argmin(np.abs(t_plot - t0)))
            if 0 <= idx < len(t_plot):
                plt.scatter(start_hour + t_plot[idx], total_all[idx])
    st.pyplot(fig, use_container_width=True)

    if debug:
        st.json({"picked_doses": opt_doses, "total_mg": total_mg})

# ===== Router =====
if mode == "Simulator":
    simulator_ui()
else:
    optimizer_ui()
