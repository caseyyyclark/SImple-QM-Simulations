"""
Streamlit app for simulating the tunneling of a one‑dimensional quantum
wave packet through a finite potential barrier.

The app exposes sliders and input widgets that let the user adjust the
parameters of both the barrier (height, width and centre) and the
initial Gaussian wave packet (width, initial position and momentum).
Once the user presses the **Run Simulation** button the time dependent
Schrödinger equation is solved numerically using the split‑step Fourier
method.  This pseudo‑spectral technique alternates between
applications of the potential operator in real space and the kinetic
operator in momentum space.  It is well suited to problems like
tunnelling where a smooth wave packet propagates over a relatively
simple potential.  A summary of the method can be found in standard
numerical quantum mechanics references — for example, search results
highlight that the split‑step (Fourier) method is a pseudo‑spectral
numerical method used to solve nonlinear partial differential equations
such as the Schrödinger equation【630314991841518†L1-L3】.

The probability density is computed as the square modulus of the
wavefunction.  According to basic quantum mechanics texts, the
probability density for finding a particle at a point \(x\) is given by
\(|\psi(x)|^2\)【386090940047661†L20-L22】.  The app plots the real and imaginary
parts of the wavefunction as well as this probability density.  The
barrier potential is superimposed on a secondary axis to give context
to the tunnelling dynamics.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt


def gaussian_wave_packet(x: np.ndarray, x0: float, sigma: float, k0: float) -> np.ndarray:
    """Return a normalised Gaussian wave packet.

    Parameters
    ----------
    x : np.ndarray
        Spatial grid on which to evaluate the packet.
    x0 : float
        Initial centre of the packet.
    sigma : float
        Width (standard deviation) of the Gaussian envelope.
    k0 : float
        Central wave number (momentum) of the plane wave component.

    Returns
    -------
    np.ndarray
        Complex array representing the initial wavefunction \(\psi(x,0)\).
    """
    # Normalisation constant for the Gaussian in one dimension.
    # (2πσ²)^(-1/4) ensures ∫|ψ|² dx = 1.
    A = (1.0 / (2.0 * np.pi * sigma ** 2)) ** 0.25
    return A * np.exp(-((x - x0) ** 2) / (4 * sigma ** 2)) * np.exp(1j * k0 * x)


def build_potential(x: np.ndarray, V0: float, centre: float, width: float) -> np.ndarray:
    """Construct a square potential barrier.

    The barrier of height ``V0`` extends from ``centre - width/2`` to
    ``centre + width/2`` and is zero elsewhere.

    Parameters
    ----------
    x : np.ndarray
        Spatial grid.
    V0 : float
        Height of the potential barrier.
    centre : float
        Position of the barrier centre.
    width : float
        Width (thickness) of the barrier.

    Returns
    -------
    np.ndarray
        Array containing the potential energy at each grid point.
    """
    half = width / 2.0
    mask = (x >= (centre - half)) & (x <= (centre + half))
    V = np.zeros_like(x)
    V[mask] = V0
    return V


def add_absorber(x: np.ndarray, x_min: float, x_max: float, width_frac: float = 0.15, strength: float = 10.0) -> np.ndarray:
    """
    Build a weak complex absorbing potential (CAP) at both domain edges.

    Parameters
    ----------
    width_frac : float
        Fraction of the total domain length used for each absorbing ramp (0<width_frac<0.5).
    strength : float
        Prefactor setting the imaginary strength of the absorber (larger => more absorption).

    Returns
    -------
    np.ndarray (complex)
        Purely imaginary potential (negative imaginary part) that damps outgoing waves.
    """
    Ltot = x_max - x_min
    ramp = width_frac * Ltot
    left_start = x_min + ramp
    right_start = x_max - ramp

    W = np.zeros_like(x, dtype=np.float64)

    # Quartic ramps (smooth)
    left_mask = x < left_start
    right_mask = x > right_start
    if ramp > 0:
        W[left_mask] = ((left_start - x[left_mask]) / ramp) ** 4
        W[right_mask] = ((x[right_mask] - right_start) / ramp) ** 4

    # Negative imaginary to absorb
    return -1j * strength * W


def split_step_fft(
    x_min: float,
    x_max: float,
    N: int,
    V0: float,
    centre: float,
    width: float,
    x0: float,
    sigma: float,
    k0: float,
    total_time: float,
    dt: float,
    use_absorber: bool = False,
    absorber_width_frac: float = 0.15,
    absorber_strength: float = 10.0,
):
    """
    Evolve a Gaussian wave packet using the split-step Fourier method.
    Units: ħ = 1, m = 1 => kinetic phase uses k^2/2.
    Returns x, V(real), times, psi_arr
    """
    # Spatial grid and momentum grid
    dx = (x_max - x_min) / N
    x = np.linspace(x_min, x_max, N, endpoint=False)  # periodic for FFT
    k = 2.0 * np.pi * np.fft.fftfreq(N, d=dx)

    # Initial wavefunction
    psi = gaussian_wave_packet(x, x0=x0, sigma=sigma, k0=k0)

    # Real barrier potential
    V_real = build_potential(x, V0=V0, centre=centre, width=width)

    # Optional absorbing edges (purely imaginary)
    if use_absorber:
        Wcap = add_absorber(x, x_min, x_max, width_frac=absorber_width_frac, strength=absorber_strength)
        V_eff = V_real + Wcap
    else:
        V_eff = V_real

    # Time axis
    n_steps = int(np.floor(total_time / dt))
    times = np.linspace(0.0, n_steps * dt, n_steps + 1)

    # Allocate storage
    psi_arr = np.zeros((n_steps + 1, N), dtype=np.complex128)
    psi_arr[0] = psi

    # Precompute exponentials
    # Potential half-step
    expV = np.exp(-1j * V_eff * dt / 2.0)
    # Kinetic full-step: phase = exp(-i * (k^2/2) * dt)
    expK = np.exp(-1j * (k ** 2) * dt / 2.0)

    # Time evolution loop
    for i in range(1, n_steps + 1):
        psi = expV * psi
        psi_k = np.fft.fft(psi)
        psi_k *= expK
        psi = np.fft.ifft(psi_k)
        psi = expV * psi
        psi_arr[i] = psi

    return x, V_real, times, psi_arr


def run_app() -> None:
    """Main entry point for the Streamlit application."""
    # If a reset has been triggered in a previous run, update parameter values
    # before any widgets are instantiated.  This avoids the Streamlit
    # restriction on modifying widget values after creation.  After updating
    # the defaults the flag is cleared so that subsequent runs proceed
    # normally.  This check must occur before any sliders or other widgets
    # are defined.
    if st.session_state.get("_reset_flag", False):
        # Default values for all sliders and selectors.  The choice of
        # these values follows the initial defaults used when the sliders
        # are first created (see definitions below).  In particular,
        # ``x0`` defaults to -15.0, which corresponds to ``-L + 5`` when
        # the domain half‑width ``L`` is 20.0.
        defaults = {
            "L": 20.0,
            "N": 512,
            "V0": 5.0,
            "width": 1.0,
            "centre": 0.0,
            "sigma": 0.5,
            "x0": -15.0,
            "k0": 5.0,
            "total_time": 2.0,
            "dt": 0.01,
            "use_absorber": False,
            "absorber_width_frac": 0.15,
            "absorber_strength": 10.0,
        }
        # Update session state with these default values only if the key is
        # not already present; this ensures that the reset does not
        # inadvertently override other state variables (e.g. from widgets that
        # might not be rendered in this run).  Use update to set multiple
        # keys atomically.
        st.session_state.update({k: v for k, v in defaults.items()})
        # Clear simulation data and internal states
        for state_key in ["sim_data", "prev_params", "idx", "play", "slider_placeholder", "plot_placeholder", "explanation_placeholder"]:
            if state_key in st.session_state:
                del st.session_state[state_key]
        # Clear the reset flag so that this block does not run on the next rerun
        st.session_state._reset_flag = False
        
    # >>> INSERT: apply "mostly reflected" preset BEFORE widgets are created
    if st.session_state.get("_preset_mostly_reflected", False):
        L_current = float(st.session_state.get("L", 20.0))
        L_new = max(L_current, 30.0)
        st.session_state.update({
            "V0": 10.0,
            "width": 1.5,
            "k0": 2.0,        # E = k0^2/2 = 2.0
            "sigma": 2.0,     # narrower Δk
            "L": L_new,
            "x0": -L_new + 5.0,
            "total_time": max(float(st.session_state.get("total_time", 2.0)), 6.0),
            "dt": min(float(st.session_state.get("dt", 0.01)), 0.01),
            # If you added the absorber, keep these; otherwise you can omit:
            "use_absorber": True,
            "absorber_width_frac": 0.15,
            "absorber_strength": 10.0,
        })
        # Clear derived state so the new preset is cleanly applied
        for state_key in ["sim_data", "prev_params", "idx", "play",
                        "slider_placeholder", "plot_placeholder",
                        "explanation_placeholder"]:
            st.session_state.pop(state_key, None)
        st.session_state._preset_mostly_reflected = False
    # <<< END INSERT 

    st.title("Quantum Wave‑Packet Tunneling in One Dimension")
    st.markdown(
        """
        This interactive tool solves the time‑dependent Schrödinger equation for a
        Gaussian wave packet encountering a finite potential barrier in one
        dimension.  You can adjust the parameters of both the barrier and the
        initial wave packet using the controls in the sidebar.  After running
        the simulation, use the **Time** slider to visualise the wavefunction at
        different moments.  The real (blue) and imaginary (red) parts of the
        wavefunction are plotted along with the probability density (green) and
        the barrier potential (black).
        
        The probability density of finding the particle at position \(x\) is
        given by $|\\psi(x,t)|^2 = \\psi^*(x,t)\\psi(x,t)$, where the complex
        conjugate $\\psi^*$ multiplies the wavefunction.  The split‑step
        Fourier method is used to integrate the Schrödinger equation in time
        because it efficiently handles the alternating kinetic and potential
        terms.
        """,
    )

    # Sidebar for parameter selection
    st.sidebar.header("Simulation Parameters")

    # Spatial domain with keys so that values can be programmatically reset
    L = st.sidebar.slider(
        "Half width of spatial domain (L)",
        5.0,
        50.0,
        20.0,
        step=1.0,
        key="L",
    )
    N = st.sidebar.select_slider(
        "Number of spatial points (power of two recommended)",
        options=[256, 512, 1024, 2048],
        value=512,
        key="N",
    )
    x_min = -L
    x_max = L

    # Barrier parameters
    st.sidebar.subheader("Barrier")
    V0 = st.sidebar.slider("Height V₀", 0.0, 50.0, 5.0, step=0.1, key="V0")
    width = st.sidebar.slider("Width", 0.1, 10.0, 1.0, step=0.1, key="width")
    centre = st.sidebar.slider(
        "Centre position",
        float(-L + 1.0),
        float(L - 1.0),
        0.0,
        step=0.1,
        key="centre",
    )

    # Wave packet parameters
    st.sidebar.subheader("Initial wave packet")
    sigma = st.sidebar.slider("Width σ", 0.1, 5.0, 0.5, step=0.1, key="sigma")
    x0 = st.sidebar.slider(
        "Initial centre x₀",
        float(-L + 1.0),
        float(L - 1.0),
        float(-L + 5.0),
        step=0.5,
        key="x0",
    )
    k0 = st.sidebar.slider("Central wave number k₀", -20.0, 20.0, 5.0, step=0.1, key="k0")

    # Time parameters
    st.sidebar.subheader("Time integration")
    # Allow a wider range for the total simulation time to explore longer dynamics
    total_time = st.sidebar.slider(
        "Total simulation time",
        0.1,
        50.0,
        2.0,
        step=0.1,
        key="total_time",
    )
    # Allow a wider range for the time step to give users more control.  A smaller Δt
    # produces more accurate results while larger values speed up the simulation at
    # the cost of accuracy.
    dt = st.sidebar.slider(
        "Time step Δt",
        0.001,
        1.00,
        0.01,
        step=0.001,
        key="dt",
    )


    # Absorbing boundaries (optional)
    st.sidebar.subheader("Boundary (optional)")
    use_absorber = st.sidebar.checkbox("Use absorbing edges (CAP)", value=st.session_state.get("use_absorber", False), key="use_absorber")
    absorber_width_frac = st.sidebar.slider(
        "CAP width (fraction of domain on each side)",
        0.01, 0.30, st.session_state.get("absorber_width_frac", 0.15), step=0.01, key="absorber_width_frac"
    )
    absorber_strength = st.sidebar.slider(
        "CAP strength", 0.0, 50.0, st.session_state.get("absorber_strength", 10.0), step=0.5, key="absorber_strength"
    )

    # --- Transmission estimate & preset ---
    with st.sidebar.expander("Transmission estimate"):
        # ħ=1, m=1 => E ≈ k0^2/2 for a narrow packet
        E = 0.5 * (k0 ** 2)
        if E < V0 and width > 0:
            kappa = np.sqrt(2.0 * (V0 - E))
            T_est = np.exp(-2.0 * kappa * width)
            st.write(f"E ≈ {E:.3f}, κ ≈ {kappa:.3f},  T_est ≈ {T_est:.2e}")
            st.caption("Heuristic for a square barrier (use to set scales).")
        elif E >= V0:
            st.write(f"E ≈ {E:.3f} ≥ V₀ ⇒ above-barrier scattering (T can be large).")
        else:
            st.write("Set width > 0 for a valid estimate.")

    if st.sidebar.button("Make it mostly reflected"):
        st.session_state._preset_mostly_reflected = True
        if hasattr(st, "rerun"):
            st.rerun()
        elif hasattr(st, "experimental_rerun"):
            st.experimental_rerun()

    # Form a dictionary of the current parameters to detect changes
    current_params = {
        "L": L,
        "N": N,
        "V0": V0,
        "width": width,
        "centre": centre,
        "sigma": sigma,
        "x0": x0,
        "k0": k0,
        "total_time": total_time,
        "dt": dt,
    }

    if "prev_params" not in st.session_state:
        st.session_state.prev_params = None
    if "sim_data" not in st.session_state:
        st.session_state.sim_data = None

    # Button to run the simulation
    if st.sidebar.button("Run Simulation") or (st.session_state.prev_params != current_params and st.session_state.prev_params is not None):
        with st.spinner("Computing wavefunction evolution..."):
            x, V, times, psi_arr = split_step_fft(
                x_min=x_min,
                x_max=x_max,
                N=int(N),
                V0=V0,
                centre=centre,
                width=width,
                x0=x0,
                sigma=sigma,
                k0=k0,
                total_time=total_time,
                dt=dt,
                use_absorber=use_absorber,
                absorber_width_frac=absorber_width_frac,
                absorber_strength=absorber_strength,
            )
        st.session_state.sim_data = {
            "x": x,
            "V": V,
            "times": times,
            "psi_arr": psi_arr,
        }
        st.session_state.prev_params = current_params
        st.success("Simulation complete.")
    # Reset button to restore default parameter values.  The callback
    # simply sets a flag in session_state and forces a rerun.  At the
    # beginning of ``run_app`` (see above) the flag is detected and the
    # default values are applied before any widgets are created.  This
    # avoids the Streamlit error that arises when modifying widget
    # variables after they have been instantiated.
    def reset_parameters():
        st.session_state._reset_flag = True
        # Trigger a rerun so the reset flag is processed on the next run
        if hasattr(st, "rerun"):
            st.rerun()
        elif hasattr(st, "experimental_rerun"):
            st.experimental_rerun()

    st.sidebar.button("Reset Parameters", on_click=reset_parameters)

    # If simulation results are available, allow time selection and plotting
    if st.session_state.sim_data is not None:
        # Bring in time module only when needed to minimise overhead
        import time

        sim = st.session_state.sim_data
        max_idx = len(sim["times"]) - 1

        # If the stored time index exceeds the number of available time slices
        # (which can happen if the user changes the total time or time step and reruns
        # the simulation), clamp it to the last valid index to avoid an IndexError.
        if "idx" in st.session_state and st.session_state.idx > max_idx:
            st.session_state.idx = max_idx

        # Initialise session state variables for the time index (idx) and playback state
        if "idx" not in st.session_state:
            st.session_state.idx = 0
        if "play" not in st.session_state:
            st.session_state.play = False


        # Persistent slider placeholder to prevent multiple sliders being created on rerun
        if "slider_placeholder" not in st.session_state:
            st.session_state.slider_placeholder = st.empty()
        slider_container = st.session_state.slider_placeholder
        selected = slider_container.slider(
            "Time stamp",
            0,
            max_idx,
            value=int(st.session_state.idx),
            step=1,
            format="%d",
            help="Time index from 0 to {} (total time = {:.3f})".format(max_idx, sim["times"][-1]),
        )
        # If the user moves the slider manually, update the playback index accordingly
        if selected != st.session_state.idx:
            st.session_state.idx = selected

        # Play and pause controls in the sidebar to avoid duplication on rerun
        with st.sidebar.container():
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("▶ Play"):
                    st.session_state.play = True
            with col2:
                if st.button("⏸ Pause"):
                    st.session_state.play = False

        # We do not advance the time index immediately; playback will be handled
        # after the plot is drawn to ensure each frame is rendered.

        # Access the selected time index
        time_index = st.session_state.idx
        t = sim["times"][time_index]
        psi = sim["psi_arr"][time_index]
        real_part = np.real(psi)
        imag_part = np.imag(psi)
        prob_density = np.abs(psi) ** 2
        
        # NEW: compute scale so |ψ|² peak sits just above wave peaks
        wave_peak = max(np.max(np.abs(real_part)), np.max(np.abs(imag_part)))
        dens_peak = np.max(prob_density)
        if dens_peak > 0:
            # If you added the sidebar controls, use them; otherwise hard-code margin = 1.10
            margin = scale_margin if 'scale_margin' in locals() else 1.10
            do_scale = scale_density if 'scale_density' in locals() else True
            density_scale = (margin * wave_peak / dens_peak) if (do_scale and wave_peak > 0) else 1.0
        else:
            density_scale = 1.0
            
        # Plot within a persistent placeholder to avoid accumulating multiple
        # charts.  Create the placeholder once in session state if it does
        # not already exist.
        if "plot_placeholder" not in st.session_state:
            st.session_state.plot_placeholder = st.empty()
        plot_placeholder = st.session_state.plot_placeholder

        # Build the figure and reserve space on the right for the legend and
        # formula annotations.  ``subplots_adjust`` with ``right=0.7`` leaves
        # 30% of the width unused by the axes; this margin will be used to
        # display the legend and formulas without overlapping the data.
        # Increase DPI for a very sharp plot (use 800 to satisfy the user's
        # request).  Reserve space on the right and bottom for annotations.
        fig, ax1 = plt.subplots(figsize=(9, 5), dpi=1200)
        fig.subplots_adjust(right=0.7, bottom=0.3)
        ax1.plot(sim["x"], real_part, label=r"$\Re[\psi(x,t)]$", color="tab:blue")
        ax1.plot(sim["x"], imag_part, label=r"$\Im[\psi(x,t)]$", color="tab:red")
        ax1.plot(sim["x"], density_scale * prob_density, label=rf"$|\psi(x,t)|^2$ × {density_scale:.2g}", color="tab:green")
        ax1.set_xlabel("Position x")
        ax1.set_ylabel("Wavefunction components / Probability density")
        ax1.set_title(f"Wave packet at t = {t:.3f}")
        
        # Match the y-axis to the *scaled* probability density
        scaled_dens_peak = density_scale * dens_peak
        wave_min = min(real_part.min(), imag_part.min(), 0.0)
        wave_max = max(real_part.max(), imag_part.max())

        headroom = 1.05  # small cushion so curves don't touch the top
        ymin = wave_min - (headroom - 1) * max(1e-12, wave_max - wave_min)
        ymax = headroom * max(wave_max, scaled_dens_peak)

        ax1.set_ylim(ymin, ymax)

        # Secondary axis for the potential
        ax2 = ax1.twinx()
        ax2.plot(sim["x"], sim["V"], label="Potential V(x)", color="black", linestyle="--", alpha=0.6)
        ax2.set_ylabel("Potential energy V(x)")

        # Combine legends and position them in the reserved right margin.  The
        # ``bbox_to_anchor`` coordinates place the legend further to the right
        # so it doesn't overlap with the plot or the formulas.  Disable the
        # frame for a clean appearance.
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(
            lines + lines2,
            labels + labels2,
            loc="upper left",
            bbox_to_anchor=(1.15, 0.9),
            frameon=False,
        )

        # Annotate the figure with formulas and descriptions.  Place the
        # formulas below the plot (in the bottom margin) to keep them
        # separate from the data.  Use a larger font size for readability.
        formula_text = (
            "$\\psi(x,0) = A \\exp\\left[-\\frac{(x - x_0)^2}{4 \\sigma^2}\\right] \\exp(i k_0 x)$\n"
            "$V(x) = V_0$ for $|x - x_c| \\leq w/2$; 0 otherwise"
        )
        fig.text(
            0.02,
            0.02,
            formula_text,
            ha='left',
            va='bottom',
            fontsize=12,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
        )

        # Display the figure in the persistent placeholder
        plot_placeholder.pyplot(fig)

        # Provide explanatory text below the plot.  Describe the dynamics,
        # define each variable in the formula and outline how changes to
        # those variables influence the tunnelling behaviour.  This text
        # appears beneath the figure (outside of the Matplotlib canvas) and
        # will not overlap with the plot or annotations.
        # Use a persistent placeholder to render the explanatory text below the plot.
        if "explanation_placeholder" not in st.session_state:
            st.session_state.explanation_placeholder = st.empty()
        explanation_placeholder = st.session_state.explanation_placeholder
        explanation_placeholder.markdown(
            """
            **Tip for tiny transmission:** Keep $\(E/V_0 \lesssim 0.2\$ and make $\(\kappa w \gtrsim 8\)$,
            where $\(E \approx k_0^2/2\), \(\kappa=\sqrt{2(V_0-E)}\)$.
            Increase $\(\\sigma\)$ to narrow the momentum spread so high-$\(k\)$ components don't hop the barrier.
            
            **Interpreting the plots.** The wavepacket begins as a Gaussian centred at $x_0$ with width $\sigma$ and momentum proportional to the wave number $k_0$.  As it propagates in time, part of the wavefunction is reflected by the potential barrier and part is transmitted through it.  The real and imaginary parts of $\\psi(x,t)$ oscillate, while the probability density $|\\psi(x,t)|^2$ highlights where the particle is most likely to be found.  When the kinetic energy ($\\propto k_0^2$) is less than the barrier height $V_0$, quantum tunnelling allows a non‑zero probability of the packet appearing on the far side of the barrier.

            **Variables in the formula.**

            - **$A$** is the normalisation constant $A=(2\pi \sigma^2)^{-1/4}$ ensuring the integral of $|\psi|^2$ over all space equals one.
            - **$x_0$** is the initial centre of the Gaussian packet.
            - **$\sigma$** controls the width (spread) of the packet; a larger $\sigma$ gives a broader packet and narrower momentum distribution.
            - **$k_0$** is the central wave number; it sets the momentum and hence the kinetic energy of the particle.
            - **$V_0$** is the height of the square potential barrier.
            - **$w$** is the barrier width (thickness); the barrier extends from $x_c - w/2$ to $x_c + w/2$ where $x_c$ (written as $x_c$ to denote a subscript) is the barrier centre.
            - **$L$** denotes half the width of the spatial domain; the simulation runs from $-L$ to $L$.
            - **$N$** is the number of spatial grid points used for the discrete Fourier transform; larger values yield smoother results but require more computation.
            - **$\Delta t$** is the integration time step and **total\_time** is the total simulation duration.

            **Impact of changing parameters.**

            - Increasing **$V_0$** or the barrier width **$w$** reduces the probability of tunnelling; a high, thick barrier reflects more of the wavepacket and allows less transmission.
            - Raising **$k_0$** (increasing the particle’s energy) enhances the likelihood of penetrating or surmounting the barrier.
            - A larger **$\sigma$** produces a more spread‑out wavepacket that moves slowly and has a narrow momentum distribution; a smaller **$\sigma$** yields a more localised packet that disperses rapidly.
            - Moving **$x_0$** changes the starting position; placing the packet further from the barrier delays the interaction.
            - Expanding **$L$** enlarges the computational domain, reducing boundary effects, while increasing **$N$** improves spatial resolution.  However, both also increase computational cost.

            **Split‑step Fourier method.**

            The exact time evolution operator for a time‑independent Hamiltonian $\\hat{H}=\\hat{T}+\\hat{V}$ over a small time step $\\Delta t$ is $e^{-i\\hat{H}\\Delta t/\\hbar}=e^{-i(\\hat{T}+\\hat{V})\\Delta t/\\hbar}$.  Because the kinetic operator $\\hat{T}$ and the potential operator $\\hat{V}$ generally do not commute, this exponential cannot be factorised exactly into separate kinetic and potential terms.  The split‑step method uses a second‑order Trotter approximation,

            $e^{-i(\\hat{T}+\\hat{V})\\Delta t/\\hbar}\\;\\approx\\;e^{-i \\hat{V} \\Delta t/(2\\hbar)}\\,e^{-i \\hat{T} \\Delta t/\\hbar}\\,e^{-i \\hat{V} \\Delta t/(2\\hbar)}$
                
            which propagates the wavefunction by a half‑step of the potential, a full step of the kinetic term, and a second half‑step of the potential.  This splitting is unitary and accurate to second order in $\\Delta t$.

            **Algorithmic steps:**

            - **Half‑step potential evolution:** multiply the wavefunction in real space by $\\exp\\bigl[-i V(x)\\Delta t/(2\\hbar)\\bigr]$.
            - **Full‑step kinetic evolution:** transform the wavefunction to momentum space via a Fourier transform, multiply by $\\exp\\bigl[-i \\hbar k^2 \\Delta t/(2m)\\bigr]$, and transform back with an inverse Fourier transform.
            - **Second half‑step potential evolution:** multiply again in real space by $\\exp\\bigl[-i V(x)\\Delta t/(2\\hbar)\\bigr]$ to complete the time step.

            **Comparison with the full evolution operator.**

            In the exact approach one would apply $e^{-i\\hat{H}\\Delta t/\\hbar}$ directly at each step.  This requires exponentiating the full Hamiltonian, which in practice means diagonalising a large matrix or using very fine finite‑difference schemes—both of which are computationally expensive.  The split‑step Fourier method avoids this by using fast Fourier transforms to handle the kinetic term and by applying the potential term as simple multiplications, offering an efficient approximation that preserves unitarity and captures the essential physics.
            """
        )

        # After rendering the current frame, handle automatic playback outside
        # of the plotting context.  This avoids rendering duplicate figures and
        # allows the UI to update smoothly between frames.  The playback speed
        # can be adjusted by changing the sleep duration; halving the sleep
        # time makes the animation approximately twice as fast.
        if st.session_state.play:
            if st.session_state.idx >= max_idx:
                st.session_state.play = False
            else:
                st.session_state.idx += 1
                time.sleep(0.025)  # shorten pause further to double the playback speed again
                # Trigger a rerun to refresh the app state
                if hasattr(st, "rerun"):
                    st.rerun()
                elif hasattr(st, "experimental_rerun"):
                    st.experimental_rerun()
                else:
                    st.session_state.play = False
                    st.warning(
                        "Automatic playback requires a Streamlit version that supports rerun functionality; please upgrade or use the slider manually.",
                    )

    else:
        st.info("Adjust parameters and click 'Run Simulation' to begin.")


if __name__ == "__main__":
    run_app()
