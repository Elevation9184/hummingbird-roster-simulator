# Streamlit app: Hummingbird‑style roster simulator (v7)
# -----------------------------------------------------------
# Changelog
#   • Threshold lines now plotted *on* integer ticks (no -0.5 shift)

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom, poisson
from matplotlib.ticker import MaxNLocator

# -----------------------------
# Page config & global styling
# -----------------------------
st.set_page_config(page_title="Hummingbird‑style Roster Simulator", layout="wide")

# Inject custom CSS once
st.markdown(
    """
    <style>
        /* Reduce Streamlit top padding and tighten layout */
        .css-1lcbmhc.e1fqkh3o2, .block-container {
            padding-top: 1rem;
            margin-top: 0;
        }
        /* Unified font for widgets and main content */
        html, body, [class*="css"], .stSlider > div, .stNumberInput > div {
            font-family: "DejaVu Sans", sans-serif;
            font-size: 17px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

plt.rcParams.update({
    "font.size": 13,
    "axes.titlesize": 17,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
})

# -----------------------------
# Helper functions
# -----------------------------

def generate_roster(n_nurses: int, n_shifts: int, fill_rates: np.ndarray, rng: np.random.Generator):
    roster = np.zeros((n_nurses, n_shifts), dtype=np.uint8)
    for i, f in enumerate(fill_rates):
        n_work = int(round(f * n_shifts))
        roster[i, rng.choice(n_shifts, n_work, replace=False)] = 1
    return roster


def run_simulation(fill_rate_1: float, fill_rate_others: float, n_incidents: int,
                   n_shifts: int = 731, n_nurses: int = 38, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    fill_rates = np.full(n_nurses, fill_rate_others, dtype=float)
    fill_rates[0] = fill_rate_1
    roster = generate_roster(n_nurses, n_shifts, fill_rates, rng)

    incident_vec = np.zeros(n_shifts, dtype=np.uint8)
    incident_vec[rng.choice(n_shifts, n_incidents, replace=False)] = 1

    counts = roster @ incident_vec
    winner_idx = int(np.argmax(counts))
    winner_fill = fill_rates[winner_idx]
    c_max = counts[winner_idx]

    return roster, incident_vec, counts, winner_idx, winner_fill, c_max


def build_compressed_mapping(winner_incident_positions):
    # 0-based mapping; display positions will be +1
    return {idx: new_x for new_x, idx in enumerate(sorted(winner_incident_positions))}


def plot_compressed_roster(roster, incident_vec, winner_idx, c_max,
                           winner_fill, n_incidents, title_suffix=""):
    n_nurses, n_shifts = roster.shape

    winner_positions = np.where((roster[winner_idx] == 1) & (incident_vec == 1))[0]
    mapping = build_compressed_mapping(winner_positions)

    # Dynamic figure height: 0.4 inch per nurse, min 6 inch
    fig_height = max(6, 0.4 * n_nurses)
    fig, ax = plt.subplots(figsize=(14, fig_height))
    rng_jitter = np.random.default_rng(42)

    for nurse in range(n_nurses):
        y = nurse + 1
        incident_cols = np.where((roster[nurse] == 1) & (incident_vec == 1))[0]
        x_comp = [mapping[idx] + 1 for idx in incident_cols if idx in mapping]
        if not x_comp:
            continue
        if nurse != winner_idx:
            x_comp = [x + rng_jitter.uniform(-0.1, 0.1) for x in x_comp]
        marker = "s" if nurse == winner_idx else "x"
        size = 150 if nurse == winner_idx else 65
        color = "red" if nurse == winner_idx else "blue"
        ax.scatter(x_comp, [y] * len(x_comp), marker=marker, s=size, c=color, alpha=0.85)

    ax.set_title(
        f"Compressed roster view — {title_suffix}: Nurse {winner_idx+1} at {c_max} co‑occurrences",
        pad=6,
    )
    ax.set_xlabel("Compressed incident timeline (winner nurse only)")

    # Y-axis labels and limits
    ax.set_yticks(range(1, n_nurses + 1))
    ax.set_yticklabels([f"Nurse {i}" for i in range(1, n_nurses + 1)])
    ax.invert_yaxis()
    ax.set_ylim(n_nurses + 0.5, 0.5)

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(alpha=0.15, linewidth=0.5)

    # Compute thresholds
    p_winner = winner_fill
    lambda_winner = p_winner * n_incidents

    binom_cut = int(binom.ppf(0.9999, n_incidents, p_winner))
    poisson_cut = int(poisson.ppf(0.9999, lambda_winner))

    # Extend x-axis to show thresholds (timeline starts at 1)
    right_edge = max(len(mapping), binom_cut, poisson_cut)
    ax.set_xlim(0.5, right_edge + 0.5)
    ax.set_xticks(list(range(1, right_edge + 1)))

    # Plot threshold lines directly on integer positions
    ax.axvline(binom_cut, color="purple", linestyle="--", linewidth=2,
               label=f"Binomial 99.99% ≈ {binom_cut}")
    ax.axvline(poisson_cut, color="green", linestyle="--", linewidth=2,
               label=f"Poisson 99.99% ≈ {poisson_cut}")

    ax.legend(loc="upper right", framealpha=0.92)
    plt.tight_layout()
    return fig

# -----------------------------
# Streamlit UI
# -----------------------------

st.sidebar.header("Simulation controls")

nurse1_fill = st.sidebar.slider("Nurse 1 shift fill (%)", 10, 100, 50) / 100.0
other_fill = st.sidebar.slider("Other nurses shift fill (%)", 10, 100, 33) / 100.0
n_incidents = st.sidebar.slider("Incident count", 5, 100, 25)

runs = st.sidebar.number_input(
    "Number of simulation runs (searches for the longest streak)",
    min_value=1, step=1, value=1,
)

seed_str = st.sidebar.text_input("Optional random‑seed (leave blank for random)")
button = st.sidebar.button("Simulate")

st.title("🩺 Hummingbird‑style Roster Simulator")
st.markdown(
    "Adjust the shift‑fill sliders, hit **Simulate**, and watch how a long string of "
    "incident overlaps can appear **by chance** when one nurse works just a bit more "
    "than the others."
)

if button:
    rng = np.random.default_rng(int(seed_str) if seed_str else None)
    best_result = None
    for _ in range(int(runs)):
        outcome = run_simulation(nurse1_fill, other_fill, n_incidents, rng=rng)
        if best_result is None or outcome[5] > best_result[5]:
            best_result = outcome

    roster, incident_vec, counts, winner_idx, winner_fill, c_max = best_result
    fig = plot_compressed_roster(
        roster,
        incident_vec,
        winner_idx,
        c_max,
        winner_fill,
        n_incidents,
        title_suffix=f"best of {runs} run{'s' if runs!=1 else ''}"
    )
    st.pyplot(fig, use_container_width=True)
else:
    st.info("👈 Set your parameters and click *Simulate* to begin.")
