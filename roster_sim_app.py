# Streamlit app: Hummingbird‑style roster simulator (v11)
# -----------------------------------------------------------
# Changelog (v11)
#   • Legend title now uses red‑square emoji (🟥) so colour survives across back‑ends
#   • Added fallback: if emoji fonts are absent the label still renders, just not coloured
#   • No functional logic changes — UI/arrow/title behaviour unchanged
# -----------------------------------------------------------

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom, poisson
from matplotlib.ticker import MaxNLocator
import matplotlib.lines as mlines

# -----------------------------
# Page config & global styling
# -----------------------------
st.set_page_config(page_title="Hummingbird‑style Roster Simulator", layout="wide")

st.markdown(
    """
    <style>
        .block-container {padding-top: 0.8rem;}
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
    return {idx: new_x for new_x, idx in enumerate(sorted(winner_incident_positions))}


def plot_compressed_roster(roster, incident_vec, winner_idx, c_max,
                           winner_fill, n_incidents, title_suffix=""):
    n_nurses, n_shifts = roster.shape

    winner_positions = np.where((roster[winner_idx] == 1) & (incident_vec == 1))[0]
    mapping = build_compressed_mapping(winner_positions)
    fig_height = max(6, 0.4 * n_nurses)
    fig, ax = plt.subplots(figsize=(14, fig_height))
    rng_jitter = np.random.default_rng(42)

    # Scatter plot of overlaps
    for nurse in range(n_nurses):
        y = nurse + 1
        incident_cols = np.where((roster[nurse] == 1) & (incident_vec == 1))[0]
        x_comp = [mapping[idx] + 1 for idx in incident_cols if idx in mapping]
        if not x_comp:
            continue
        if nurse != winner_idx:
            x_comp = [x + rng_jitter.uniform(-0.1, 0.1) for x in x_comp]
        marker = 's' if nurse == winner_idx else 'x'
        size = 150 if nurse == winner_idx else 65
        color = 'red' if nurse == winner_idx else 'blue'
        ax.scatter(x_comp, [y] * len(x_comp), marker=marker, s=size, c=color, alpha=0.85)

    # Axes & title
    ax.set_title(
        f"Compressed roster view — {title_suffix}: Nurse {winner_idx+1} ({c_max} events)\nSo where should (statistical) suspicion start from where there is no direct evidence?",
        pad=6,
    )

    ax.set_xlabel("Compressed incident timeline (winner nurse only)")
    ax.set_yticks(range(1, n_nurses + 1))
    ax.set_yticklabels([f"Nurse {i}" for i in range(1, n_nurses + 1)])
    ax.invert_yaxis()
    ax.set_ylim(n_nurses + 0.5, 0.5)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(alpha=0.15, linewidth=0.5)

    # Thresholds
    p_winner = winner_fill
    binom_cut = int(binom.ppf(0.9999, n_incidents, p_winner))
    poisson_cut = int(poisson.ppf(0.9999, p_winner * n_incidents))
    right_edge = max(len(mapping), binom_cut, poisson_cut)
    ax.set_xlim(0.5, right_edge + 0.5)
    ax.set_xticks(list(range(1, right_edge + 1)))
    # Add in
    # “Intuitive” 1-in-20 binomial line (95th percentile)
    binom_95 = int(binom.ppf(0.95, n_incidents, p_winner))
    ax.axvline(binom_95,
            color='orange',
            alpha=0.6,
            linestyle='--',
            linewidth=2,
            label=f"Binomial 95 % ≈ {binom_95}")
    ax.axvline(binom_cut, color='purple', linestyle='--', linewidth=2, alpha=0.6,
               label=f"Binomial 99.99% ≈ {binom_cut}")
    ax.axvline(poisson_cut, color='green', linestyle='--', linewidth=2, 
               label=f"Poisson 99.99% ≈ {poisson_cut}")
    
    # … annotate binomial 95% on-plot for clarity
    y_max = ax.get_ylim()[1]
    ax.text(binom_95,    # a small horizontal offset
        y_max + 8,         # near the top of the plot
        "Model-mismatch zone:\nUsing a coin-flip ruler in a Poisson world",
        rotation=270,
        va='top',
        ha='center',
        fontsize=10,
        color='black',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='orange'))
    
    # … annotate 99.99% binomial on-plot for clarity
    y_max = ax.get_ylim()[1]
    ax.text(binom_cut,    # a small horizontal offset
        y_max + 8,         # near the top of the plot
        "Strict '1-in-10,000' threshold (Binomial)\nStill notable risk of false positives",
        rotation=270,
        va='top',
        ha='center',
        fontsize=10,
        color='purple',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='purple'))

    # … annotate 99.99% poisson on-plot for clarity
    y_max = ax.get_ylim()[1]
    ax.text(poisson_cut,    # a small horizontal offset
        y_max + 8,         # near the top of the plot
        "Investigation threshold (Poisson)\nCorrect model—high bar due to no direct evidence",
        rotation=270,
        va='top',
        ha='center',
        fontsize=10,
        color='green',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='green'))

    # Prosecutor’s Poisson Fallacy arrow & centred box
    y_fallacy = 5.5
    mid_x = (binom_95 + poisson_cut) / 2
    ax.annotate('', xy=(poisson_cut, y_fallacy), xytext=(binom_95, y_fallacy),
                arrowprops=dict(arrowstyle='<->', color='dimgray', lw=2), zorder=1)
    
    ax.text(mid_x, y_fallacy, "Prosecutor's\nPoisson Fallacy Region", ha='center', va='center',
            rotation=0, zorder=2,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='black'))

    # Legend — red squares inside brackets using emoji (robust across back‑ends)
    legend_title = "Red Line Stats Thresholds"
    ax.legend(loc='upper right', framealpha=0.92, title=legend_title)

    # place behind everything (zorder=0)
    ax.axvspan(binom_95, poisson_cut,
            color='grey', alpha=0.07, zorder=0)    
    
    ax.text(
        8, y_max +  8,
        "Why Poisson, not Binomial?\n\n"
        "Incidents in hospitals aren't like coin flips.\n"
        "They're like random raindrops falling over time:\n"
        "Workers with more shifts have wider 'umbrellas',\n"
        "so naturally they catch more drops.\n\n"
        "Treating incidents like biased coin flips risks\n" 
        "false accusations based purely on workload.\n\n"
        "A cluster of overlaps isn't suspicious by itself—\n"
        "especially when there's no direct evidence.",
        ha='center',
        va='center',
        fontsize=10,
        color='darkgreen',
        bbox=dict(
            boxstyle='round,pad=0.5',
            facecolor='honeydew',
            edgecolor='green',
            linewidth=1.5,
            alpha=0.85
        )
    )

    plt.tight_layout()
    return fig

# -----------------------------
# Streamlit UI
# -----------------------------

st.title("🩺 Hummingbird‑style Roster Simulator")
st.markdown(
    "Adjust the shift‑fill sliders, hit **Simulate**, and watch how a long string of "
    "incident overlaps can appear **by chance** when one nurse works just a bit more "
    "than the others."
)

st.sidebar.header("Simulation controls")

nurse1_fill = st.sidebar.slider("Nurse 1 shift fill (%)", 10, 100, 60) / 100.0
other_fill = st.sidebar.slider("Other nurses shift fill (%)", 10, 100, 33) / 100.0
n_incidents = st.sidebar.slider("Incident count", 5, 100, 25)

runs = st.sidebar.number_input(
    "Number of simulation runs (searches for the longest streak)",
    min_value=1, max_value=1000, step=1, value=100, format="%d",
)

if st.sidebar.button("Simulate"):
    rng = np.random.default_rng()
    best_result = None
    for _ in range(int(runs)):
        outcome = run_simulation(nurse1_fill, other_fill, n_incidents, rng=rng)
        if best_result is None or outcome[5] > best_result[5]:
            best_result = outcome
    roster, incident_vec, counts, winner_idx, winner_fill, c_max = best_result
    fig = plot_compressed_roster(
        roster, incident_vec, winner_idx, c_max, winner_fill, n_incidents,
        title_suffix=f"max incidents in {runs} run{'s' if runs!=1 else ''}"
    )
    st.pyplot(fig, use_container_width=True)
else:
    st.info("👈 Set your parameters and click *Simulate* to begin.")
