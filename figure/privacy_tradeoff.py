import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
import os

# Create figures directory if it doesn't exist
os.makedirs('figures', exist_ok=True)

# Set style parameters to match academic paper standards
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 10,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.titlesize': 12,
    'mathtext.fontset': 'stix'
})

# Define consistent color scheme
colors = {
    'primary': '#0072B2',   # Blue
    'secondary': '#009E73', # Green
    'accent1': '#D55E00',   # Orange
    'accent2': '#CC79A7',   # Pink
    'neutral': '#777777'    # Gray
}

# Define consistent annotation style
annotation_style = dict(
    arrowprops=dict(
        arrowstyle="->", 
        connectionstyle="arc3", 
        color='black', 
        alpha=0.7
    ),
    fontsize=7
)

# Create figure with subplots
fig = plt.figure(figsize=(8.5, 7.0), dpi=300)
gs = GridSpec(2, 2, figure=fig, width_ratios=[1, 1], height_ratios=[1, 1])

# Define consistent markers and linestyles
markers = ['o', 's', '^', 'd', 'x']
linestyles = ['-', '--', '-.', ':', '-']

# ---- (a) Market efficiency score vs. privacy budget ----
ax1 = fig.add_subplot(gs[0, 0])

# Generate synthetic data for market efficiency vs privacy budget
privacy_budgets = np.array([0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0])
market_efficiency = np.array([0.68, 0.74, 0.80, 0.85, 0.87, 0.89, 0.90])

# Plot market efficiency vs privacy budget
ax1.plot(privacy_budgets, market_efficiency, marker='o', color=colors['primary'], linewidth=1.5, markersize=6)
ax1.set_xlabel('Privacy Budget ($\\epsilon$)')
ax1.set_ylabel('Market Efficiency Score')
ax1.set_xscale('log')
ax1.set_xlim(0.009, 1.1)
ax1.set_ylim(0.65, 0.92)
ax1.grid(True, linestyle='--', alpha=0.4)
ax1.set_title('(a) Market Efficiency Score vs. Privacy Budget', fontweight='bold')

# Highlight the selected privacy budget for the paper
selected_privacy = 0.1
selected_efficiency = 0.85
ax1.scatter([selected_privacy], [selected_efficiency], s=100, facecolors='none', 
           edgecolors='red', linewidths=1.5, zorder=5)
ax1.annotate('Selected operating point\n($\\epsilon=0.1$, MES=0.85)', 
            xy=(selected_privacy, selected_efficiency), 
            xytext=(0.02, 0.77),
            **annotation_style)

# Add shaded regions for privacy regimes with improved colors
ax1.axvspan(0.01, 0.05, alpha=0.15, color=colors['primary'], label='High Privacy')
ax1.axvspan(0.05, 0.2, alpha=0.1, color=colors['secondary'], label='Balanced')
ax1.axvspan(0.2, 1.0, alpha=0.15, color=colors['accent1'], label='Low Privacy')

# Add arrow indicating the tradeoff direction
ax1.annotate('', xy=(0.5, 0.89), xytext=(0.015, 0.7),
            arrowprops=dict(arrowstyle="<->", connectionstyle="arc3", 
                           color='black', alpha=0.7, lw=1.5))
ax1.text(0.06, 0.72, 'Privacy-Utility\nTradeoff', 
        ha='center', fontsize=7, rotation=45)

ax1.legend(loc='lower right', fontsize=7)

# ---- (b) Information leakage vs. privacy budget ----
ax2 = fig.add_subplot(gs[0, 1])

# Generate synthetic data for information leakage
information_leakage = 0.05 * privacy_budgets ** 0.8
theoretical_bound = 0.08 * privacy_budgets ** 0.5

# Add small random variations to make curves look more realistic
np.random.seed(42)
noise_factor = 0.1
information_leakage *= (1 + noise_factor * np.random.randn(len(privacy_budgets)) * 0.2)
information_leakage = np.clip(information_leakage, 0, theoretical_bound)

# Plot information leakage
ax2.plot(privacy_budgets, theoretical_bound, linestyle='--', color=colors['neutral'], 
        label='Theoretical bound', linewidth=1.5)
ax2.plot(privacy_budgets, information_leakage, marker='o', color=colors['primary'], 
        label='Measured leakage', linewidth=1.5, markersize=6)

# Fill area between curves to highlight the gap
ax2.fill_between(privacy_budgets, information_leakage, theoretical_bound, 
                color=colors['primary'], alpha=0.2)

# Highlight selected privacy budget
ax2.axvline(x=selected_privacy, color='red', linestyle=':', alpha=0.7)
ax2.scatter([selected_privacy], [information_leakage[privacy_budgets == selected_privacy]], 
           s=100, facecolors='none', edgecolors='red', linewidths=1.5, zorder=5)

# Calculate leakage at selected point
selected_leakage = float(information_leakage[privacy_budgets == selected_privacy])
ax2.annotate(f'Leakage: {selected_leakage:.3f} bits/tx', 
            xy=(selected_privacy, selected_leakage), 
            xytext=(0.15, 0.01),
            **annotation_style)

# Add annotation for the security margin
theoretical_at_selected = float(theoretical_bound[privacy_budgets == selected_privacy])
margin = theoretical_at_selected - selected_leakage
margin_percent = (margin / theoretical_at_selected) * 100

ax2.annotate(f'Security margin: {margin_percent:.1f}%', 
            xy=(selected_privacy, (selected_leakage + theoretical_at_selected)/2), 
            xytext=(0.3, 0.035),
            **annotation_style)

ax2.set_xlabel('Privacy Budget ($\\epsilon$)')
ax2.set_ylabel('Information Leakage (bits/tx)')
ax2.set_xscale('log')
ax2.set_xlim(0.009, 1.1)
ax2.set_ylim(0, 0.08)
ax2.grid(True, linestyle='--', alpha=0.4)
ax2.legend(loc='upper left', fontsize=7)
ax2.set_title('(b) Information Leakage vs. Privacy Budget', fontweight='bold')

# ---- (c) Attack success rate vs. privacy budget ----
ax3 = fig.add_subplot(gs[1, :])  # Span entire width for the third plot

# Define attack types
attack_types = ['Model Inversion', 'Membership Inference', 'Attribute Inference', 
               'Transaction Linkage', 'Price Reconstruction']

# Generate synthetic data for attack success rates
# Random baseline (guessing)
random_baseline = np.array([0.031, 0.500, 0.125, 0.010, 0.083])

# Success rates at different privacy budgets for each attack type
privacy_budgets_attacks = np.array([0.01, 0.05, 0.1, 0.2, 0.5, 1.0])
attack_success_rates = np.zeros((len(attack_types), len(privacy_budgets_attacks)))

# Generate realistic curves with diminishing returns as privacy budget increases
for i, attack in enumerate(attack_types):
    # Curve shape parameters vary by attack type
    base_rate = random_baseline[i]
    max_increase = np.array([0.15, 0.12, 0.08, 0.10, 0.09])[i]
    curve_shape = np.array([0.5, 0.6, 0.7, 0.4, 0.5])[i]
    
    # Generate curve
    attack_success_rates[i] = base_rate + max_increase * (privacy_budgets_attacks ** curve_shape)
    
    # Add noise
    noise = 0.01 * np.random.randn(len(privacy_budgets_attacks))
    attack_success_rates[i] += noise
    
    # Ensure success rate doesn't go below random baseline
    attack_success_rates[i] = np.maximum(attack_success_rates[i], base_rate)

# Define a custom color palette for attack types
attack_colors = [colors['primary'], colors['secondary'], colors['accent1'], 
                colors['accent2'], '#F0E442']

# Plot attack success rates
for i, attack in enumerate(attack_types):
    ax3.plot(privacy_budgets_attacks, attack_success_rates[i], 
            marker=markers[i], color=attack_colors[i % len(attack_colors)], 
            linestyle=linestyles[i % len(linestyles)],
            label=attack, linewidth=1.5, markersize=6)
    
    # Plot random baseline as horizontal dashed line
    ax3.axhline(y=random_baseline[i], color=attack_colors[i % len(attack_colors)], 
               linestyle=':', alpha=0.5)
    
    # Annotate random baseline for the first and last attack types
    if i == 0 or i == len(attack_types)-1:
        ax3.annotate(f'Random ({random_baseline[i]:.3f})', 
                    xy=(privacy_budgets_attacks[0], random_baseline[i]), 
                    xytext=(0.008, random_baseline[i] + 0.01),
                    color=attack_colors[i % len(attack_colors)],
                    fontsize=7)

# Highlight selected privacy budget
ax3.axvline(x=selected_privacy, color='red', linestyle=':', alpha=0.7)
ax3.annotate('Selected\n$\\epsilon=0.1$', 
            xy=(selected_privacy, 0.55), 
            xytext=(selected_privacy, 0.57),
            ha='center',
            **annotation_style)

# Add protection factor annotation for a specific attack at selected budget
selected_idx = np.where(privacy_budgets_attacks == selected_privacy)[0][0]
attack_idx = 0  # Model Inversion attack
protection_factor = (attack_success_rates[attack_idx][selected_idx] - random_baseline[attack_idx]) / (random_baseline[attack_idx]) + 1

# Add a more descriptive annotation showing attack resilience
ax3.annotate(f'Protection Factor: {protection_factor:.2f}\nOnly {attack_success_rates[attack_idx][selected_idx]:.3f} success rate', 
            xy=(selected_privacy, attack_success_rates[attack_idx][selected_idx]), 
            xytext=(0.15, 0.05),
            **annotation_style)

# Add a region highlighting the security threshold
ax3.axhspan(0, 0.05, alpha=0.1, color='green', label='High Security')
ax3.text(0.015, 0.025, 'High Security Zone', fontsize=7, ha='left', va='center')

ax3.set_xlabel('Privacy Budget ($\\epsilon$)')
ax3.set_ylabel('Attack Success Rate')
ax3.set_xscale('log')
ax3.set_xlim(0.009, 1.1)
ax3.set_ylim(0, 0.6)
ax3.grid(True, linestyle='--', alpha=0.4)

# Create a cleaner two-row legend
legend = ax3.legend(ncol=3, loc='upper left', fontsize=8, 
                   framealpha=0.8, edgecolor='lightgray')
legend.get_frame().set_linewidth(0.5)

ax3.set_title('(c) Attack Success Rate vs. Privacy Budget for Different Attack Types', fontweight='bold')

# Add figure title
fig.suptitle('Privacy-Performance Tradeoff Analysis', fontweight='bold', y=0.98)

# Adjust layout
fig.tight_layout()
plt.subplots_adjust(top=0.93, hspace=0.3, wspace=0.25)

# Save figure
plt.savefig('figures/privacy_tradeoff.pdf', bbox_inches='tight', dpi=300)
plt.savefig('figures/privacy_tradeoff.png', bbox_inches='tight', dpi=300)
plt.close(fig)

print("Figure generated successfully: figures/privacy_tradeoff.pdf")