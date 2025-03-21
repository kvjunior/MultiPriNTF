import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_pdf import PdfPages

# Set style parameters to match academic paper template
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 10,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.titlesize': 12,
    'mathtext.fontset': 'stix'
})

# Define color scheme
colors = {
    'MultiPriNTF': '#0072B2',  # Blue
    'Baseline': '#D55E00'      # Orange
}

# Create figure with adjusted size and spacing
fig = plt.figure(figsize=(9.0, 6.5), dpi=300)
gs = GridSpec(2, 2, figure=fig, wspace=0.4, hspace=0.4)

# Baseline methods data
methods = ['ARIMA', 'GRU', 'UI', 'DCGAN', 'CoC-GAN', 'PrivacyNFT', 'MultiPriNTF']

# (a) Market Efficiency Score
ax1 = fig.add_subplot(gs[0, 0])
mes_scores = [0.61, 0.70, 0.72, 0.74, 0.76, 0.78, 0.85]
x = np.arange(len(methods))
ax1.bar(x[:-1], mes_scores[:-1], color=colors['Baseline'], alpha=0.7, label='Baseline')
ax1.bar(x[-1], mes_scores[-1], color=colors['MultiPriNTF'], alpha=0.9, label='MultiPriNTF')
ax1.set_xticks(x)
ax1.set_xticklabels(methods, rotation=45, ha='right', rotation_mode='anchor')
ax1.set_ylabel('Market Efficiency Score')
ax1.set_title('(a) Market Efficiency Score', fontweight='bold')
ax1.set_ylim(0.5, 0.9)
ax1.grid(axis='y', linestyle='--', alpha=0.7)

# (b) Mean Absolute Error
ax2 = fig.add_subplot(gs[0, 1])
mae_values = [43.62, 29.84, 24.31, 20.75, 18.92, 17.43, 12.56]
ax2.bar(x[:-1], mae_values[:-1], color=colors['Baseline'], alpha=0.7)
ax2.bar(x[-1], mae_values[-1], color=colors['MultiPriNTF'], alpha=0.9)
ax2.set_xticks(x)
ax2.set_xticklabels(methods, rotation=45, ha='right', rotation_mode='anchor')
ax2.set_ylabel('Mean Absolute Error')
ax2.set_title('(b) Mean Absolute Error', fontweight='bold')
ax2.set_ylim(0, 50)
ax2.grid(axis='y', linestyle='--', alpha=0.7)

# (c) Price Direction Accuracy
ax3 = fig.add_subplot(gs[1, 0])
pda_values = [71.2, 78.5, 82.6, 81.9, 83.1, 84.2, 88.4]
ax3.bar(x[:-1], pda_values[:-1], color=colors['Baseline'], alpha=0.7)
ax3.bar(x[-1], pda_values[-1], color=colors['MultiPriNTF'], alpha=0.9)
ax3.set_xticks(x)
ax3.set_xticklabels(methods, rotation=45, ha='right', rotation_mode='anchor')
ax3.set_ylabel('Price Direction Accuracy (%)')
ax3.set_title('(c) Price Direction Accuracy', fontweight='bold')
ax3.set_ylim(60, 95)
ax3.grid(axis='y', linestyle='--', alpha=0.7)

# (d) Throughput
ax4 = fig.add_subplot(gs[1, 1])
throughput_values = [2156, 653, 412, 378, 305, 247, 832]
ax4.bar(x[:-1], throughput_values[:-1], color=colors['Baseline'], alpha=0.7)
ax4.bar(x[-1], throughput_values[-1], color=colors['MultiPriNTF'], alpha=0.9)
ax4.set_xticks(x)
ax4.set_xticklabels(methods, rotation=45, ha='right', rotation_mode='anchor')
ax4.set_ylabel('Throughput (transactions/s)')
ax4.set_title('(d) Throughput', fontweight='bold')
ax4.set_ylim(0, 2500)
ax4.grid(axis='y', linestyle='--', alpha=0.7)

# Add overall figure title
plt.suptitle('Performance Improvements of MultiPriNTF', fontweight='bold', y=0.98)

# Adjust layout with more padding
plt.tight_layout(rect=[0, 0.03, 1, 0.95], pad=2.0)

# Add extra vertical space at the bottom
plt.subplots_adjust(bottom=0.15)

# Save as PDF
with PdfPages('figures/improvement_summary.pdf') as pdf:
    pdf.savefig(fig, bbox_inches='tight')

plt.close(fig)
print("Figure generated successfully: figures/improvement_summary.pdf")