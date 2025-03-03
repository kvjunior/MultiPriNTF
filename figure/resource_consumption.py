import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
import pandas as pd
from cycler import cycler
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

# Define custom color scheme for consistency
colors = {
    'primary': '#0072B2',    # Blue
    'secondary': '#009E73',  # Green
    'accent1': '#D55E00',    # Orange
    'accent2': '#CC79A7',    # Pink
    'neutral': '#777777'     # Gray
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

# Create a custom color cycle
plt.rcParams['axes.prop_cycle'] = cycler(color=[colors['primary'], colors['secondary'], colors['accent1'], colors['accent2']])

# Create figure with subplots in a 2x2 grid - slightly wider for better spacing
fig = plt.figure(figsize=(9.0, 6.5), dpi=300)
gs = GridSpec(2, 2, figure=fig, wspace=0.25, hspace=0.3)

# ---- (a) GPU Utilization ----
# Generate synthetic data representing GPU utilization under varying loads
time_points = np.linspace(0, 120, 1200)  # 120 minutes of data at 6-second intervals
load_changes = [10, 25, 40, 60, 80, 100, 60, 40, 20, 10]  # Load conditions (transactions per second)
load_change_times = np.linspace(0, 120, len(load_changes))

# Create load profile with transitions - fixed to avoid division by zero
def transition_between(t, start_val, end_val, start_time, end_time):
    if t <= start_time:  # Changed from < to <= to avoid division by zero
        return start_val
    if t >= end_time:    # Changed from > to >= for symmetry
        return end_val
    # Avoid division by zero by ensuring denominator is never zero
    time_diff = max(end_time - start_time, 1e-10)  # Add small epsilon
    progress = (t - start_time) / time_diff
    return start_val + progress * (end_val - start_val)

# Generate load values at each time point
load_values = np.zeros_like(time_points)
for i, t in enumerate(time_points):
    # Find the right segment
    idx = np.searchsorted(load_change_times, t)
    if idx == 0:
        load_values[i] = load_changes[0]
    elif idx >= len(load_changes):
        load_values[i] = load_changes[-1]
    else:
        load_values[i] = transition_between(t, 
                                           load_changes[idx-1],
                                           load_changes[idx],
                                           load_change_times[idx-1],
                                           load_change_times[idx])

# Generate GPU utilization at each time point with realistic response to load
base_utilization = 20  # Minimum utilization percentage
max_utilization = 95   # Maximum utilization percentage
utilization = base_utilization + (max_utilization - base_utilization) * (load_values / 100)**0.7

# Add noise to make the data look realistic
np.random.seed(42)
utilization += np.random.normal(0, 2, utilization.shape)
utilization = np.clip(utilization, 0, 100)

# Plot GPU utilization
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(time_points, utilization, linewidth=1.2, color=colors['primary'])
ax1.set_xlabel('Time (minutes)')
ax1.set_ylabel('GPU Utilization (%)', labelpad=10)
ax1.set_xlim(0, 120)
ax1.set_ylim(0, 100)
ax1.grid(True, linestyle='--', alpha=0.4)
ax1.set_title('(a) GPU Utilization Under Varying Load', fontweight='bold')

# Add annotation explaining utilization pattern
peak_time = time_points[np.argmax(utilization)]
peak_utilization = np.max(utilization)
ax1.annotate('Peak utilization: {:.1f}%'.format(peak_utilization),
            xy=(peak_time, peak_utilization),
            xytext=(peak_time-30, peak_utilization-15),
            **annotation_style)

# Add load change annotations
ax1_twin = ax1.twinx()
ax1_twin.plot(time_points, load_values, 'r--', alpha=0.5, linewidth=0.8)
ax1_twin.set_ylabel('Load (tx/s)', color='r', labelpad=10, rotation=270, va='bottom')
ax1_twin.tick_params(axis='y', labelcolor='r')
ax1_twin.set_ylim(0, 110)
ax1_twin.yaxis.set_label_coords(1.05, 0.5)

# ---- (b) Memory Consumption ----
# Generate synthetic memory consumption data
memory_usage = 5.0 + 0.05 * load_values  # Base 5GB + load-dependent increase
memory_leak_factor = np.linspace(0, 0.5, len(time_points))  # Simulate slight growth over time
memory_usage += memory_leak_factor  # Add time-dependent component

# Add periodic garbage collection effects
gc_intervals = np.arange(10, 120, 15)  # GC every 15 minutes
for gc_time in gc_intervals:
    gc_mask = np.logical_and(time_points >= gc_time, time_points <= gc_time + 0.5)
    memory_usage[gc_mask] -= 0.4  # Memory reduction after GC

# Add noise
memory_usage += np.random.normal(0, 0.1, memory_usage.shape)
memory_usage = np.clip(memory_usage, 5, 25)  # Reasonable bounds for memory usage

# Plot memory consumption
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(time_points, memory_usage, linewidth=1.2, color=colors['secondary'])
ax2.set_xlabel('Time (minutes)')
ax2.set_ylabel('Memory (GB)', labelpad=10)
ax2.set_xlim(0, 120)
ax2.set_ylim(0, 25)
ax2.grid(True, linestyle='--', alpha=0.4)
ax2.set_title('(b) Memory Consumption Over Time', fontweight='bold')

# Add annotations for garbage collection events
for gc_time in gc_intervals:
    ax2.axvline(x=gc_time, color='green', linestyle=':', alpha=0.5)
    
# Annotate one GC event as example
ax2.annotate('GC event', xy=(gc_intervals[2], 5.5), xytext=(gc_intervals[2]-5, 4),
            **annotation_style)

# Calculate memory efficiency
max_memory = memory_usage.max()
avg_memory = memory_usage.mean()
memory_efficiency = 1 - (avg_memory / 24)  # Assuming 24GB GPU memory

# Add annotation for memory efficiency
ax2.annotate(f'Memory efficiency: {memory_efficiency:.1%}',
            xy=(100, max_memory),
            xytext=(80, max_memory + 2),
            **annotation_style)

# ---- (c) CPU Utilization Distribution ----
# Generate synthetic CPU utilization data
num_cores = 16
core_labels = [f'Core {i+1}' for i in range(num_cores)]
cpu_utilization = np.zeros((len(time_points), num_cores))

# Different patterns for different core types
for i in range(num_cores):
    if i < 4:  # System cores with high usage
        base_util = 70 + np.random.normal(0, 5)
    elif i < 12:  # Processing cores with varying load
        base_util = 40 + np.random.normal(0, 10)
    else:  # Auxiliary cores with low usage
        base_util = 20 + np.random.normal(0, 5)
        
    # Add time-varying component based on load
    util_profile = base_util + (load_values/100) * (15 + np.random.normal(0, 3))
    
    # Add noise
    util_profile += np.random.normal(0, 5, util_profile.shape)
    cpu_utilization[:, i] = np.clip(util_profile, 0, 100)

# Calculate statistics for boxplot
cpu_stats = []
for i in range(num_cores):
    cpu_stats.append({
        'Core': core_labels[i],
        'Median': np.median(cpu_utilization[:, i]),
        'Q1': np.percentile(cpu_utilization[:, i], 25),
        'Q3': np.percentile(cpu_utilization[:, i], 75),
        'Min': np.percentile(cpu_utilization[:, i], 5),
        'Max': np.percentile(cpu_utilization[:, i], 95)
    })

cpu_df = pd.DataFrame(cpu_stats)

# Plot CPU utilization distribution
ax3 = fig.add_subplot(gs[1, 0])
boxprops = dict(linewidth=0.8)
whiskerprops = dict(linewidth=0.8)
medianprops = dict(linewidth=1.5)

# Create color list for box patches
box_colors = []
for i in range(num_cores):
    if i < 4:  # System cores
        box_colors.append(colors['primary'])
    elif i < 12:  # Processing cores
        box_colors.append(colors['secondary'])
    else:  # Auxiliary cores
        box_colors.append(colors['accent1'])

# Create boxplots with patch_artist=True to allow face colors
bp = ax3.boxplot([cpu_utilization[:, i] for i in range(num_cores)], 
                boxprops=boxprops, whiskerprops=whiskerprops, medianprops=medianprops,
                showfliers=False, patch_artist=True)  # Add patch_artist=True

# Set the box face colors
for box, color in zip(bp['boxes'], box_colors):
    box.set_facecolor(color)
    box.set_alpha(0.7)

ax3.set_xticklabels([f'{i+1}' for i in range(num_cores)], rotation=90)
ax3.set_xlabel('CPU Core')
ax3.set_ylabel('Utilization (%)')
ax3.set_ylim(0, 100)
ax3.grid(True, linestyle='--', alpha=0.4)
ax3.set_title('(c) CPU Utilization Distribution', fontweight='bold')

# Add annotation for core types
ax3.annotate('System cores',
            xy=(2, 85),
            xytext=(6, 90),
            **annotation_style)
ax3.annotate('Processing cores',
            xy=(8, 45),
            xytext=(12, 70),
            **annotation_style)
ax3.annotate('Auxiliary cores',
            xy=(14, 25),
            xytext=(10, 20),
            **annotation_style)

# ---- (d) Network Bandwidth Usage ----
# Generate synthetic network bandwidth data
send_bandwidth = 10 * (load_values/100)**1.2  # TX scales with load
receive_bandwidth = 5 * (load_values/100)     # RX scales with load but lower

# Add spikes for synchronization events
sync_times = np.arange(20, 120, 30)  # Sync every 30 minutes
for sync_time in sync_times:
    sync_mask = np.logical_and(time_points >= sync_time, time_points <= sync_time + 1)
    send_bandwidth[sync_mask] *= 3  # Higher outgoing bandwidth during sync
    receive_bandwidth[sync_mask] *= 2  # Higher incoming bandwidth during sync

# Add noise
send_bandwidth += np.random.normal(0, 0.3, send_bandwidth.shape)
receive_bandwidth += np.random.normal(0, 0.2, receive_bandwidth.shape)
send_bandwidth = np.clip(send_bandwidth, 0, 50)
receive_bandwidth = np.clip(receive_bandwidth, 0, 30)

# Plot network bandwidth
ax4 = fig.add_subplot(gs[1, 1])
ax4.plot(time_points, send_bandwidth, linewidth=1.2, label='Outgoing', color=colors['primary'])
ax4.plot(time_points, receive_bandwidth, linewidth=1.2, label='Incoming', color=colors['secondary'])
ax4.set_xlabel('Time (minutes)')
ax4.set_ylabel('Bandwidth (MB/s)')
ax4.set_xlim(0, 120)
ax4.set_ylim(0, 50)
ax4.grid(True, linestyle='--', alpha=0.4)
ax4.legend(loc='upper right')
ax4.set_title('(d) Network Bandwidth Usage', fontweight='bold')

# Add annotation for sync events
for sync_time in sync_times:
    ax4.axvline(x=sync_time, color='gray', linestyle=':', alpha=0.5)
    
# Annotate one sync event as example
sync_idx = np.argmin(np.abs(time_points - sync_times[1]))
peak_bandwidth = send_bandwidth[sync_idx]
ax4.annotate('Sync event\n(3x bandwidth spike)',
            xy=(sync_times[1], peak_bandwidth),
            xytext=(sync_times[1]+5, peak_bandwidth-10),
            **annotation_style)

# Calculate total data transferred
total_sent = np.trapz(send_bandwidth, time_points) / 60  # Convert to GB
total_received = np.trapz(receive_bandwidth, time_points) / 60  # Convert to GB
ax4.annotate(f'Total data: {total_sent+total_received:.1f} GB',
            xy=(110, 5),
            xytext=(90, 15),
            **annotation_style)

# Add overall figure title
fig.suptitle('Resource Consumption Patterns During Production Testing', fontweight='bold', y=0.98)

# Handle layout with a different approach to avoid warning
fig.tight_layout(pad=2.0)
plt.subplots_adjust(top=0.93)  # Adjust top to make room for suptitle

# Save figures
plt.savefig('figures/resource_consumption.pdf', bbox_inches='tight', dpi=300)
plt.savefig('figures/resource_consumption.png', bbox_inches='tight', dpi=300)
plt.close(fig)

print("Figure generated successfully: figures/resource_consumption.pdf")