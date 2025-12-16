import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np

# ==========================================
# 1. Data Preparation
# ==========================================
episodes = ['Ep1', 'Ep2', 'Ep3', 'Ep4', 'Ep6', 'Ep7', 'Ep8', 'Ep9', 'Ep10']

# --- Part A: Accuracy Trends ---
acc_except_narration = [0.7514124294, 0.7973333333, 0.7990970655, 0.8646616541, 0.8206896552, 0.7968337731, 0.7619047619, 0.8193832599, 0.8496042216]
overall_accuracy = [0.6759259259, 0.7984693878, 0.76875, 0.7867803838, 0.785106383, 0.75, 0.7249466951, 0.7795591182, 0.7897196262]

# Effective total subtitles (Total - Brief) derived for calculation
n_effective = [432, 392, 480, 469, 470, 444, 469, 499, 428]

# --- Part B: Narration Error Data [Total, Correct] ---
# Speaker invisible
inv_data = [[41, 17], [16, 14], [34, 15], [30, 11], [25, 7], [59, 29], [32, 11], [45, 17], [26, 7]]
# Background
bg_data = [[17, 9], [0, 0], [2, 0], [34, 13], [4, 0], [2, 0], [5, 4], [0, 0], [1, 1]]
# Other
oth_data = [[20, 0], [1, 0], [1, 0], [6, 0], [6, 5], [4, 2], [12, 5], [0, 0], [22, 8]]

# Function to calculate error rate within the category
def calculate_category_error_rate(data_list):
    rates = []
    for total, correct in data_list:
        if total == 0:
            rates.append(0)
        else:
            rates.append((total - correct) / total)
    return rates

inv_rates = calculate_category_error_rate(inv_data)
bg_rates = calculate_category_error_rate(bg_data)
oth_rates = calculate_category_error_rate(oth_data)

# --- Part C: Contribution to Overall Accuracy Drop ---
# Calculate number of errors for each category
inv_errors = [d[0] - d[1] for d in inv_data]
bg_errors = [d[0] - d[1] for d in bg_data]
oth_errors = [d[0] - d[1] for d in oth_data]

# Calculate impact (percentage points deducted from overall accuracy)
inv_impact = [e / n for e, n in zip(inv_errors, n_effective)]
bg_impact = [e / n for e, n in zip(bg_errors, n_effective)]
oth_impact = [e / n for e, n in zip(oth_errors, n_effective)]

# Calculate total error rate and non-narration error rate
total_error_rate = [1 - acc for acc in overall_accuracy]
non_narration_impact = []
for i in range(len(episodes)):
    narration_impact = inv_impact[i] + bg_impact[i] + oth_impact[i]
    non_narration_impact.append(total_error_rate[i] - narration_impact)

# ==========================================
# 2. Plotting
# ==========================================
# Create figure with 3 subplots (3 rows)
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 18))

# ------------------------------------------
# Plot 1: Accuracy Trends (Line Chart)
# ------------------------------------------
ax1.plot(episodes, acc_except_narration, marker='o', linestyle='-', color='#1f77b4', label='Accuracy (except narration)', linewidth=2.5)
ax1.plot(episodes, overall_accuracy, marker='s', linestyle='--', color='#ff7f0e', label='Overall Accuracy', linewidth=2.5)

for i, txt in enumerate(acc_except_narration):
    ax1.annotate(f"{txt:.1%}", (episodes[i], acc_except_narration[i]), xytext=(0, 10), textcoords="offset points", ha='center', color='#1f77b4', fontweight='bold')
for i, txt in enumerate(overall_accuracy):
    ax1.annotate(f"{txt:.1%}", (episodes[i], overall_accuracy[i]), xytext=(0, -20), textcoords="offset points", ha='center', color='#ff7f0e', fontweight='bold')

ax1.set_title('(A) Overall Accuracy Trends', fontsize=16, fontweight='bold')
ax1.set_ylabel('Accuracy', fontsize=12)
ax1.set_ylim(0.5, 1.0)
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.legend(loc='lower right', fontsize=12)
ax1.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

# ------------------------------------------
# Plot 2: Narration Error Rates Breakdown (Grouped Bar)
# ------------------------------------------
x = np.arange(len(episodes))
width = 0.25

rects1 = ax2.bar(x - width, inv_rates, width, label='Speaker invisible', color='#5DADE2')
rects2 = ax2.bar(x, bg_rates, width, label='Background', color='#F5B041')
rects3 = ax2.bar(x + width, oth_rates, width, label='Other(Unknown)', color='#82E0AA')

def autolabel(rects, data_sources, ax):
    for i, rect in enumerate(rects):
        height = rect.get_height()
        total = data_sources[i][0]
        label = "N/A" if total == 0 else f"{height:.1%}"
        ax.annotate(label, xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

autolabel(rects1, inv_data, ax2)
autolabel(rects2, bg_data, ax2)
autolabel(rects3, oth_data, ax2)

ax2.set_title('(B) Error Rate Within Narration Categories', fontsize=16, fontweight='bold')
ax2.set_ylabel('Error Rate (within category)', fontsize=12)
ax2.set_xticks(x)
ax2.set_xticklabels(episodes, fontsize=12)
ax2.set_ylim(0, 1.2)
ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=12)
ax2.grid(axis='y', linestyle='--', alpha=0.3)
ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

# ------------------------------------------
# Plot 3: Contribution to Overall Accuracy Drop (Stacked Bar)
# ------------------------------------------
bar_width = 0.6

# Stack: Non-Narration (Bottom, Gray) -> Inv (Blue) -> Bg (Orange) -> Oth (Green)
p1 = ax3.bar(episodes, non_narration_impact, bar_width, label='Non-Narration Errors', color='#D3D3D3')
p2 = ax3.bar(episodes, inv_impact, bar_width, bottom=non_narration_impact, label='Speaker invisible', color='#5DADE2')
p3 = ax3.bar(episodes, bg_impact, bar_width, bottom=[i+j for i,j in zip(non_narration_impact, inv_impact)], label='Background', color='#F5B041')
p4 = ax3.bar(episodes, oth_impact, bar_width, bottom=[i+j+k for i,j,k in zip(non_narration_impact, inv_impact, bg_impact)], label='Other(Unknown)', color='#82E0AA')

# Label the total height (Overall Error Rate)
for i, v in enumerate(total_error_rate):
    ax3.text(i, v + 0.015, f"Total Err\n{v:.1%}", ha='center', fontweight='bold', fontsize=10)

# Optional: Label segments > 2% impact
for i, rect in enumerate(p2):
    height = rect.get_height()
    if height > 0.02:
        ax3.text(rect.get_x() + rect.get_width()/2, rect.get_y() + height/2, f"{height:.1%}", ha='center', va='center', fontsize=8, color='white')

ax3.set_title('(C) Contribution to Overall Accuracy Drop (Error Composition)', fontsize=16, fontweight='bold')
ax3.set_ylabel('Impact on Overall Accuracy', fontsize=12)
ax3.set_ylim(0, 0.6) 
ax3.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4, fontsize=12)
ax3.grid(axis='y', linestyle='--', alpha=0.3)
ax3.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

# ==========================================
# 3. Output
# ==========================================
plt.tight_layout()
plt.savefig('subtitle_analysis_result_v2.png', dpi=300)
plt.show()