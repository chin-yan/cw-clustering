import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# Set font to support proper label display
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']

# Episode data
episodes = [1, 2, 3, 4, 6, 7, 8, 9, 10]
clear_identified_speaker = [333, 391, 404, 384, 448, 382, 427, 466, 385]
correct_annotation = [201, 222, 242, 178, 240, 169, 286, 296, 270]
accuracy = [0.6036036036, 0.5677749361, 0.599009901, 0.4635416667, 0.5357142857, 
            0.442408377, 0.6697892272, 0.635193133, 0.7012987013]

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# Plot 1: Accuracy over episodes
ax1.plot(episodes, accuracy, marker='o', linewidth=2, markersize=8, color='#2E86AB')
ax1.set_xlabel('Episode', fontsize=12, fontweight='bold')
ax1.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
ax1.set_title('Speaker Annotation Accuracy by Episode (Season 2)', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.set_xticks(episodes)
ax1.set_ylim([0.4, 0.75])

# Calculate and plot average accuracy line
avg_accuracy = np.mean(accuracy)
ax1.axhline(y=avg_accuracy, color='#D62828', linestyle='--', linewidth=2, label=f'Average Accuracy: {avg_accuracy:.4f}')
ax1.legend(fontsize=11, loc='lower right')

# Add value labels on points
for ep, acc in zip(episodes, accuracy):
    ax1.text(ep, acc + 0.01, f'{acc:.3f}', ha='center', va='bottom', fontsize=9)

# Plot 2: Clear identified speaker vs Correct annotation
x = np.arange(len(episodes))
width = 0.35

bars1 = ax2.bar(x - width/2, clear_identified_speaker, width, label='Clear Identified Speaker', color='#A23B72')
bars2 = ax2.bar(x + width/2, correct_annotation, width, label='Correct Annotation', color='#F18F01')

ax2.set_xlabel('Episode', fontsize=12, fontweight='bold')
ax2.set_ylabel('Count', fontsize=12, fontweight='bold')
ax2.set_title('Clear Identified Speaker vs Correct Annotation Count', fontsize=14, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels([f'ep{ep}' for ep in episodes])
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar in bars1:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{int(height)}', ha='center', va='bottom', fontsize=8)

for bar in bars2:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{int(height)}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig('annotation_results.png', dpi=300, bbox_inches='tight')
plt.show()