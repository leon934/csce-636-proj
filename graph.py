import os
import re
import glob
import matplotlib.pyplot as plt

# 1. Base directory containing the logs
base_dir = "logs-laptop"

# 2. Regex pattern to catch the final test loss
# Example target: "2026-03-16 13:27:44,570 - INFO - Final Test Loss for ResColumnCNN: 2.429110"
test_loss_pattern = re.compile(r"Final Test Loss for (.*?): ([\d.]+)")

# 3. The 9 parameter triplets specified in the PDF project description
triplets = [
    (9, 4, 2), (9, 4, 3), (9, 4, 4), (9, 4, 5),
    (9, 5, 2), (9, 5, 3), (9, 5, 4),
    (9, 6, 2), (9, 6, 3)
]

# 4. Create a 3x3 figure for the subplots
fig, axs = plt.subplots(3, 3, figsize=(18, 15))
axs = axs.flatten()  # Flatten the 2D array of axes for easy iteration

for i, (n, k, m) in enumerate(triplets):
    folder_name = f"n{n}_k{k}_m{m}"
    folder_path = os.path.join(base_dir, folder_name)
    
    ax = axs[i]  # Select the appropriate subplot
    
    # Check if the directory exists
    if not os.path.exists(folder_path):
        ax.set_title(f"n={n}, k={k}, m={m}\n(Directory not found)")
        ax.axis('off')
        continue

    # Find all .log files in the specific directory
    log_files = glob.glob(os.path.join(folder_path, "*.log"))
    
    run_names = []
    final_losses = []
    
    for log_file in log_files:
        file_identifier = os.path.basename(log_file).replace(".log", "")
        
        with open(log_file, 'r') as f:
            content = f.read()
            match = test_loss_pattern.search(content)
            
            if match:
                loss_val = float(match.group(2))
                
                # Split the long filename to fit better on a smaller subplot's x-axis
                # Example: "ResColumnCNN_20260316_110537" -> "ResColumnCNN\n20260316_110537"
                name_parts = file_identifier.split('_', 1)
                if len(name_parts) > 1:
                    short_name = f"{name_parts[0]}\n{name_parts[1]}"
                else:
                    short_name = file_identifier
                
                run_names.append(short_name)
                final_losses.append(loss_val)
    
    # Generate the bar plot on this specific subplot axis
    if final_losses:
        bars = ax.bar(run_names, final_losses, color='coral', edgecolor='black')
        
        # Add the exact values on top of the bars
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f"{yval:.4f}", 
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

        ax.set_title(f"Final Test Loss (n={n}, k={k}, m={m})")
        ax.set_ylabel("Test Loss")
        
        # Format the x-axis tick labels to prevent overlap
        ax.tick_params(axis='x', rotation=45, labelsize=9)
        for tick in ax.get_xticklabels():
            tick.set_ha('right')
    else:
        ax.set_title(f"n={n}, k={k}, m={m}\n(No valid log data found)")
        ax.axis('off')

# Adjust layout to prevent overlapping titles and labels
plt.tight_layout()

# Save the final consolidated plot
output_filename = "final_test_loss_3x3_grid.png"
plt.savefig(output_filename, dpi=300)
print(f"Saved 3x3 plot: {output_filename}")

plt.close()