import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.patches as patches

# Data
holding_periods = [100, 125, 128, 129, 130, 135, 140, 150, 200]
melo_proportions = [1.0, 1.0, 1.0, 0.0439, 0.0, 0.0, 0.0, 0.0, 0.0]

# Create the main plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(holding_periods, melo_proportions, marker='o', linestyle='-', color='b', label='MELO Proportion')
ax.set_title('Proportion of MOBI Traders Using MELO Market In Equilibrium over Varying Holding Periods')
ax.set_xlabel('Holding Period (Timesteps)')
ax.set_ylabel('Proportion of MOBI Traders Using MELO Market')
ax.axvline(x=129, color='r', linestyle='--', label='Critical Holding Period (129)')
ax.legend()
ax.grid(True)

# # Create the inset
# axins = inset_axes(ax, width="30%", height="30%", loc='upper right', 
#                   borderpad=2)

# # Plot the same data on the inset
# axins.plot(holding_periods, melo_proportions, marker='o', linestyle='-', color='b')
# axins.set_xlim(124, 136)
# axins.set_ylim(-0.1, 1.1)
# axins.grid(True)
# axins.axvline(x=129, color='r', linestyle='--')

# # Draw a rectangle in the main axes to highlight the zoomed region
# zoom_x_min, zoom_x_max = 124, 136
# zoom_y_min, zoom_y_max = -0.1, 1.1

# rect = patches.Rectangle((zoom_x_min, zoom_y_min), 
#                         zoom_x_max - zoom_x_min, 
#                         zoom_y_max - zoom_y_min,
#                         linewidth=1, edgecolor='blue', facecolor='none',
#                         linestyle=':')
# ax.add_patch(rect)

# # Get the coordinates of the inset axes in the figure coordinate system
# inset_pos = axins.get_position()
# inset_bottom_left = (inset_pos.x0, inset_pos.y0)

# # Calculate the center of the zoomed region
# zoom_center_x = (zoom_x_min + zoom_x_max) / 2
# zoom_center_y = (zoom_y_min + zoom_y_max) / 2
# zoom_center_display = ax.transData.transform((zoom_center_x, zoom_center_y))
# zoom_center_fig = fig.transFigure.inverted().transform(zoom_center_display)

# # Calculate the bottom-left of the inset in figure coordinates
# inset_center_x = inset_pos.x0 + 0.02  # Slightly offset from the left edge
# inset_center_y = inset_pos.y0 + 0.02  # Slightly offset from the bottom edge

# # Draw a single connection line
# fig.add_artist(patches.ConnectionPatch(
#     xyA=zoom_center_fig, coordsA='figure fraction',
#     xyB=(inset_center_x, inset_center_y), coordsB='figure fraction',
#     color='gray', linewidth=1.5, linestyle='-', alpha=0.7))

# Save and show the plot
plt.savefig('./data_with_single_connection.jpg')
plt.tight_layout()
plt.show()