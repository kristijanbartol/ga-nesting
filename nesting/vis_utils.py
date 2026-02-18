import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from spec import TextureSpec

def plot_layout(fabric_state, texture_spec: TextureSpec, title="Nesting Layout"):
    """
    Visualizes the final layout with texture constraints.
    """
    width = fabric_state.width
    height = max(fabric_state.total_height * 1.1, width * 0.5) # Add margin
    
    fig, ax = plt.subplots(figsize=(12, 12 * (height/width)))
    
    # 1. Draw Fabric Background (The "Roll")
    # We use a clipping path to confine texture to the fabric rect
    fabric_rect = patches.Rectangle((0, 0), width, height, linewidth=2, edgecolor='black', facecolor='none', zorder=10)
    ax.add_patch(fabric_rect)
    
    # 2. Draw Texture Pattern
    if texture_spec.name.lower() == "stripes":
        # Draw horizontal lines
        # Period Y is the vertical repeat distance
        period = texture_spec.period_y
        
        # Calculate how many lines fit
        y_lines = np.arange(0, height, period)
        
        for y in y_lines:
            # Draw line across full width
            ax.hlines(y, 0, width, colors='lightgray', linestyles='--', linewidth=1)
            
    elif texture_spec.name.lower() == "grid":
        # Draw vertical and horizontal lines
        px = texture_spec.period_x
        py = texture_spec.period_y
        
        x_lines = np.arange(0, width, px)
        y_lines = np.arange(0, height, py)
        
        for y in y_lines:
            ax.hlines(y, 0, width, colors='lightgray', linestyles='-', linewidth=0.5)
        for x in x_lines:
            ax.vlines(x, 0, height, colors='lightgray', linestyles='-', linewidth=0.5)

    # 3. Draw Placed Items
    for item, cx, cy, poly in fabric_state.placed_items:
        # Extract polygon coordinates
        x, y = poly.exterior.xy
        
        # Draw filled polygon
        ax.fill(x, y, alpha=0.7, label=item.name, edgecolor='black')
        
        # Draw Centroid / Anchor (Red Dot) to verify snapping
        ax.plot(cx, cy, 'ro', markersize=3)
        
        # Text Label
        ax.text(cx, cy, item.name, ha='center', va='center', fontsize=8, color='white', weight='bold')

    ax.set_xlim(-50, width + 50)
    ax.set_ylim(-50, height + 50)
    ax.set_aspect('equal')
    ax.set_title(f"{title} - Efficiency: {fabric_state.efficiency:.1%}")
    plt.show()