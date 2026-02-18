import matplotlib.pyplot as plt
import matplotlib.patches as patches

def visualize_layout(fabric_state, texture_spec):
    fig, ax = plt.subplots(figsize=(12, 12))
    
    max_h = max(fabric_state.total_height + 50, 200)
    width = fabric_state.width
    
    # 1. Draw Texture Lattice
    if texture_spec.name == "Stripes" or texture_spec.name == "Grid":
        # Vertical lines (if Grid)
        if texture_spec.name == "Grid":
            for x in range(0, int(width) + 1, int(texture_spec.period_x)):
                ax.axvline(x, color='gray', linestyle=':', alpha=0.3, lw=0.5)
        
        # Horizontal lines (Always for Stripes/Grid)
        for y in range(0, int(max_h) + 1, int(texture_spec.period_y)):
            ax.axhline(y, color='gray', linestyle=':', alpha=0.3, lw=0.5)

    # 2. Draw Fabric Boundary
    rect = patches.Rectangle((0, 0), width, fabric_state.total_height, 
                             linewidth=2, edgecolor='black', facecolor='none', zorder=10)
    ax.add_patch(rect)

    # 3. Draw Placed Items
    for item, cx, cy, poly in fabric_state.placed_items:
        x, y = poly.exterior.xy
        ax.fill(x, y, alpha=0.6, label=item.name, edgecolor='darkblue')
        
        # Draw the Anchor Point (Centroid) - Should land on a grid intersection
        ax.scatter([cx], [cy], color='red', s=20, zorder=15)
        ax.text(cx, cy, f" {item.name}", fontsize=7, verticalalignment='bottom')

    ax.set_xlim(-10, width + 10)
    ax.set_ylim(-10, max_h)
    ax.set_aspect('equal')
    ax.set_title(f"Nesting Result - Texture: {texture_spec.name}")
    plt.legend(loc='upper right', fontsize='small', ncol=2)
    plt.show()