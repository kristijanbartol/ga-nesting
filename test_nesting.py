from nesting.loader import PatchLoader
from nesting.engine import NestingEngine

# 1. Load Real Patches
loader = PatchLoader("results/pattern/latest")
items = loader.load_items()

# 2. Run Engine
print("\n[Nesting] Starting Layout Engine...")
engine = NestingEngine(fabric_width=1500.0, texture_spec=None) # 1500mm = 1.5m
final_state = engine.nest(items)

# 3. Report
print(f"\n[Result] Layout Complete.")
print(f"   Items Placed: {len(final_state.placed_items)}")
print(f"   Fabric Length Used: {final_state.total_height:.2f} mm")
print(f"   Efficiency: {(sum(i.area for i in items) / (1500 * final_state.total_height)):.2%}")

# 4. Visualize Result
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 20))
# Draw Fabric
ax.add_patch(plt.Rectangle((0,0), 10, final_state.total_height, fill=None, edgecolor='black'))

for item, x, y, poly in final_state.placed_items:
    # Extract coords
    px, py = poly.exterior.xy
    ax.fill(px, py, alpha=0.5, label=item.name)
    ax.text(x, y, item.name, fontsize=8)

ax.set_aspect('equal')
plt.show()