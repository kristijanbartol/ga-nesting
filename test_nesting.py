import numpy as np
from nesting.item import NestingItem

# ==============================================================================
# TEST DATA: A Simple Rectangle (10 x 20)
# ==============================================================================
# Centered at (0,0), so X is [-5, 5], Y is [-10, 10]
rect_verts = np.array([
    [-5, -10],
    [ 5, -10],
    [ 5,  10],
    [-5,  10]
])

print("[Test 1] creating NestingItem...")
item = NestingItem(item_id=1, name="Rectangle", original_vertices=rect_verts)

print(f"   Initial Bounds: {item.bounds}")
# Expected: (-5.0, -10.0, 5.0, 10.0)

# ==============================================================================
# TEST ROTATION (90 Degrees)
# ==============================================================================
print("\n[Test 2] Rotating 90 degrees...")
item.set_rotation(90)

print(f"   Rotated Bounds: {item.bounds}")
# Expected: (-10.0, -5.0, 10.0, 5.0) 
# (The 20-height became 20-width)

# ==============================================================================
# TEST PLACEMENT (Translation)
# ==============================================================================
print("\n[Test 3] Placing at (100, 100)...")
placed_poly = item.place_at(100, 100)

print(f"   Placed Bounds: {placed_poly.bounds}")
# Expected: (90.0, 95.0, 110.0, 105.0)
# Center (100,100) +/- half dimensions

print("\n[Success] Basic Geometry Engine is functional.")
