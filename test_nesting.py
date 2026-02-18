from nesting.loader import PatchLoader
from nesting.engine import NestingEngine
from spec import TextureSpec
from nesting.vis_utils import visualize_layout

# 1. Define Texture (e.g. 50mm x 50mm grid)
# This is what the GA will provide
texture = TextureSpec(name="Grid", period_x=50.0, period_y=50.0)
FABRIC_WIDTH = 1500.0 # 1.5 meters

# 2. Load Geometry
loader = PatchLoader("results/pattern/latest")
items = loader.load_items()

# 3. Run Engine
engine = NestingEngine(fabric_width=FABRIC_WIDTH, texture_spec=texture)
final_state = engine.nest(items)

# 4. Report & Visualize
print(f"Nesting Complete. Efficiency: {(sum(i.area for i in items)/(FABRIC_WIDTH*final_state.total_height)):.2%}")
visualize_layout(final_state, texture)