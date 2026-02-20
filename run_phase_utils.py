# test_phase_utils.py
import numpy as np
from nesting.phase_utils import TextureLattice, seam_phase_mismatch

def main():
    # Two trivial "patches" each with 3 vertices
    V1 = np.array([[0.0, 0.0],
                   [1.0, 0.0],
                   [0.0, 1.0]], dtype=float)

    # Patch 2 is shifted by half-period in x -> causes phase mismatch for U direction
    V2 = V1 + np.array([0.5, 0.0])

    # seam pairs compare vertex 0->0 and 1->1
    seam_pairs = [(0, 0), (1, 1)]

    # lattice: U=(1,0)*1.0, V=(0,1)*1.0
    lattice = TextureLattice(
        u_dir=np.array([1.0, 0.0]),
        v_dir=np.array([0.0, 1.0]),
        period_u=1.0,
        period_v=1.0
    )

    # no phase bins
    K = 8
    k1 = 0
    k2 = 0

    # weight 1 -> mismatch should be ~0.25 (mean of du=0.5 and dv=0.0 => 0.25)
    m = seam_phase_mismatch(seam_pairs, V1, V2, lattice, k1, k2, K, weight=1.0)
    assert abs(m - 0.25) < 1e-8, f"Expected 0.25, got {m}"

    # weight 0 disables
    m0 = seam_phase_mismatch(seam_pairs, V1, V2, lattice, k1, k2, K, weight=0.0)
    assert m0 == 0.0

    # If we compensate by kappa (phase bin), mismatch should go down
    # half period => kappa = K/2 (when K even)
    m_comp = seam_phase_mismatch(seam_pairs, V1, V2, lattice, 0, K//2, K, weight=1.0)
    assert m_comp - 1e-7 < m, f"Expected compensation to reduce mismatch, got {m_comp} vs {m}"

    print("test_phase_utils.py: OK")

if __name__ == "__main__":
    main()
