# test_phase_utils.py
import numpy as np
from nesting.phase_utils import TextureLattice, seam_phase_mismatch, seam_phase_mismatch_scalar

# Shared test lattice: axis-aligned, period=1 in both directions.
LATTICE = TextureLattice(
    u_dir=np.array([1.0, 0.0]),
    v_dir=np.array([0.0, 1.0]),
    period_u=1.0,
    period_v=1.0,
)
K = 8


def test_basic_mismatch():
    """
    V2 is shifted by half a period in X relative to V1.
    Both pairs have delta_U = 0.5, delta_V = 0.0.
    Per-point mean = (0.5 + 0.0) / 2 = 0.25.
    Overall mean = 0.25.  weight=1 -> 0.25.
    """
    V1 = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float)
    V2 = V1 + np.array([0.5, 0.0])
    pairs = [(0, 0), (1, 1)]

    m = seam_phase_mismatch(pairs, V1, V2, LATTICE, 0, 0, K, weight=1.0)
    assert abs(m - 0.25) < 1e-8, f"Expected 0.25, got {m}"


def test_weight_zero():
    """weight=0 must return exactly 0.0 regardless of geometry."""
    V1 = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float)
    V2 = V1 + np.array([0.5, 0.0])
    pairs = [(0, 0), (1, 1)]

    m = seam_phase_mismatch(pairs, V1, V2, LATTICE, 0, 0, K, weight=0.0)
    assert m == 0.0, f"Expected 0.0, got {m}"


def test_weight_scales_linearly():
    """Mismatch should scale exactly with weight."""
    V1 = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float)
    V2 = V1 + np.array([0.5, 0.0])
    pairs = [(0, 0), (1, 1)]

    m1 = seam_phase_mismatch(pairs, V1, V2, LATTICE, 0, 0, K, weight=1.0)
    m2 = seam_phase_mismatch(pairs, V1, V2, LATTICE, 0, 0, K, weight=2.0)
    assert abs(m2 - 2 * m1) < 1e-8, f"Expected {2*m1}, got {m2}"


def test_perfect_alignment_is_zero():
    """Identical patches have zero mismatch at any kappa."""
    V1 = np.array([[0.0, 0.0], [1.0, 0.3], [0.7, 0.9]], dtype=float)
    pairs = [(0, 0), (1, 1), (2, 2)]

    m = seam_phase_mismatch(pairs, V1, V1, LATTICE, 0, 0, K, weight=1.0)
    assert abs(m) < 1e-10, f"Expected 0.0, got {m}"


def test_max_mismatch_is_half():
    """
    Maximum possible mismatch per axis is 0.5 (half-period shift).
    Two patches shifted by 0.5 in both axes -> delta_U=0.5, delta_V=0.5
    -> per-point mean = 0.5 -> seam mean = 0.5.
    """
    V1 = np.array([[0.0, 0.0]], dtype=float)
    V2 = V1 + np.array([0.5, 0.5])
    pairs = [(0, 0)]

    m = seam_phase_mismatch(pairs, V1, V2, LATTICE, 0, 0, K, weight=1.0)
    assert abs(m - 0.5) < 1e-8, f"Expected 0.5, got {m}"


def test_kappa_compensates_both_axes():
    """
    V2 shifted by 0.5 in both X and Y (half period in both axes).
    With kappa_j = K//2, the phase offset adds 0.5 to both phi_j axes,
    wrapping them back to align with phi_i -> mismatch drops to ~0.
    """
    V1 = np.array([[0.0, 0.0], [0.3, 0.3]], dtype=float)
    V2 = V1 + np.array([0.5, 0.5])
    pairs = [(0, 0), (1, 1)]

    m_before = seam_phase_mismatch(pairs, V1, V2, LATTICE, 0, 0, K, weight=1.0)
    m_after  = seam_phase_mismatch(pairs, V1, V2, LATTICE, 0, K // 2, K, weight=1.0)

    assert abs(m_before - 0.5) < 1e-8, f"Expected baseline 0.5, got {m_before}"
    assert m_after < 1e-8, f"Expected ~0 after kappa compensation, got {m_after}"


def test_kappa_introduces_mismatch_on_aligned_patches():
    """
    Perfectly aligned patches get non-zero mismatch when kappa is non-zero,
    because kappa shifts one patch's phase relative to the other.
    """
    V1 = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float)
    pairs = [(0, 0), (1, 1)]

    m_k0 = seam_phase_mismatch(pairs, V1, V1, LATTICE, 0, 0, K, weight=1.0)
    m_k1 = seam_phase_mismatch(pairs, V1, V1, LATTICE, 0, 1, K, weight=1.0)

    assert abs(m_k0) < 1e-10, f"Expected 0 with k=0, got {m_k0}"
    assert m_k1 > 0, f"Expected mismatch > 0 with k=1, got {m_k1}"


def test_alias_matches():
    """seam_phase_mismatch_scalar must be an alias giving identical results."""
    V1 = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float)
    V2 = V1 + np.array([0.25, 0.0])
    pairs = [(0, 0), (1, 1)]

    m1 = seam_phase_mismatch(pairs, V1, V2, LATTICE, 0, 0, K, weight=1.0)
    m2 = seam_phase_mismatch_scalar(pairs, V1, V2, LATTICE, 0, 0, K, weight=1.0)
    assert abs(m1 - m2) < 1e-12, f"Alias mismatch: {m1} vs {m2}"


def main():
    tests = [
        test_basic_mismatch,
        test_weight_zero,
        test_weight_scales_linearly,
        test_perfect_alignment_is_zero,
        test_max_mismatch_is_half,
        test_kappa_compensates_both_axes,
        test_kappa_introduces_mismatch_on_aligned_patches,
        test_alias_matches,
    ]
    for t in tests:
        t()
        print(f"  PASS  {t.__name__}")
    print("\ntest_phase_utils.py: ALL OK")


if __name__ == "__main__":
    main()