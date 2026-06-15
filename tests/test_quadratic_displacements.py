"""
Tests for quadratic shape function displacement correctness.

Strategy
--------
A truss bar under pure axial load has a *linear* displacement field.
Quadratic shape functions can represent linear functions exactly, so:

  - End-node displacements must match the analytical result  u = F·L/(E·A).
  - The back-substituted mid-node displacement must equal the exact midpoint
    value (u_n1 + u_n2) / 2  for uniform strain.
  - get_displacement_on_element(xi) must reproduce the quadratic interpolant
    at arbitrary ξ, which for a linear field also equals the linear interpolant.
  - Changing from "linear" to "quadratic" must NOT change end-node results.

Node / DOF layout (single horizontal bar, quadratic)
-----------------------------------------------------
After __add_mid_node the nodes list becomes:
    node 0: (0, 0)   → DOF 0 (ux), DOF 1 (uy)
    node 1: (1, 0)   → DOF 2 (ux), DOF 3 (uy)
    node 2: (0.5, 0) → DOF 4 (ux), DOF 5 (uy)  ← appended mid-node
Element = [n1=0, mid=2, n2=1]

The global stiffness is built on the *original* 2 nodes only (static
condensation). The full displacement vector has 6 entries (3 nodes × 2 DOFs).
"""

import numpy as np
import pytest
import sys, os

# ---------------------------------------------------------------------------
# Make the package importable when running directly from this directory.
# Adjust the path below to match your project layout if needed.
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from CSM.plane_truss import PlaneTrussProblem


# ===========================================================================
# Shared fixtures
# ===========================================================================

def _bar(shape_func="quadratic", E=1.0, A=1.0):
    """Single horizontal bar (0,0)→(1,0), E=A=L=1."""
    return PlaneTrussProblem(
        nodes=[(0.0, 0.0), (1.0, 0.0)],
        elements=[(0, 1)],
        elasticity_modulus=E,
        cross_sectional_area=A,
        shape_func=shape_func,
    )


def _solved_bar(F=100.0, shape_func="quadratic", E=1.0, A=1.0):
    """
    Single horizontal bar, pinned at node 0 + roller-y at node 1.
    Axial load F applied at node 1 (DOF 2).
    Analytical: u_x_node1 = F·L/(E·A)
    """
    t = _bar(shape_func=shape_func, E=E, A=A)
    t.assemble_global_stiffness()
    t.solve({2: F}, constrained_dofs=[0, 1, 3])
    return t


def _diagonal_bar(shape_func="quadratic"):
    """45° bar (0,0)→(1/√2, 1/√2), L=1, E=A=1."""
    r = 1.0 / np.sqrt(2)
    return PlaneTrussProblem(
        nodes=[(0.0, 0.0), (r, r)],
        elements=[(0, 1)],
        elasticity_modulus=1.0,
        cross_sectional_area=1.0,
        shape_func=shape_func,
    )


def _solved_diagonal_bar(F=100.0, shape_func="quadratic"):
    """
    45° bar under axial load F applied in x-direction at node 1 (DOF 2).
    Analytical: u_x_node1 = F·L/(E·A·cos²θ) = 2·F  (θ=45°, c=1/√2).
    """
    t = _diagonal_bar(shape_func=shape_func)
    t.assemble_global_stiffness()
    t.solve({2: F}, constrained_dofs=[0, 1, 3])
    return t


# ===========================================================================
# 1. Node count and mid-node geometry
# ===========================================================================

class TestQuadraticNodeSetup:
    """Ensure the mid-node is inserted correctly before any FE computation."""

    def test_midnode_inserted(self):
        """Quadratic bar must have 3 nodes (original 2 + 1 mid)."""
        t = _bar()
        assert t.num_nodes == 3

    def test_midnode_position_horizontal(self):
        """Mid-node of a horizontal bar must lie exactly at x=0.5, y=0."""
        t = _bar()
        # Mid-node is appended as node index 2
        np.testing.assert_allclose(t.nodes[2], [0.5, 0.0], atol=1e-14)

    def test_midnode_position_diagonal_45(self):
        """Mid-node of a 45° bar must lie at (r/2, r/2)."""
        t = _diagonal_bar()
        r = 1.0 / np.sqrt(2)
        np.testing.assert_allclose(t.nodes[2], [r / 2, r / 2], atol=1e-14)

    def test_element_references_midnode(self):
        """Element tuple must contain the mid-node index."""
        t = _bar()
        assert len(t.elements[0]) == 3
        assert t.elements[0][1] == 2  # mid-node is the second entry


# ===========================================================================
# 2. Global stiffness after static condensation
# ===========================================================================

class TestQuadraticCondensedStiffness:
    """
    After static condensation the condensed 4×4 stiffness of a quadratic
    element must be identical to the linear element stiffness (same physics,
    same EA/L, same orientation).
    """

    @pytest.mark.parametrize("nodes", [
        [(0, 0), (1, 0)],        # horizontal
        [(0, 0), (0, 1)],        # vertical
        [(0, 0), (1, 1)],        # 45°
        [(0, 0), (-1, 2)],       # obtuse angle
    ])
    def test_condensed_equals_linear(self, nodes):
        t_lin  = PlaneTrussProblem(nodes, [(0, 1)], 210e9, 0.01, shape_func="linear")
        t_quad = PlaneTrussProblem(nodes, [(0, 1)], 210e9, 0.01, shape_func="quadratic")
        k_lin  = t_lin.ElementStiffness(0)
        k_quad = t_quad.ElementStiffness(0)
        np.testing.assert_allclose(k_quad, k_lin, rtol=1e-10,
            err_msg=f"Condensed quadratic stiffness differs from linear for nodes={nodes}")

    def test_global_stiffness_sized_on_original_nodes_only(self):
        """k_global must be 2*n_original × 2*n_original, not include mid-nodes."""
        t = _bar()
        t.assemble_global_stiffness()
        # 2 original nodes → 4×4 global matrix
        assert t.k_global.shape == (4, 4)

    def test_global_stiffness_is_symmetric(self):
        t = _bar()
        K = t.assemble_global_stiffness()
        np.testing.assert_allclose(K, K.T, atol=1e-12)


# ===========================================================================
# 3. End-node displacements after solve
# ===========================================================================

class TestQuadraticEndNodeDisplacements:
    """
    The condensed system only involves the original (end) nodes.
    Quadratic and linear must give identical end-node displacements.
    """

    @pytest.mark.parametrize("F", [1.0, 50.0, 100.0, 1e6])
    def test_axial_displacement_matches_analytical(self, F):
        """u_x at loaded end = F·L/(E·A) = F  for E=A=L=1."""
        t = _solved_bar(F=F)
        assert t.displacements[2] == pytest.approx(F, rel=1e-10)

    def test_constrained_dofs_are_zero(self):
        """DOFs 0, 1 (pin) and 3 (roller-y) must be zero after solve."""
        t = _solved_bar(F=77.0)
        for dof in (0, 1, 3):
            assert t.displacements[dof] == pytest.approx(0.0, abs=1e-12)

    def test_quadratic_equals_linear_end_nodes(self):
        """Quadratic end-node result must match linear for all DOFs of original nodes."""
        F = 42.0
        t_lin  = _solved_bar(F=F, shape_func="linear")
        t_quad = _solved_bar(F=F, shape_func="quadratic")
        # Compare only the first 4 DOFs (original nodes)
        np.testing.assert_allclose(
            t_quad.displacements[:4], t_lin.displacements[:4], atol=1e-10,
            err_msg="Quadratic end-node displacements differ from linear"
        )

    @pytest.mark.parametrize("E, A, F", [
        (210e9, 0.01, 1000.0),
        (70e9,  0.05, -500.0),
        (1.0,   1.0,  100.0),
    ])
    def test_axial_displacement_various_material(self, E, A, F):
        """u = F·L/(E·A) must hold for any E, A with L=1."""
        t = _solved_bar(F=F, E=E, A=A)
        expected = F * 1.0 / (E * A)   # L=1
        assert t.displacements[2] == pytest.approx(expected, rel=1e-8)

    def test_diagonal_bar_end_node_displacement(self):
        """45° bar: u_x_node1 = F / (E·A·c²) = 2·F for c=1/√2, E=A=1."""
        F = 100.0
        t = _solved_diagonal_bar(F=F)
        expected_ux = 2 * F   # analytical
        assert t.displacements[2] == pytest.approx(expected_ux, rel=1e-8)


# ===========================================================================
# 4. Mid-node displacement (back-substitution)
# ===========================================================================

class TestQuadraticMidNodeDisplacement:
    """
    For a bar under uniform axial strain the displacement field is linear,
    so the back-substituted mid-node displacement must be exactly the average
    of the two end-node displacements.
    """

    def test_midnode_dof_index(self):
        """Mid-node is stored as node 2 → DOFs 4 and 5."""
        t = _solved_bar(F=100.0)
        mid_ux = t.displacements[4]
        mid_uy = t.displacements[5]
        assert mid_ux == pytest.approx(50.0, rel=1e-10)
        assert mid_uy == pytest.approx(0.0,  abs=1e-12)

    @pytest.mark.parametrize("F", [10.0, 75.0, 1000.0])
    def test_midnode_is_average_of_end_nodes(self, F):
        """u_mid = (u_n1 + u_n2) / 2 for uniform axial loading."""
        t = _solved_bar(F=F)
        u_n1 = t.displacements[0]  # ux of node 0
        u_n2 = t.displacements[2]  # ux of node 1
        u_mid = t.displacements[4] # ux of node 2 (mid)
        assert u_mid == pytest.approx((u_n1 + u_n2) / 2.0, rel=1e-10)

    def test_midnode_uy_is_zero_for_horizontal_bar(self):
        """A horizontal bar under axial load has no transverse displacement."""
        t = _solved_bar(F=200.0)
        assert t.displacements[5] == pytest.approx(0.0, abs=1e-12)

    def test_midnode_diagonal_bar_averages(self):
        """45° bar: mid-node x and y components must each be the average of endpoints."""
        F = 100.0
        t = _solved_diagonal_bar(F=F)
        # End nodes: node0 = (0,0), node1 = (2F·c, 2F·s) at 45° → (200*c, 200*c)
        # with c = s = 1/√2 ≈ 0.707
        # But DOF 3 (uy node1) was constrained → uy_node1 = 0
        # Mid-node must average the axial projection that back-substitution gives.
        # The back-sub works in local axial direction then projects to global.
        # Here we just assert the mid-node ux averages node0 and node1 ux:
        u_mid_x = t.displacements[4]
        u_n0_x  = t.displacements[0]
        u_n1_x  = t.displacements[2]
        print(u_n0_x ,u_n1_x )
        assert u_mid_x == pytest.approx((u_n0_x + u_n1_x) / 2.0, rel=1e-8)

    def test_get_displacement_returns_midnode_correctly(self):
        """get_displacement(mid_node_index) must return back-substituted values."""
        t = _solved_bar(F=100.0)
        ux, uy = t.get_displacement(2)   # node 2 = mid-node
        assert ux == pytest.approx(50.0, rel=1e-10)
        assert uy == pytest.approx(0.0,  abs=1e-12)


# ===========================================================================
# 5. Displacement interpolation along the element  (get_displacement_on_element)
# ===========================================================================

class TestQuadraticDisplacementOnElement:
    """
    get_displacement_on_element(element_index, xi) uses the 3-node
    shape functions: N = [0.5ξ(ξ-1), (1-ξ²), 0.5ξ(ξ+1)].

    For uniform strain the field is linear, so the quadratic interpolant
    must reproduce the same values as linear interpolation would.
    """

    def test_xi_minus1_returns_start_node(self):
        """ξ = -1 must return node 0 displacement = (0, 0)."""
        t = _solved_bar(F=100.0)
        d = t.get_displacement_on_element(0, -1.0)
        np.testing.assert_allclose(d, [0.0, 0.0], atol=1e-12)

    def test_xi_plus1_returns_end_node(self):
        """ξ = +1 must return node 1 displacement = (100, 0)."""
        t = _solved_bar(F=100.0)
        d = t.get_displacement_on_element(0, 1.0)
        np.testing.assert_allclose(d, [100.0, 0.0], atol=1e-12)

    def test_xi_0_returns_midnode_displacement(self):
        """ξ = 0 maps to the physical mid-point; must match the back-substituted mid-node."""
        t = _solved_bar(F=100.0)
        d = t.get_displacement_on_element(0, 0.0)
        np.testing.assert_allclose(d, [50.0, 0.0], atol=1e-12)

    @pytest.mark.parametrize("xi, expected_ux", [
        (-0.5,  25.0),
        ( 0.0,  50.0),
        ( 0.5,  75.0),
    ])
    def test_interior_points_match_linear_field(self, xi, expected_ux):
        """
        For constant-strain bar the quadratic interpolant is exact.
        u(ξ) = 50·(1 + ξ) because u_n1=0, u_n2=100 (linear in ξ).
        """
        t = _solved_bar(F=100.0)
        d = t.get_displacement_on_element(0, xi)
        np.testing.assert_allclose(d[0], expected_ux, atol=1e-10,
            err_msg=f"u_x at ξ={xi} should be {expected_ux}")
        np.testing.assert_allclose(d[1], 0.0, atol=1e-12)

    def test_quadratic_on_element_equals_linear_on_element(self):
        """
        For a bar under pure axial load, linear and quadratic interpolation
        must give the same displacement at any ξ interior point.
        """
        F = 60.0
        t_lin  = _solved_bar(F=F, shape_func="linear")
        t_quad = _solved_bar(F=F, shape_func="quadratic")
        for xi in np.linspace(-1, 1, 11):
            d_lin  = t_lin.get_displacement_on_element(0, xi)
            d_quad = t_quad.get_displacement_on_element(0, xi)
            np.testing.assert_allclose(d_quad, d_lin, atol=1e-10,
                err_msg=f"Mismatch at ξ={xi:.2f}: linear={d_lin}, quad={d_quad}")


# ===========================================================================
# 6. Stress and force correctness for quadratic elements
# ===========================================================================

class TestQuadraticStressAndForce:
    """Stress must be uniform along a bar under constant axial load."""

    @pytest.mark.parametrize("xi", [-1.0, -0.5, 0.0, 0.5, 1.0])
    def test_stress_is_uniform_along_element(self, xi):
        """σ = F/A everywhere; for constant strain the quadratic B is exact."""
        F, A = 80.0, 1.0
        t = _solved_bar(F=F, A=A)
        stress = t.get_element_stress(0, xi)
        assert stress == pytest.approx(F / A, rel=1e-8)

    def test_force_equals_applied_load(self):
        """Axial force in element = external load (statically determinate)."""
        F = 55.0
        t = _solved_bar(F=F)
        assert t.get_element_force(0) == pytest.approx(F, rel=1e-8)

    def test_stress_quadratic_equals_linear(self):
        """Quadratic and linear must give identical stress for uniform strain."""
        F = 33.0
        t_lin  = _solved_bar(F=F, shape_func="linear")
        t_quad = _solved_bar(F=F, shape_func="quadratic")
        assert t_quad.get_element_stress(0) == pytest.approx(t_lin.get_element_stress(0), rel=1e-10)


# ===========================================================================
# 7. Reaction forces
# ===========================================================================

class TestQuadraticReactionForces:
    """Static equilibrium must hold: sum of reactions in each direction = 0."""

    def test_x_equilibrium(self):
        t = _solved_bar(F=100.0)
        rxns = t.get_reaction_forces()
        assert rxns[0::2].sum() == pytest.approx(0.0, abs=1e-10)

    def test_y_equilibrium(self):
        t = _solved_bar(F=100.0)
        rxns = t.get_reaction_forces()
        assert rxns[1::2].sum() == pytest.approx(0.0, abs=1e-10)

    def test_pin_reaction_balances_load(self):
        """The pin at node 0 must carry –F in the x direction."""
        F = 123.0
        t = _solved_bar(F=F)
        rxns = t.get_reaction_forces()
        assert rxns[0] == pytest.approx(-F, rel=1e-10)


# ===========================================================================
# 8. Multi-element quadratic truss (two bars in series)
# ===========================================================================

class TestQuadraticTwoBarSeries:
    """
    Two collinear unit bars [0]──[1]──[2], E=A=1, force at DOF 2 (ux of node 1).
    Both ends pinned (nodes 0 and 2 fully fixed).
    The structure is symmetric: the two springs EA/L = 1 in series meet at node 1.
    Total stiffness at node 1: k = EA/L + EA/L = 2  →  u_node1 = F / 2.
    """

    def _make(self):
        t = PlaneTrussProblem(
            nodes=[(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)],
            elements=[(0, 1), (1, 2)],
            elasticity_modulus=1.0,
            cross_sectional_area=1.0,
            shape_func="quadratic",
        )
        t.assemble_global_stiffness()
        return t

    def test_midnode_count(self):
        """Two quadratic bars → 2 extra mid-nodes → 5 nodes total."""
        t = self._make()
        assert t.num_nodes == 5

    @pytest.mark.parametrize("F", [50.0, 100.0, 200.0])
    def test_midpoint_node_displacement(self, F):
        """u_x at node 1 (mid of the structure) = F / 2 for symmetric pinned-pinned."""
        t = self._make()
        t.solve({2: F}, constrained_dofs=[0, 1, 3, 4, 5])
        assert t.displacements[2] == pytest.approx(F / 2.0, rel=1e-10)

    def test_constrained_nodes_zero(self):
        t = self._make()
        t.solve({2: 50.0}, constrained_dofs=[0, 1, 3, 4, 5])
        for dof in (0, 1, 4, 5):   # node 0 and node 2 fully fixed
            assert t.displacements[dof] == pytest.approx(0.0, abs=1e-12)

    def test_quadratic_equals_linear_for_series(self):
        """Both discretisations must give the same midpoint displacement."""
        F = 70.0
        t_lin = PlaneTrussProblem(
            nodes=[(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)],
            elements=[(0, 1), (1, 2)],
            elasticity_modulus=1.0, cross_sectional_area=1.0, shape_func="linear",
        )
        t_lin.assemble_global_stiffness()
        t_lin.solve({2: F}, constrained_dofs=[0, 1, 3, 4, 5])

        t_quad = PlaneTrussProblem(
            nodes=[(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)],
            elements=[(0, 1), (1, 2)],
            elasticity_modulus=1.0, cross_sectional_area=1.0, shape_func="quadratic",
        )
        t_quad.assemble_global_stiffness()
        t_quad.solve({2: F}, constrained_dofs=[0, 1, 3, 4, 5])

        np.testing.assert_allclose(
            t_quad.displacements[:6],   # first 6 = original-node DOFs
            t_lin.displacements[:6],
            atol=1e-10,
        )


# ===========================================================================
# 9. Multi-geometry parametric checks (global equilibrium & zero constraints)
# ===========================================================================

@pytest.mark.parametrize("shape_func", ["linear", "quadratic"])
@pytest.mark.parametrize("name, nodes, elements, constrained, forces", [
    (
        "right_triangle",
        [(0, 0), (1, 0), (0, 1)],
        [(0, 1), (1, 2), (0, 2)],
        [0, 1, 3],
        {4: 100.0, 5: -50.0},
    ),
    (
        "square_cross_braced",
        [(0, 0), (1, 0), (1, 1), (0, 1)],
        [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)],
        [0, 1, 6, 7],
        {3: -200.0, 5: -200.0},
    ),
])
def test_global_equilibrium(shape_func, name, nodes, elements, constrained, forces):
    """Sum of reaction forces must vanish in x and y (Newton's 1st law)."""
    t = PlaneTrussProblem(nodes, elements, 210e9, 0.01, shape_func=shape_func)
    t.assemble_global_stiffness()
    t.solve(forces, constrained_dofs=constrained)
    rxns = t.get_reaction_forces()

    for dof in constrained:
        assert t.displacements[dof] == pytest.approx(0.0, abs=1e-12), \
            f"[{name}/{shape_func}] constrained DOF {dof} is not zero"

    assert rxns[0::2].sum() == pytest.approx(0.0, abs=1e-6), \
        f"[{name}/{shape_func}] x-equilibrium failed"
    assert rxns[1::2].sum() == pytest.approx(0.0, abs=1e-6), \
        f"[{name}/{shape_func}] y-equilibrium failed"
