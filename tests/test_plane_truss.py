"""Unit tests for PlaneTrussProblem (plane_truss.py)."""

import numpy as np
import pytest
from CSM.plane_truss import PlaneTrussProblem


# ─────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────

shape_funcs = ["linear","quadratic"]

def make_single_bar(shape_func="linear"):
    """Single horizontal bar (0,0)→(1,0), E=A=L=1."""
    return PlaneTrussProblem(
        nodes=[(0, 0), (1, 0)],
        elements=[(0, 1)],
        elasticity_modulus=1.0,
        cross_sectional_area=1.0,
        shape_func=shape_func,
    )


def solved_single_bar(u=2,F=100.0, shape_func="linear"):
    """Assembled + solved single horizontal bar with axial load F at node 1.

    DOF layout (linear, 2 nodes):
        0 = ux_node0,  1 = uy_node0,  2 = ux_node1,  3 = uy_node1
    Constraints: 0, 1 (pin node 0) + 3 (roller y at node 1)
    Free DOF:    2  →  u_x_node1 = F * L / (E * A) = F
    """
    truss = make_single_bar(shape_func)
    truss.assemble_global_stiffness()
    # FIX: Pass dictionary for DOF 2 (Node 1, X-axis)
    truss.solve({2: F}, constrained_dofs=[0, 1, 3])
    return truss

def make_single_bar_diagonal(shape_func="linear"):
    """Single horizontal bar (0,0)→(1,0), E=A=L=1."""
    return PlaneTrussProblem(
        nodes=[(0, 0), (1/(np.sqrt(2)), 1/(np.sqrt(2)))],
        elements=[(0, 1)],
        elasticity_modulus=1.0,
        cross_sectional_area=1.0,
        shape_func=shape_func,
    )

def solved_single_bar_diagonal(u=2,F=100.0, shape_func="linear"):
    """Assembled + solved single horizontal bar with axial load F at node 1.

    DOF layout (linear, 2 nodes):
        0 = ux_node0,  1 = uy_node0,  2 = ux_node1,  3 = uy_node1
    Constraints: 0, 1 (pin node 0) + 3 (roller y at node 1)
    Free DOF:    2  →  u_x_node1 = F * L / (E * A) = F
    """
    truss = make_single_bar_diagonal(shape_func)
    truss.assemble_global_stiffness()
    # FIX: Pass dictionary for DOF 2 (Node 1, X-axis)
    truss.solve({2: F}, constrained_dofs=[0, 1, 3])
    return truss


def make_right_triangle(shape_func="linear"):
    """Right triangle truss: nodes at (0,0), (1,0), (0,1), E=A=1.0."""
    return PlaneTrussProblem(
        nodes=[(0, 0), (1, 0), (0, 1)],
        elements=[(0, 1), (1, 2), (0, 2)],
        elasticity_modulus=1.0,
        cross_sectional_area=1.0,
        shape_func=shape_func,
    )

def solved_right_triangle():
    """
    Assembled + solved right triangle truss.
    Constraints: 0, 1 (pin node 0) + 3 (roller y at node 1).
    Loads: Fx = 100.0 at node 2 (DOF 4), Fy = -50.0 at node 2 (DOF 5).
    """
    truss = make_right_triangle()
    truss.assemble_global_stiffness()
    truss.solve({4: 100.0, 5: -50.0}, constrained_dofs=[0, 1, 3])
    return truss

# ─────────────────────────────────────────────────────────────
# Initialization
# ─────────────────────────────────────────────────────────────

class TestInitialization:

    def test_linear_node_and_element_counts(self):
        t = make_single_bar()
        assert t.num_nodes == 2
        assert t.num_elements == 1

    def test_quadratic_inserts_midnode(self):
        t = make_single_bar("quadratic")
        assert t.num_nodes == 3
        assert len(t.elements[0]) == 3

    def test_quadratic_midnode_position(self):
        """Mid-node must be exactly halfway between the two original nodes."""
        t = make_single_bar("quadratic")
        np.testing.assert_allclose(t.nodes[2], [0.5, 0.0])

    def test_global_stiffness_initialized_to_zero(self):
        assert np.all(make_single_bar().k_global == 0.0)

    def test_global_stiffness_correct_shape(self):
        t = PlaneTrussProblem([(0, 0), (1, 0), (2, 0)], [(0, 1), (1, 2)], 1.0, 1.0)
        assert t.k_global.shape == (6, 6)

    def test_right_triangle_node_and_element_counts(self):
        t = make_right_triangle()
        assert t.num_nodes == 3
        assert t.num_elements == 3


    # — Validation errors ——————————————————————

    def test_too_few_nodes_raises(self):
        with pytest.raises(ValueError, match="At least two nodes"):
            PlaneTrussProblem([(0, 0)], [(0, 0)], 1.0, 1.0)

    def test_too_few_elements_raises(self):
        with pytest.raises(ValueError, match="At least one element"):
            PlaneTrussProblem([(0, 0), (1, 0)], [], 1.0, 1.0)

    def test_non_2d_nodes_raises(self):
        with pytest.raises(ValueError, match="2D"):
            PlaneTrussProblem([(0, 0, 0), (1, 0, 0)], [(0, 1)], 1.0, 1.0)

    def test_invalid_shape_function_raises(self):
        with pytest.raises(ValueError, match="not a valid shape function"):
            PlaneTrussProblem([(0, 0), (1, 0)], [(0, 1)], 1.0, 1.0, "hermite")


# ─────────────────────────────────────────────────────────────
# get_angle
# ─────────────────────────────────────────────────────────────

class TestGetAngle:

    def test_horizontal_right_is_0(self):
        t = PlaneTrussProblem([(0, 0), (1, 0)], [(0, 1)], 1.0, 1.0)
        assert t.get_angle(0) == pytest.approx(0.0)

    def test_vertical_up_is_90(self):
        t = PlaneTrussProblem([(0, 0), (0, 1)], [(0, 1)], 1.0, 1.0)
        assert t.get_angle(0) == pytest.approx(90.0)

    def test_diagonal_is_45(self):
        t = PlaneTrussProblem([(0, 0), (1, 1)], [(0, 1)], 1.0, 1.0)
        assert t.get_angle(0) == pytest.approx(45.0)

    def test_horizontal_left_is_180(self):
        t = PlaneTrussProblem([(0, 0), (-1, 0)], [(0, 1)], 1.0, 1.0)
        assert t.get_angle(0) == pytest.approx(180.0)

    def test_downward_right_wrapped_to_315(self):
        # arctan2(-1, 1) = -45°  →  normalised to 315°
        t = PlaneTrussProblem([(0, 0), (1, -1)], [(0, 1)], 1.0, 1.0)
        assert t.get_angle(0) == pytest.approx(315.0)

    def test_negative_index_raises(self):
        with pytest.raises(ValueError, match="Element index out of range"):
            make_single_bar().get_angle(-1)

    def test_out_of_bounds_index_raises(self):
        with pytest.raises(ValueError, match="Element index out of range"):
            make_single_bar().get_angle(1)

    def test_right_triangle_angles(self):
        t = make_right_triangle()
        assert t.get_angle(0) == pytest.approx(0.0)
        # Element 1: (1,0) to (0,1) => dx=-1, dy=1 => 135 degrees
        assert t.get_angle(1) == pytest.approx(135.0)
        # Element 2: (0,0) to (0,1) => dx=0, dy=1 => 90 degrees
        assert t.get_angle(2) == pytest.approx(90.0)


# ─────────────────────────────────────────────────────────────
# get_length
# ─────────────────────────────────────────────────────────────

class TestGetLength:

    def test_unit_horizontal(self):
        assert make_single_bar().get_length(0) == pytest.approx(1.0)

    def test_unit_diagonal(self):
        assert make_single_bar_diagonal().get_length(0) == pytest.approx(1.0)

    def test_diagonal_is_sqrt2(self):
        t = PlaneTrussProblem([(0, 0), (1, 1)], [(0, 1)], 1.0, 1.0)
        assert t.get_length(0) == pytest.approx(np.sqrt(2))

    def test_vertical_length_3(self):
        t = PlaneTrussProblem([(0, 0), (0, 3)], [(0, 1)], 1.0, 1.0)
        assert t.get_length(0) == pytest.approx(3.0)

    def test_negative_index_raises(self):
        with pytest.raises(ValueError, match="Element index out of range"):
            make_single_bar().get_length(-1)

    def test_out_of_bounds_index_raises(self):
        with pytest.raises(ValueError, match="Element index out of range"):
            make_single_bar().get_length(5)

    def test_right_triangle_lengths(self):
        t = make_right_triangle()
        assert t.get_length(0) == pytest.approx(1.0)
        assert t.get_length(1) == pytest.approx(np.sqrt(2))
        assert t.get_length(2) == pytest.approx(1.0)

# ─────────────────────────────────────────────────────────────
# ElementStiffness
# ─────────────────────────────────────────────────────────────

class TestElementStiffness:

    def test_linear_element_shape_is_4x4(self):
        assert make_single_bar().ElementStiffness(0).shape == (4, 4)

    def test_quadratic_element_shape_is_6x6(self):
        assert make_single_bar("quadratic").ElementStiffness(0).shape == (6, 6)

    def test_linear_element_is_symmetric(self):
        k = make_single_bar().ElementStiffness(0)
        np.testing.assert_allclose(k, k.T, atol=1e-12)

    def test_quadratic_element_is_symmetric(self):
        k = make_single_bar("quadratic").ElementStiffness(0)
        np.testing.assert_allclose(k, k.T, atol=1e-12)

    def test_horizontal_unit_bar_analytical(self):
        """E=A=L=1, θ=0° → K = (EA/L)·[[1,0,-1,0],[0,0,0,0],[-1,0,1,0],[0,0,0,0]]."""
        k = make_single_bar().ElementStiffness(0)
        expected = np.array([
            [ 1.,  0., -1.,  0.],
            [ 0.,  0.,  0.,  0.],
            [-1.,  0.,  1.,  0.],
            [ 0.,  0.,  0.,  0.],
        ])
        np.testing.assert_allclose(k, expected, atol=1e-12)

    def test_out_of_range_raises(self):
        with pytest.raises(ValueError, match="Element index out of range"):
            make_single_bar().ElementStiffness(99)


# ─────────────────────────────────────────────────────────────
# assemble_global_stiffness
# ─────────────────────────────────────────────────────────────

class TestAssembleGlobalStiffness:

    def test_returns_correct_shape(self):
        t = PlaneTrussProblem([(0, 0), (1, 0), (2, 0)], [(0, 1), (1, 2)], 1.0, 1.0)
        assert t.assemble_global_stiffness().shape == (6, 6)

    def test_global_matrix_is_symmetric(self):
        t = PlaneTrussProblem([(0, 0), (1, 0), (2, 0)], [(0, 1), (1, 2)], 1.0, 1.0)
        K = t.assemble_global_stiffness()
        np.testing.assert_allclose(K, K.T, atol=1e-12)

    def test_two_collinear_bars_analytical(self):
        """Known result for two collinear unit bars (E=A=L=1):
        node DOFs: [ux0,uy0, ux1,uy1, ux2,uy2].
        """
        t = PlaneTrussProblem([(0, 0), (1, 0), (2, 0)], [(0, 1), (1, 2)], 1.0, 1.0)
        K = t.assemble_global_stiffness()
        expected = np.array([
            [ 1,  0, -1,  0,  0,  0],
            [ 0,  0,  0,  0,  0,  0],
            [-1,  0,  2,  0, -1,  0],
            [ 0,  0,  0,  0,  0,  0],
            [ 0,  0, -1,  0,  1,  0],
            [ 0,  0,  0,  0,  0,  0],
        ], dtype=float)
        np.testing.assert_allclose(K, expected, atol=1e-12)
    @pytest.mark.parametrize("shape_func",shape_funcs)
    def test_unconstrained_matrix_is_singular(self, shape_func):
        """Without boundary conditions the global K must be rank-deficient."""
        t = PlaneTrussProblem([(0, 0), (1, 0), (2, 0)], [(0, 1), (1, 2)], 1.0, 1.0,shape_func=shape_func)
        K = t.assemble_global_stiffness()
        assert np.linalg.matrix_rank(K) < K.shape[0]

    @pytest.mark.parametrize("shape_func",shape_funcs)
    def test_old_vs_new(self,shape_func):
        t = PlaneTrussProblem([(0, 0), (1, 0), (1, 1), (0, 1)],
                            [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)],
                                1.0, 1.0,
                                shape_func=shape_func)
        k= t.assemble_global_stiffness()
        k2= t.assemble_global_stiffness_old()
        assert np.array_equal(k,k2)
        assert np.linalg.matrix_rank(k) < k.shape[0]

    def test_right_triangle_matrix_shape_and_symmetry(self):
        t = make_right_triangle()
        K = t.assemble_global_stiffness()
        assert K.shape == (6, 6)
        np.testing.assert_allclose(K, K.T, atol=1e-12)

# ─────────────────────────────────────────────────────────────
# set_external_constraints
# ─────────────────────────────────────────────────────────────

class TestSetExternalConstraints:

    def test_one_free_dof_gives_1x1_matrix(self):
        t = make_single_bar()
        t.assemble_global_stiffness()
        K_free = t.set_external_constraints([0, 1, 3])
        assert K_free.shape == (1, 1)

    def test_two_free_dofs_gives_2x2_matrix(self):
        t = PlaneTrussProblem([(0, 0), (1, 0), (2, 0)], [(0, 1), (1, 2)], 1.0, 1.0)
        t.assemble_global_stiffness()
        # Constrain 4 out of 6 DOFs → 2 free
        K_free = t.set_external_constraints([0, 1, 3, 5])
        assert K_free.shape == (2, 2)

    def test_right_triangle_constraints_give_3x3_matrix(self):
        t = make_right_triangle()
        t.assemble_global_stiffness()
        # 6 DOFs total. Constraining 3 DOFs (0, 1, 3) leaves 3 free DOFs.
        K_free = t.set_external_constraints([0, 1, 3])
        assert K_free.shape == (3, 3)

# ─────────────────────────────────────────────────────────────
# solve
# ─────────────────────────────────────────────────────────────

class TestSolve:

    @pytest.mark.parametrize("shape_func",shape_funcs)
    def test_single_bar_axial_displacement(self,shape_func):
        """u_x at loaded end = F·L/(E·A) = F  for E=A=L=1."""
        t = solved_single_bar(F=100.0,shape_func=shape_func)
        assert t.displacements[2] == pytest.approx(100.0)

    @pytest.mark.parametrize("shape_func",shape_funcs)
    def test_single_bar_diagonal_displacement(self,shape_func):
        """u_x at loaded end = F·L/(E·A) = F  for E=A=L=1."""
        t = solved_single_bar_diagonal(F=100.0,shape_func=shape_func)
        assert t.displacements[2] == pytest.approx(100.0)

    @pytest.mark.parametrize("shape_func",shape_funcs)
    def test_constrained_dofs_are_zero(self,shape_func):
        t = solved_single_bar(F=50.0, shape_func=shape_func)
        for dof in (0, 1, 3):
            assert t.displacements[dof] == pytest.approx(0.0)

    def test_displacement_vector_length(self):
        t = solved_single_bar()
        assert len(t.displacements) == 4   # 2 nodes × 2 DOFs

    def test_displacement_matrix_shape(self):
        t = solved_single_bar()
        assert t.displacements_matrix.shape == (2, 2)

    def test_two_bar_midnode_displacement(self):
        """Middle node of 2 serial unit bars under axial load: u = F / (2EA/L)."""
        t = PlaneTrussProblem([(0, 0), (1, 0), (2, 0)], [(0, 1), (1, 2)], 1.0, 1.0, shape_func="quadratic")
        t.assemble_global_stiffness()
        F = 50.0
        # Pin nodes 0 and 2, roller y at node 1
        # FIX: Pass dictionary for DOF 2
        t.solve({2: F}, constrained_dofs=[0, 1, 3, 4, 5])
        assert t.displacements[2] == pytest.approx(F / 2.0)

    def test_right_triangle_constrained_dofs_are_zero(self):
        t = solved_right_triangle()
        for dof in (0, 1, 3):
            assert t.displacements[dof] == pytest.approx(0.0, abs=1e-12)
            
    def test_right_triangle_free_dofs_are_nonzero(self):
        t = solved_right_triangle()
        # Ensure node 2 shifted due to the 100/-50 load
        assert abs(t.displacements[4]) > 1e-6
        assert abs(t.displacements[5]) > 1e-6

# ─────────────────────────────────────────────────────────────
# get_displacement
# ─────────────────────────────────────────────────────────────

class TestGetDisplacement:

    def test_fixed_node_returns_zero(self):
        ux, uy = solved_single_bar().get_displacement(0)
        assert ux == pytest.approx(0.0)
        assert uy == pytest.approx(0.0)

    def test_loaded_node_matches_solution(self):
        F = 42.0
        ux, uy = solved_single_bar(F=F).get_displacement(1)
        assert ux == pytest.approx(F)
        assert uy == pytest.approx(0.0)

    def test_negative_index_raises(self):
        with pytest.raises(ValueError, match="Node index out of range"):
            solved_single_bar().get_displacement(-1)

    def test_out_of_bounds_index_raises(self):
        with pytest.raises(ValueError, match="Node index out of range"):
            solved_single_bar().get_displacement(100)

    def test_right_triangle_displacement_extraction(self):
        t = solved_right_triangle()
        ux, uy = t.get_displacement(2)
        assert ux == t.displacements[4]
        assert uy == t.displacements[5]

# ─────────────────────────────────────────────────────────────
# get_displacement_on_element
# ─────────────────────────────────────────────────────────────

class TestGetDisplacementOnElement:

    def test_xi_minus1_matches_start_node(self):
        """ξ = -1 must return the displacement of the first endpoint."""
        d = solved_single_bar(F=10.0).get_displacement_on_element(0, -1.0)
        np.testing.assert_allclose(d, [0.0, 0.0], atol=1e-12)

    def test_xi_plus1_matches_end_node(self):
        """ξ = +1 must return the displacement of the second endpoint."""
        d = solved_single_bar(F=10.0).get_displacement_on_element(0, 1.0)
        np.testing.assert_allclose(d, [10.0, 0.0], atol=1e-12)

    def test_xi_0_is_linear_midpoint(self):
        """At ξ = 0, linear shape functions give the simple average."""
        d_mid = solved_single_bar(F=10.0).get_displacement_on_element(0, 0.0)
        np.testing.assert_allclose(d_mid, [5.0, 0.0], atol=1e-12)

    def test_element_index_out_of_range_raises(self):
        with pytest.raises(ValueError):
            solved_single_bar().get_displacement_on_element(10, 0.0)

    def test_xi_greater_than_1_raises(self):
        with pytest.raises(ValueError, match="Node position out of range"):
            solved_single_bar().get_displacement_on_element(0, 1.5)

    def test_xi_less_than_minus1_raises(self):
        with pytest.raises(ValueError, match="Node position out of range"):
            solved_single_bar().get_displacement_on_element(0, -2.0)


# ─────────────────────────────────────────────────────────────
# get_reaction_forces
# ─────────────────────────────────────────────────────────────

class TestGetReactionForces:

    def test_x_direction_equilibrium(self):
        """Sum of all x-component reactions must vanish (Newton's 3rd law)."""
        for i in range(4):
            rxns = solved_single_bar(u=i,F=100.0).get_reaction_forces()
            assert rxns[0::2].sum() == pytest.approx(0.0, abs=1e-10)

    def test_support_reaction_equals_negative_applied_load(self):
        """The pinned-support reaction must exactly balance the applied force."""
        F = 75.0
        rxns = solved_single_bar(F=F).get_reaction_forces()
        assert rxns[0] == pytest.approx(-F, abs=1e-10)

    def test_right_triangle_equilibrium(self):
        t = solved_right_triangle()
        rxns = t.get_reaction_forces()
        
        # Static Equilibrium: Sum of reactions + Sum of applied forces = 0
        # External forces applied: Fx = 100 at DOF 4, Fy = -50 at DOF 5
        sum_fx = rxns[0::2].sum()
        sum_fy = rxns[1::2].sum()
        # print(rxns)
        assert sum_fx == pytest.approx(0.0, abs=1e-10)
        assert sum_fy == pytest.approx(0.0, abs=1e-10)

# ─────────────────────────────────────────────────────────────
# get_element_stress
# ─────────────────────────────────────────────────────────────

class TestGetElementStress:

    def test_stress_equals_force_over_area(self):
        """σ = F/A for a unit-area horizontal bar under axial load."""
        F, A = 50.0, 1.0
        assert solved_single_bar(F=F).get_element_stress(0) == pytest.approx(F / A)

    def test_linear_stress_is_constant_along_element(self):
        """The B-matrix for a linear element is constant → σ must be uniform."""
        t = solved_single_bar(F=30.0)
        stresses = [t.get_element_stress(0, xi=xi) for xi in (-1.0, 0.0, 1.0)]
        assert stresses[0] == pytest.approx(stresses[1])
        assert stresses[1] == pytest.approx(stresses[2])

    def test_out_of_range_index_raises(self):
        with pytest.raises(ValueError, match="Element index out of range"):
            solved_single_bar().get_element_stress(99)

    def test_right_triangle_element_stresses_are_computable(self):
        t = solved_right_triangle()
        # Verify the calculation executes cleanly and returns a scalar for multi-angle truss
        for i in range(t.num_elements):
            stress = t.get_element_stress(i)
            assert isinstance(stress, (float, np.float64))


# ─────────────────────────────────────────────────────────────
# get_element_force
# ─────────────────────────────────────────────────────────────

class TestGetElementForce:

    def test_force_equals_stress_times_area(self):
        """N = σ · A must hold for any E and A."""
        E, A = 2.0, 0.5
        t = PlaneTrussProblem([(0, 0), (1, 0)], [(0, 1)], E, A)
        t.assemble_global_stiffness()
        # FIX: Pass dictionary for DOF 2
        t.solve({2: 100.0}, [0, 1, 3])
        assert t.get_element_force(0) == pytest.approx(t.get_element_stress(0) * A)

    def test_internal_force_equals_applied_load(self):
        """In a statically determinate single bar the axial force = external load."""
        F = 75.0
        assert solved_single_bar(F=F).get_element_force(0) == pytest.approx(F)

    def test_out_of_range_index_raises(self):
        with pytest.raises(ValueError, match="Element index out of range"):
            solved_single_bar().get_element_force(99)


# ─────────────────────────────────────────────────────────────
# add_inclined_support
# ─────────────────────────────────────────────────────────────

class TestAddInclinedSupport:

    def test_zero_angle_leaves_matrix_unchanged(self):
        """A 0° rotation is the identity transform → K must be unaffected."""
        t = make_single_bar()
        t.assemble_global_stiffness()
        K_before = t.k_global.copy()
        t.add_inclined_support(0, 0.0)
        np.testing.assert_allclose(t.k_global, K_before, atol=1e-12)

    def test_result_remains_symmetric(self):
        """Orthogonal similarity transforms preserve symmetry."""
        t = make_single_bar()
        t.assemble_global_stiffness()
        t.add_inclined_support(0, 45.0)
        np.testing.assert_allclose(t.k_global, t.k_global.T, atol=1e-12)

    def test_negative_node_index_raises(self):
        t = make_single_bar()
        t.assemble_global_stiffness()
        with pytest.raises(ValueError, match="Node index out of range"):
            t.add_inclined_support(-1, 30.0)

    def test_out_of_bounds_node_index_raises(self):
        t = make_single_bar()
        t.assemble_global_stiffness()
        with pytest.raises(ValueError, match="Node index out of range"):
            t.add_inclined_support(100, 30.0)

# ─────────────────────────────────────────────────────────────
# Test Stiffness Matrix Equivalence (Old vs New)
# ─────────────────────────────────────────────────────────────

class TestStiffnessEquivalence:

    @pytest.mark.parametrize("nodes, angle_desc", [
        ([(0, 0), (1, 0)], "0_deg"),
        ([(0, 0), (0, 1)], "90_deg"),
        ([(0, 0), (1, 1)], "45_deg"),
        ([(0, 0), (-1, 2)], "obtuse_angle")
    ])
    def test_linear_stiffness_equivalence(self, nodes, angle_desc):
        """Verify that analytical (old) and numerical (new) stiffness match for linear elements."""
        t = PlaneTrussProblem(nodes, [(0, 1)], elasticity_modulus=210e9, cross_sectional_area=0.01, shape_func="linear")
        k_old = t.ElementStiffness_old(0)
        k_new = t.ElementStiffness(0)
        np.testing.assert_allclose(k_old, k_new, atol=1e-10)

    @pytest.mark.parametrize("nodes, angle_desc", [
        ([(0, 0), (1, 0)], "0_deg"),
        ([(0, 0), (1, 1)], "45_deg")
    ])
    def test_quadratic_stiffness_equivalence(self, nodes, angle_desc):
        """Verify that analytical (old) and numerical (new) stiffness match for quadratic elements."""
        t = PlaneTrussProblem(nodes, [(0, 1)], elasticity_modulus=210e9, cross_sectional_area=0.01, shape_func="quadratic")
        # For quadratic, the init creates a mid-node, so the element array is updated
        k_old = t.ElementStiffness_old(0)
        k_new = t.ElementStiffness(0)
        np.testing.assert_allclose(k_old, k_new, atol=1e-10)


# ─────────────────────────────────────────────────────────────
# Parameterized 2D Geometries & External Loads
# ─────────────────────────────────────────────────────────────

class TestParameterized2DTrusses:

    @pytest.mark.parametrize("shape_func",shape_funcs)
    @pytest.mark.parametrize("name, nodes, elements, constrained_dofs, external_forces_dict", [
        (
            "Right_Triangle_Pin_Roller",
            [(0, 0), (1, 0), (0, 1)],
            [(0, 1), (1, 2), (0, 2)],
            [0, 1, 3],  # Node 0 pinned (X,Y), Node 1 roller (Y-fixed)
            {4: 100.0, 5: -50.0}  # Node 2 loaded in X (DOF 4) and Y (DOF 5)
        ),
        (
            "Square_Cross_Braced_Cantilever",
            [(0, 0), (1, 0), (1, 1), (0, 1)],
            [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)],
            [0, 1, 6, 7],  # Nodes 0 and 3 pinned to a wall
            {3: -200.0, 5: -200.0}  # Downward loads on free nodes 1 and 2 (Y DOFs 3 and 5)
        ),
        (
            "Three_Triangle_Bridge",
            [(0, 0), (1, 0), (2, 0), (0.5, 1), (1.5, 1)],
            [(0, 1), (1, 2), (0, 3), (1, 3), (1, 4), (2, 4), (3, 4)],
            [0, 1, 5],  # Node 0 pinned, Node 2 roller (Y-fixed)
            {3: -500.0}  # Downward load at Node 1 (Y DOF 3)
        ),
        (
            "Symmetric_Roof_Truss",
            [(0, 0), (2, 0), (4, 0), (1, 1.5), (3, 1.5)],
            [(0, 1), (1, 2), (0, 3), (3, 1), (1, 4), (4, 2), (3, 4)],
            [0, 1, 5],  # Pinned at left (Node 0), Roller at right (Node 2)
            {7: -150.0, 9: -150.0} # Symmetrical downward loads on roof peaks (Nodes 3 and 4)
        )
    ])
    def test_global_equilibrium_and_constraints(self, shape_func, name, nodes, elements, constrained_dofs, external_forces_dict):
        """
        Dynamically tests various truss geometries to ensure they satisfy basic laws of physics:
        1. Nodes that are supposed to be fixed don't move.
        2. The entire structure remains in static equilibrium (Newton's First Law).
        Evaluates both linear and quadratic shape functions.
        """
        # Standardize material properties for these tests
        E, A = 210e9, 0.01  
        
        # Initialize with the parameterized shape function
        t = PlaneTrussProblem(nodes, elements, E, A, shape_func=shape_func)
        t.assemble_global_stiffness()

        # Build external force vector from the dictionary
        # CRITICAL FIX: Use t.num_nodes to account for the appended mid-nodes in quadratic elements
        F = np.zeros(2 * t.num_nodes)
        for dof, val in external_forces_dict.items():
            F[dof] = val

        # Solve the system
        t.solve(F, constrained_dofs)
        rxns = t.get_reaction_forces()

        # 1. Verify constrained DOFs have strictly 0 displacement
        for dof in constrained_dofs:
            assert t.displacements[dof] == pytest.approx(0.0, abs=1e-12), \
                f"[{name} - {shape_func}] Displacement at constrained DOF {dof} was not 0!"

        # 2. Verify static equilibrium in X direction (Sum of Fx = 0)
        # Note: sum of rxns includes both the applied external forces and the reaction forces
        sum_fx = np.sum(rxns[0::2])
        assert sum_fx == pytest.approx(0.0, abs=1e-6), \
            f"[{name} - {shape_func}] Failed X-axis static equilibrium!"

        # 3. Verify static equilibrium in Y direction (Sum of Fy = 0)
        sum_fy = np.sum(rxns[1::2])
        assert sum_fy == pytest.approx(0.0, abs=1e-6), \
            f"[{name} - {shape_func}] Failed Y-axis static equilibrium!"

    @pytest.mark.parametrize("E, A, force_val, expected_stress", [
        (200e9, 0.05, 1000.0, 1000.0 / 0.05),
        (70e9,  0.01, -500.0, -500.0 / 0.01)
    ])
    def test_parameterized_stress_calculation(self, E, A, force_val, expected_stress):
        """Ensure stress is calculated correctly across different materials and cross-sections."""
        t = PlaneTrussProblem([(0, 0), (2, 0)], [(0, 1)], E, A)
        t.assemble_global_stiffness()
        
        # Load in X at Node 1
        F = np.zeros(4)
        F[2] = force_val
        t.solve(F, [0, 1, 3])
        
        assert t.get_element_stress(0) == pytest.approx(expected_stress, rel=1e-6)

# ─────────────────────────────────────────────────────────────
# Test Inclined Support Transformations
# ─────────────────────────────────────────────────────────────

class TestInclinedSupportTransformations:

    def test_90_degree_support_swaps_stiffness(self):
        """A 90-degree inclined support should effectively swap the X and Y stiffness components."""
        t = PlaneTrussProblem([(0, 0), (1, 0)], [(0, 1)], 1.0, 1.0)
        t.assemble_global_stiffness()
        
        k_original = t.k_global.copy()
        
        # Apply 90 degree support at node 0
        t.add_inclined_support(node_index=0, angle=90.0)
        
        # Original k_xx for node 0 should now be at k_yy (DOF 1,1)
        assert t.k_global[1, 1] == pytest.approx(k_original[0, 0], abs=1e-10)
        # Original k_yy for node 0 should now be at k_xx (DOF 0,0)
        assert t.k_global[0, 0] == pytest.approx(k_original[1, 1], abs=1e-10)