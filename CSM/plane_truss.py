"""Module for solving 2D plane truss problems."""

import numpy as np
import matplotlib.pyplot as plt
import yaml
class PlaneTrussProblem:
    """Class for solving 2D plane truss problems."""
    
    def __init__(self, nodes, elements, elasticity_modulus, cross_sectional_area, shape_func="linear"):
        """
        Initialize the plane truss problem.

        Parameters:
        nodes (list of tuples): List of node coordinates (x, y).
        elements (list of tuples): List of elements defined by node indices.
        elasticity_modulus (float): Elasticity modulus of the material.
        cross_sectional_area (float): Cross-sectional area of the truss members.
        shape_func (string): [linear, quadratic]
        Example:
        >>> nodes = [(0, 0), (1, 0), (1, 1), (0, 1)]
        >>> elements = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]
        >>> elasticity_modulus = 210e9
        >>> cross_sectional_area = 0.01
        >>> truss = PlaneTrussProblem(nodes, elements, elasticity_modulus, cross_sectional_area)
        
        """
        
        self.nodes = np.array(nodes)
        self.elements = np.array(elements)
        self.num_nodes = len(nodes)
        self.num_elements = len(elements)
        self.E = np.full(self.num_elements,elasticity_modulus)
        self.A = np.full(self.num_elements,cross_sectional_area)

        #TODO getter of shape function
        self.shape_func = shape_func
        
        # Perform initial checks
        if self.num_nodes < 2:
            raise ValueError("At least two nodes are required.")
        if self.num_elements < 1:
            raise ValueError("At least one element is required.")
        if self.nodes.shape[1] != 2:
            raise ValueError("Node coordinates must be 2D (x, y).")
        if self.elements.shape[1] != 2:
            raise ValueError("Elements must be defined by two node indices.")
        if  self.shape_func not in ["linear", "quadratic"]:
            raise ValueError("not a valid shape function")
        
        #NOTE we are going to mantain everything on a "element" in [-1,1], demostrations are written on paper
        #NOTE with respect to lecture notes N2 and N3 are inverted

        self.shape_functions_array = {
            2: lambda x: np.array([(1-x)/2, (x+1)/2]),
            3: lambda x: np.array([0.5*x*(x-1), (1-x**2), 0.5*x*(x+1)])
        }

        #NOTE derived by hand, maybe there is a way to derive via python
        self.strain_dispacement_array = {
            2: lambda x,l: (2/l)* np.array([-0.5,0.5]),
            3: lambda x,l: (2/l)* np.array([x-0.5,-2*x,x+0.5])
        }
    
        # Compute the plane truss elements' lengths
        #length and angle don't change by adding nodes in the middle so i just compute them now
        self.L = np.zeros(self.num_elements)
        for i, (n1, n2) in enumerate(self.elements):
            x1, y1 = self.nodes[n1]
            x2, y2 = self.nodes[n2]
            self.L[i] = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            
        # Compute the angles of the elements in degrees
        self.angles = np.zeros(self.num_elements)
        for i, (n1, n2) in enumerate(self.elements):
            x1, y1 = self.nodes[n1]
            x2, y2 = self.nodes[n2]
            print(x1, y1)
            print(x2, y2)
            self.angles[i] = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi  # Angles in degrees
            
            if self.angles[i] < 0:
                self.angles[i] += 360
        
        #create middle points of elements for quadratic shape_func
        if self.shape_func != "linear":
            nodes_per_element = {"quadratic": 3, "cubic": 4}[self.shape_func]
            n_interior = nodes_per_element - 2  # 1 for quadratic, 2 for cubic
            num_seg = nodes_per_element - 1

            new_nodes = []
            new_elements = []

            for i, [n1, n2] in enumerate(self.elements):
                interior_ids = []
                for n in range(n_interior):
                    x_pos = self.nodes[n1][0] + (n+1) * (self.nodes[n2][0] - self.nodes[n1][0]) / num_seg
                    y_pos = self.nodes[n1][1] + (n+1) * (self.nodes[n2][1] - self.nodes[n1][1]) / num_seg
                    new_node_id = self.num_nodes + len(new_nodes)
                    new_nodes.append([x_pos, y_pos])
                    interior_ids.append(new_node_id)
                
                new_elements.append([n1] + interior_ids + [n2])

            if new_nodes:
                self.nodes = np.vstack([self.nodes, new_nodes])
            self.elements = np.array(new_elements)

        self.num_nodes = len(self.nodes)

        # Initialize global stiffness matrix
        self.k_global = np.zeros((2 * self.num_nodes, 2 * self.num_nodes))

    @classmethod
    def load_file(cls,file_path:str):
        """
        give in input a yaml file with the description of the nodes and object
        
        structure of file:
        default: contains the generic infos for each element or the structure as a whole
            shape_func: [2,3] number of 

        """
        with open(file_path, "r") as file:
            structure = yaml.safe_load(file)



    def plot_plane_truss(self, show_node_indices=True, show_element_indices=True, show_deformed=False, scale_factor=1.0):
        """
        Function to plot the plane truss:
        - Nodes are represented as circles and optionally labeled with their indices.
        - Elements are represented as lines connecting the nodes and optionally labeled with their indices.

        Parameters:
        show_node_indices (bool): If True, display the node indices on the plot.
        show_element_indices (bool): If True, display the element indices on the plot.
        
        """
        plt.figure(figsize=(8, 6))
        
        # plot original structure
        for i, e in enumerate(self.elements):
            n1, n2 = e[0], e[-1]  # works for both 2 and 3 node elements
            x1, y1 = self.nodes[n1]
            x2, y2 = self.nodes[n2]
            plt.plot([x1, x2], [y1, y2], 'b--', lw=1.5, label='Original' if i==0 else '')
            if show_element_indices:
                mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
                plt.text(mid_x, mid_y, f"E{i}", color='blue', fontsize=12, ha='center', va='center',
                         bbox=dict(facecolor='white', edgecolor='blue', boxstyle='round,pad=0.3'))
        
        # plot deformed structure
        if show_deformed and hasattr(self, 'displacements'):
            deformed_nodes = self.nodes + self.displacements.reshape(-1, 2) * scale_factor
            for i, e in enumerate(self.elements):
                n1, n2 = e[0], e[-1]
                x1, y1 = deformed_nodes[n1]
                x2, y2 = deformed_nodes[n2]
                plt.plot([x1, x2], [y1, y2], 'r-', lw=2, label='Deformed' if i==0 else '')
        
        plt.legend()
        plt.title("Plane Truss Structure")
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.axis('equal')
        plt.grid(True, alpha=0.5, linestyle='--', linewidth=0.5)
        plt.show()
        
    def get_angle(self, element_index):
        """
        Get the angle of the element in degrees.

        Parameters:
        element_index (int): Index of the element.

        Returns:
        float: Angle of the element in degrees.
        
        """
        if element_index < 0 or element_index >= self.num_elements:
            raise ValueError("Element index out of range.")
        
        return self.angles[element_index]
    
    def get_length(self, element_index):
        """
        Get the length of the element.

        Parameters:
        element_index (int): Index of the element.

        Returns:
        float: Length of the element.
        
        """
        if element_index < 0 or element_index >= self.num_elements:
            raise ValueError("Element index out of range.")
        
        return self.L[element_index]
    
    def ElementStiffness_old(self, element_index):
        """
        Compute the element stiffness matrix for a given element.
        The size of the element stiffness matrix is 4x4.

        Parameters:
        element_index (int): Index of the element.

        Returns:
        np.ndarray: Stiffness matrix of the element.
        
        """
        #TODO generalize to a generic function, it's a bit of a hassle to deal with all the derivations


        if element_index < 0 or element_index >= self.num_elements:
            raise ValueError("Element index out of range.")
        
        x = self.angles[element_index] * np.pi / 180  # Convert angle to radians
        c = np.cos(x)
        s = np.sin(x)
        L = self.L[element_index]
        if(len(self.elements[element_index])==3):
            # FIX: Swap rows 2 and 3 with rows 4 and 5 to put the mid-node in the middle spatially
            T = np.array([[c, s, 0, 0, 0, 0],
                        [0, 0, 0, 0, c, s],   # Node 2 (end node) maps to local position 2
                        [0, 0, c, s, 0, 0]])  # Node 3 (mid node) maps to local position 3
            
            k_local = (self.E[element_index] * self.A[element_index] / (3*L)) * np.array([[7, 1, -8],
                                                                                        [1, 7, -8],
                                                                                        [-8, -8, 16],])
            k = np.transpose(T)@k_local@T
        else:
            k = (self.E[element_index] * self.A[element_index] / L) * np.array([[c**2, c*s, -c**2, -c*s],
                                       [c*s, s**2, -c*s, -s**2],
                                       [-c**2, -c*s, c**2, c*s],
                                       [-c*s, -s**2, c*s, s**2]])
        return k

    def ElementStiffness(self, element_index):    
        if element_index < 0 or element_index >= self.num_elements:
            raise ValueError("Element index out of range.")
        
        L = self.L[element_index]
        theta = self.angles[element_index] * np.pi / 180
        print(self.get_angle(element_index))
        c, s = np.cos(theta), np.sin(theta)
        print(self.angles[element_index] ,theta, c,s)
        n_nodes = len(self.elements[element_index])
        
        # transformation matrix (projects global DOFs to local axial)
        T = np.zeros((n_nodes, 2 * n_nodes))
        for i in range(n_nodes):
            T[i, 2*i]   = c
            T[i, 2*i+1] = s
        
        # Gauss quadrature points and weights
        # 2 points exact for linear, 3 points for quadratic
        
        degree_BtB = 2 * (n_nodes - 2)
        n_gauss = int(np.ceil((degree_BtB + 1) / 2))
        n_gauss = max(n_gauss, 1)  # at least 1 point
        
        points, weights = np.polynomial.legendre.leggauss(n_gauss)
        
        # integrate K_local = EA * ∫ Bᵀ B (L/2) dξ
        B_func = self.strain_dispacement_array[n_nodes]
        K_local = np.zeros((n_nodes, n_nodes))
        for xi, w in zip(points, weights):
            B = B_func(xi, L)                        # (n_nodes,)
            K_local += w * np.outer(B, B) * (L/2)
        K_local *= self.E[element_index] * self.A[element_index]
        print(K_local)
        print(T)
        print(T.T @ K_local @ T)
        # transform to global frame: Tᵀ K_local T
        return T.T @ K_local @ T
    
    def assemble_global_stiffness(self):
        """
        Assemble the global stiffness matrix for the entire truss structure.

        Returns:
        np.ndarray: Global stiffness matrix.
        
        """
        for i, e in enumerate(self.elements):
            k_local = self.ElementStiffness(i)
            dof_indices = [dof for n in e for dof in (2*n, 2*n+1)]
            for a, da in enumerate(dof_indices):
                for b, db in enumerate(dof_indices):
                    self.k_global[da, db] += k_local[a, b]
        return self.k_global
    
    def assemble_global_stiffness_old(self):
        """
        Assemble the global stiffness matrix for the entire truss structure.

        Returns:
        np.ndarray: Global stiffness matrix.
        
        """
        for i, e in enumerate(self.elements):
            k_local = self.ElementStiffness_old(i)
            dof_indices = [dof for n in e for dof in (2*n, 2*n+1)]
            for a, da in enumerate(dof_indices):
                for b, db in enumerate(dof_indices):
                    self.k_global[da, db] += k_local[a, b]
        return self.k_global
    
    def add_inclined_support(self, node_index, angle):
        """
        Add an inclined support to the truss at a specific node.

        Parameters:
        node_index (int): Index of the node where the support is applied.
        angle (float): Angle of the inclined support in degrees.
        
        """
        if node_index < 0 or node_index >= self.num_nodes:
            raise ValueError("Node index out of range.")
        
        transformation_matrix = np.eye(2 * self.num_nodes)
        x = angle * np.pi / 180  # Convert angle to radians
        transformation_matrix[2*node_index, 2*node_index] = np.cos(x)
        transformation_matrix[2*node_index, 2*node_index + 1] = np.sin(x)
        transformation_matrix[2*node_index + 1, 2*node_index] = -np.sin(x)
        transformation_matrix[2*node_index + 1, 2*node_index + 1] = np.cos(x)
        self.k_global = transformation_matrix @ self.k_global @ (transformation_matrix.T)
    
    def set_external_constraints(self, constrained_dofs):
        """
        Partition the global stiffness matrix by eliminating the rows and columns corresponding to the constrained degrees of freedom.

        Parameters:
        constrained_dofs (list of int): List of constrained degrees of freedom.

        Returns:
        np.ndarray: Partitioned global stiffness matrix.
        
        """
        free_dofs = np.setdiff1d(np.arange(2 * self.num_nodes), constrained_dofs)
        return self.k_global[np.ix_(free_dofs, free_dofs)]
    
    def solve(self, external_forces, constrained_dofs):
        """
        Solve the truss problem for the given external forces and constraints.

        Parameters:
        external_forces (np.ndarray or dict): External forces applied to the nodes/DOFs.
        constrained_dofs (list of int): List of constrained degrees of freedom.

        Returns:
        np.ndarray: Displacements of all degrees of freedom.
        
        """

        # if self.shape_func == "quadratic":
        #     for elem in self.elements:
        #         mid = int(elem[1])
        #         theta = np.arctan2(
        #             self.nodes[elem[2]][1] - self.nodes[elem[0]][1],
        #             self.nodes[elem[2]][0] - self.nodes[elem[0]][0],
        #         )
        #         c, s = np.cos(theta), np.sin(theta)
        #         # Transverse DOF: if element is horizontal → y (2*mid+1); vertical → x (2*mid)
        #         # General: the DOF more orthogonal to the element axis
        #         if abs(s) > abs(c):          # more vertical → transverse is x
        #             transverse_dof = 2 * mid
        #         else:                        # more horizontal → transverse is y
        #             transverse_dof = 2 * mid + 1
        #         if transverse_dof not in constrained_dofs:
        #             constrained_dofs.append(transverse_dof)
                    
        total_dofs = 2 * self.num_nodes
        free_dof_indices = np.setdiff1d(np.arange(total_dofs), constrained_dofs)
        
        # 1. Standardize external_forces into a full global vector
        if isinstance(external_forces, dict):
            f_global = np.zeros(total_dofs)
            for dof, val in external_forces.items():
                f_global[dof] = val
            external_forces = f_global
        else:
            external_forces = np.asarray(external_forces)
            # Backward-compatibility fallback: if input matches number of free DOFs
            if external_forces.ndim == 1 and external_forces.shape[0] == len(free_dof_indices):
                f_global = np.zeros(total_dofs)
                f_global[free_dof_indices] = external_forces
                external_forces = f_global
            elif external_forces.shape[0] != total_dofs:
                raise ValueError(f"external_forces length ({external_forces.shape[0]}) must match "
                                 f"total DOFs ({total_dofs}) or free DOFs ({len(free_dof_indices)}).")

        print("shape of external forces: ", len(external_forces), "  shape of stiffness matrix: ", np.shape(self.k_global))
        # 2. Partition BOTH the global stiffness matrix and the force vector
        k_free = self.k_global[np.ix_(free_dof_indices, free_dof_indices)]
        f_free = external_forces[free_dof_indices]
        
        print(k_free)
        # 3. Solve for displacements at unconstrained DOFs
        free_displacements = np.linalg.solve(k_free, f_free)
        
        # 4. Reconstruct the full global displacements vector
        displacements = np.zeros(total_dofs)
        displacements[free_dof_indices] = free_displacements
        
        
        self.displacements = displacements
        # Save as an Nx2 matrix for easy node-by-node extraction elsewhere
        self.displacements_matrix = displacements.reshape(-1, 2) 
        #print(self.get_reaction_forces())
        return displacements
    
    def get_displacement(self, node_index):
        """
        Get the displacement of a specific node.

        Parameters:
        node_index (int): Index of the node.

        Returns:
        tuple: Displacement of the node in x and y directions.
        
        """
        if node_index < 0 or node_index >= self.num_nodes:
            raise ValueError("Node index out of range.")
        
        return self.displacements[2*node_index], self.displacements[2*node_index + 1]
    
    def get_displacement_on_element(self, element_index, x):
        """
        choose an element, get the displacement at that point x on the element going from node 0 to node 1 of the element.

        Parameters:
        element_index (int): index of the element,
        x (float [-1,1]): position on the element of which the element is desired, -1 is the first node of the element, 1 is the second node of the element.

        Returns:
        tuple: Displacement of the point in x and y directions.

        """
        if element_index < 0 or element_index >= self.num_elements:
            raise ValueError("Node index out of range.")
        
        if x < -1 or x > 1:
            raise ValueError("Node position out of range.")
        
        nodes=self.elements[element_index] #tuple 

        displacements = self.displacements_matrix[nodes]
        return self.shape_functions_array[len(nodes)](x) @ displacements

    def get_reaction_forces(self):
        """
        Calculate the reaction forces.

        Returns:
        np.ndarray: Reaction forces.
        
        """
        reaction_forces = self.k_global @ self.displacements
        self.reaction_forces = reaction_forces
        return reaction_forces
    
    def get_element_stress(self, element_index, xi=0.0):
        """
        Calculate the stress in a specific point of an element.
        for linear element is independent from position.
        Parameters:
        element_index (int): Index of the element.

        Returns:
        float: Stress in the element.
        
        """

        if element_index < 0 or element_index >= self.num_elements:
            raise ValueError("Element index out of range.")

        L = self.L[element_index]
        theta = self.angles[element_index] * np.pi / 180
        c, s = np.cos(theta), np.sin(theta)
        nodes = self.elements[element_index]
        n_nodes = len(nodes)

        direction = np.array([c, s])
        d_global = self.displacements_matrix[nodes] 
        d_local = d_global @ direction

        B = self.strain_dispacement_array[n_nodes](xi, L)
        return self.E[element_index] * B @ d_local
    
    def get_element_force(self, element_index, x=0.0):
        """
        Calculate the force in a specific element.

        Parameters:
        element_index (int): Index of the element.

        Returns:
        float: Force in the element.
        
        """
        if element_index < 0 or element_index >= self.num_elements:
            raise ValueError("Element index out of range.")
        
        stress = self.get_element_stress(element_index, x)
        force = stress * self.A[element_index]
        return force
    