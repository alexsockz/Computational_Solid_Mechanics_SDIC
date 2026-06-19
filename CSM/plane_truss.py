"""Module for solving 2D plane truss problems."""

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import yaml

class PlaneTrussProblem:
    """Class for solving 2D plane truss problems."""

# ___________________________________________________________________
#
#  CONSTANTS FUNCTIONS
#
# ___________________________________________________________________
    MESSAGE_NODE_IDX="Node index out of range."
    MESSAGE_ELEMENT_IDX="Element index out of range."


    shape_functions_array = {
            2: lambda x: np.array([(1-x)/2, (x+1)/2]),
            3: lambda x: np.array([0.5*x*(x-1), (1-x**2), 0.5*x*(x+1)])
        }

        #NOTE derived by hand, maybe there is a way to derive via python
    strain_dispacement_array = {
            2: lambda x,l: (2/l)* np.array([-0.5,0.5]),
            3: lambda x,l: (2/l)* np.array([x-0.5,-2*x,x+0.5])
        }
# ___________________________________________________________________
#
#  INIT FUNCTIONS
#
# ___________________________________________________________________

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
        
        self.nodes = np.array(nodes, np.float64)
        self.elements = np.array(elements,np.int8)
        self.num_nodes = len(nodes)
        self.num_elements = len(elements)

        self.E = self.__into_array_if_not(elasticity_modulus)            
        self.A = self.__into_array_if_not(cross_sectional_area)

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
            self.angles[i] = np.arctan2(y2 - y1, x2 - x1)
            
            if self.angles[i] < 0:
                self.angles[i] += 2*np.pi
        
        # Store original (structural) node count before mid-nodes are added
        self.num_original_nodes = self.num_nodes

        #create middle points of elements for quadratic shape_func
        if self.shape_func != "linear":
            self.__add_mid_node()

        self.num_nodes = len(self.nodes)

        # After static condensation the mid-nodes are eliminated from the global
        # system, so k_global is sized on the original structural nodes only.
        n_global = self.num_original_nodes
        self.k_global = np.zeros((2 * n_global, 2 * n_global))


    @classmethod
    def load_file(cls,file_path:str="structure.yaml"):
        """
        give in input a yaml file with the description of the nodes and object
        
        structure of file:
        default: contains the generic infos for each element or the structure as a whole
            shape_func: [2,3] number of 

        """
        with open(file_path, "r") as file:
            structure = yaml.safe_load(file)
        
        n_elem=len(structure["elements"])
        elements=[]
        
        #globals: elastic modulus and cross sectional area should be a list, since i prepare it here
        elastic_mod = np.full(n_elem,structure["defaults"]["E"])
        cross_sec = np.full(n_elem,structure["defaults"]["A"])
        if "forces" in structure["defaults"]:
            F_elements=np.full(2*n_elem,structure["defaults"]["forces"])
        else:
            F_elements=np.zeros(2*n_elem)
        #force and type of struct tbd
        
        #formatting the nodes
        nodes, constraints, inclined_support, F_nodes = cls.__format_nodes(structure)
        #formatting the elements

 
        print(structure["elements"])
        for i,e in enumerate(structure["elements"]):
        # already formatted as lists 
            if(len(e["nodes"])==2):
                elements.append(e["nodes"])
            else:
                raise ValueError("invalid element")
        # elastic mod and cross sec area not necessary but can be set per element
            if "E" in e:
                elastic_mod[i]=e["E"]
            if "A" in e:
                cross_sec[i]=e["A"]
            if "force" in e:
                F_elements[2*i]=e["force"][0]
                F_elements[2*i+1]=e["force"][1]

        print(F_elements)

        #per element's type tbd
        #TODO maybe give to solve F_elements, so i move the responsibility from here to later and give the opportunity to hand generated to have element force

        #remaining question: should i directly solve it? answer yes
        truss= PlaneTrussProblem(nodes=nodes,elements=elements,elasticity_modulus=elastic_mod,cross_sectional_area=cross_sec,shape_func=structure["defaults"]["shape_func"])
        truss.assemble_global_stiffness()
        
        truss.solve(F_nodes, constraints, element_forces=F_elements, inclined_support=inclined_support)
        return truss
    
    #REMOVE before release
    def ElementStiffness_old(self, element_index):
        """
        Compute the element stiffness matrix for a given element.
        The size of the element stiffness matrix is 4x4.

        Parameters:
        element_index (int): Index of the element.

        Returns:
        np.ndarray: Stiffness matrix of the element.
        
        """

        if element_index < 0 or element_index >= self.num_elements:
            raise ValueError(self.MESSAGE_ELEMENT_IDX)
        
        x = self.angles[element_index] #already radians
        c, s = np.cos(x), np.sin(x)
        #error correction because cos(1)!=0 in python, it gives a small number but it annoys me
        # if s>c:
        #     a=np.sqrt(1-np.pow(s,2))
        #     c=a if a<c else c
        # else:
        #     a=np.sqrt(1-np.pow(c,2))
        #     s=a if a<s else s


        L = self.L[element_index]
        if(len(self.elements[element_index])==3):
            K_local = (self.E[element_index] * self.A[element_index] / (3*L)) * np.array([
                                                                                        [ 7, -8,  1],
                                                                                        [-8, 16, -8],
                                                                                        [ 1, -8,  7]
                                                                                    ])
            # Static condensation: eliminate the internal mid-node (local index 1).
            # Local node ordering is [n1(0), mid(1), n2(2)].
            # Active (end) nodes: indices [0, 2]; internal (mid) node: index [1].
            active = [0, 2]
            internal = [1]

            K_aa = K_local[np.ix_(active, active)]      # 2x2: end nodes
            K_ai = K_local[np.ix_(active, internal)]    # 2x1: end -> mid
            K_ia = K_local[np.ix_(internal, active)]    # 1x2: mid -> end
            K_ii = K_local[np.ix_(internal, internal)]  # 1x1: mid -> mid

            # Condensed local stiffness (2x2): K_cond = K_aa - K_ai * K_ii^{-1} * K_ia
            K_ii_inv = np.linalg.inv(K_ii)
            K_local = K_aa - K_ai @ K_ii_inv @ K_ia

            # Build a 2-node transformation matrix for the two end nodes only
            T = np.zeros((2, 4))
            T[0, 0] = c;  T[0, 1] = s   # end node 0
            T[1, 2] = c;  T[1, 3] = s   # end node 1

            return T.T @ K_local @ T
        else:
            k = (self.E[element_index] * self.A[element_index] / L) * np.array([[c**2, c*s, -c**2, -c*s],
                                       [c*s, s**2, -c*s, -s**2],
                                       [-c**2, -c*s, c**2, c*s],
                                       [-c*s, -s**2, c*s, s**2]])
        return k

    def ElementStiffness(self, element_index):    
        if element_index < 0 or element_index >= self.num_elements:
            raise ValueError(self.MESSAGE_ELEMENT_IDX)
        
        theta = self.angles[element_index]
        c, s = np.cos(theta), np.sin(theta)
        #error correction because cos(1)!=0 in python, it gives a small number but it annoys me
        # if s>c:
        #     a=np.sqrt(1-np.pow(s,2))
        #     c=a if a<c else c
        # else:
        #     a=np.sqrt(1-np.pow(c,2))
        #     s=a if a<s else s

        if self.shape_func== "linear":
            K_local = self.__calc_k_local(element_index)
        else:
            K_local, _, _, _, _=self.__calc_k_local(element_index)

            # Build a 2-node transformation matrix for the two end nodes only
        T = np.zeros((2, 4))
        T[0, 0] = c;  T[0, 1] = s   # end node 0
        T[1, 2] = c;  T[1, 3] = s   # end node 1
        return T.T @ K_local @ T
    
    def assemble_global_stiffness(self):
        """
        Assemble the global stiffness matrix for the entire truss structure.

        Returns:
        np.ndarray: Global stiffness matrix.
        
        """
        for i, e in enumerate(self.elements):
            k_elem = self.ElementStiffness(i)
            if self.shape_func == "quadratic":
                # After static condensation k_elem is 4x4 (end nodes only).
                # e = [n1, mid, n2]; end nodes are e[0] and e[-1].
                e= np.delete(e,1)
            dof_indices = [dof for n in e for dof in (2*n, 2*n+1)]
            for a, da in enumerate(dof_indices):
                for b, db in enumerate(dof_indices):
                    self.k_global[da, db] += k_elem[a, b]
        return self.k_global
    
    #REMOVE before release
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
    
    def reduce_forces(self, external_forces):
        """
        Takes a full force vector (sized 2*num_nodes, including mid-nodes)
        and returns a condensed vector (sized 2*num_original_nodes) where
        mid-node forces have been transferred to end-nodes via Guyan condensation.
        """
        if self.shape_func == "linear":
            return external_forces[:2 * self.num_original_nodes]

        # Start with end-node forces only (no mid-nodes)
        reduced = external_forces[:2 * self.num_original_nodes].copy()

        for element_index in range(self.num_elements):
            _, _, K_ai, K_ii_inv, _ = self.__calc_k_local(element_index)

            active   = [0, 2]
            internal = [1]
            active_element_nodes   = self.elements[element_index][active]
            internal_element_nodes = self.elements[element_index][internal]

            # Only read the MID-NODE forces
            f_x_internal = external_forces[2 * internal_element_nodes]
            f_y_internal = external_forces[2 * internal_element_nodes + 1]

            # Compute correction to transfer mid-node load to end-nodes
            condenser = K_ai @ K_ii_inv
            correction_x = -(condenser @ f_x_internal).flatten()
            correction_y = -(condenser @ f_y_internal).flatten()

            # Add correction to the end-node entries only
            reduced[2 * active_element_nodes]     += correction_x
            reduced[2 * active_element_nodes + 1] += correction_y

        return reduced

    def solve(self, external_forces, constrained_dofs, element_forces=[], inclined_support={}):
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
        
        if inclined_support != ():
            self.set_inclined_support(inclined_support)
        # After static condensation the global system only contains original nodes.
        total_dofs_reduced = 2 * self.num_nodes
        
        # 1. Partition BOTH the global stiffness matrix and the force vector
        k_free, f_free, free_dof_indices=self.set_external_constraints_and_forces(constrained_dofs,external_forces, element_forces, total_dofs_reduced)
        print(free_dof_indices)
        # print(external_forces, constrained_dofs)
        # print(k_free, f_free)
        # 2. Solve for displacements at unconstrained DOFs
        free_displacements = np.linalg.solve(k_free, f_free)
        
        # 3. Reconstruct the full global displacements vector, Back-substitution to get the mid node back
        displacements=np.zeros(2*self.num_nodes)
        displacements[free_dof_indices] = free_displacements
        # print(displacements)
        self.displacements_matrix=displacements.reshape(-1, 2)
        if self.shape_func == "quadratic":
            for element_index in range(self.num_elements):
                nodes = self.elements[element_index]
                n1, mid, n2 = nodes[0], nodes[1], nodes[-1]
                L = self.L[element_index]
                theta = self.angles[element_index]
                c, s = np.cos(theta), np.sin(theta)
                direction = np.array([c, s])
                normal    = np.array([-s, c])

                # End-node global displacements
                d_end = self.displacements_matrix[[n1, n2]]   # (2,2)
                u_a   = d_end @ direction                     # axial projections (2,)

                # Rebuild K_local
                B_func = self.strain_dispacement_array[3]
                pts, wts = np.polynomial.legendre.leggauss(2)
                K_local = np.zeros((3, 3))
                for xi, w in zip(pts, wts):
                    B = B_func(xi, L)
                    K_local += w * np.outer(B, B) * (L / 2)
                K_local *= self.E[element_index] * self.A[element_index]

                active   = [0, 2]
                internal = [1]
                K_ia  = K_local[np.ix_(internal, active)]    # (1,2)
                K_ii  = K_local[np.ix_(internal, internal)]  # (1,1)

                f_mid_global = self.external_forces[[2*mid, 2*mid + 1]]
                f_i_local    = np.array([f_mid_global @ direction])   # (1,)

                # Correct back-substitution: u_mid = K_ii^{-1} (f_i - K_ia @ u_a)
                u_mid_local = (np.linalg.inv(K_ii) @ (f_i_local - K_ia @ u_a))[0]
                print(u_mid_local)
                # Transverse: kinematic average of end nodes
                v_a         = d_end @ normal
                v_mid_local = (v_a[0] + v_a[-1]) / 2.0

                d_mid = (u_mid_local * direction) + (v_mid_local * normal)
                displacements[2*mid]     = d_mid[0]
                displacements[2*mid + 1] = d_mid[1]

        self.displacements = displacements
        # Save as an Nx2 matrix for easy node-by-node extraction elsewhere
        self.displacements_matrix = displacements.reshape(-1, 2) 
        return displacements
    
# ___________________________________________________________________
#
#   GET FUNCTIONS
#
# ___________________________________________________________________


    def get_displacement(self, node_index):
        """
        Get the displacement of a specific node.

        Parameters:
        node_index (int): Index of the node.

        Returns:
        tuple: Displacement of the node in x and y directions.
        
        """
        if node_index < 0 or node_index >= self.num_nodes:
            raise ValueError(self.MESSAGE_NODE_IDX)
        
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
            raise ValueError(self.MESSAGE_NODE_IDX)
        
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
        reaction_forces = self.k_global @ self.displacements[:2*self.num_original_nodes]
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
            raise ValueError(self.MESSAGE_ELEMENT_IDX)

        L = self.L[element_index]
        theta = self.angles[element_index]
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
            raise ValueError(self.MESSAGE_ELEMENT_IDX)
        
        stress = self.get_element_stress(element_index, x)
        force = stress * self.A[element_index]
        return force
    
    def get_angle(self, element_index):
        """
        Get the angle of the element in degrees.

        Parameters:
        element_index (int): Index of the element.

        Returns:
        float: Angle of the element in degrees.
        
        """
        if element_index < 0 or element_index >= self.num_elements:
            raise ValueError(self.MESSAGE_ELEMENT_IDX)
        
        return np.rad2deg(self.angles[element_index])
    
    def get_length(self, element_index):
        """
        Get the length of the element.

        Parameters:
        element_index (int): Index of the element.

        Returns:
        float: Length of the element.
        
        """
        if element_index < 0 or element_index >= self.num_elements:
            raise ValueError(self.MESSAGE_ELEMENT_IDX)
        
        return self.L[element_index]

# ___________________________________________________________________
# 
#   SET FUNCTIONS
# 
# ___________________________________________________________________    

    def set_inclined_support(self, node_and_angles:dict):
        """
        Add an inclined support to the truss at a specific node.

        Parameters:
        node_and_angles(dict {int:float}) = a dict of shape {node_number: angle_to_set}
        
        """
        for node_index in node_and_angles: 
            if node_index < 0 or node_index >= self.num_original_nodes:
                raise ValueError(self.MESSAGE_NODE_IDX)
            angle = node_and_angles[node_index]
            
            transformation_matrix = np.eye(2 * self.num_original_nodes)
            x = np.deg2rad(angle)  # Convert angle to radians)
            transformation_matrix[2*node_index, 2*node_index] = np.cos(x)
            transformation_matrix[2*node_index, 2*node_index + 1] = np.sin(x)
            transformation_matrix[2*node_index + 1, 2*node_index] = -np.sin(x)
            transformation_matrix[2*node_index + 1, 2*node_index + 1] = np.cos(x)
            self.k_global = transformation_matrix @ self.k_global @ (transformation_matrix.T)
    
    def set_external_constraints(self, constrained_dofs,n_constraints):
        """
        Partition the global stiffness matrix by eliminating the rows and columns corresponding to the constrained degrees of freedom.

        Parameters:
        constrained_dofs (list of int): List of constrained degrees of freedom.

        Returns:
        np.ndarray: Partitioned global stiffness matrix.
        
        """
        self.constrained_dofs=constrained_dofs
        free_dofs = np.setdiff1d(np.arange(n_constraints), constrained_dofs)
        return self.k_global[np.ix_(free_dofs, free_dofs)]
    
    def set_external_constraints_and_forces(self, constrained_dofs,external_forces, element_forces, n_constraints):
        """
        Partition the global stiffness matrix by eliminating the rows and columns corresponding to the constrained degrees of freedom.

        Parameters:
        constrained_dofs (list of int): List of constrained degrees of freedom.

        Returns:
        np.ndarray: Partitioned global stiffness matrix.
        
        """
        # 1. Standardize external_forces into a full global vector
        self.constrained_dofs= constrained_dofs
        free_dof_indices = np.setdiff1d(np.arange(n_constraints), constrained_dofs)
        self.external_forces=self.__reshape_force_vector(external_forces,n_constraints, free_dof_indices)

        #add element componet
        distributed_forces=self.__distribute_forces(element_forces,n_constraints)
        self.external_forces=self.external_forces+distributed_forces
        external_forces=self.external_forces
        # calculate how to get rid of mid nodes
        external_forces=self.reduce_forces(external_forces)

        #gets rid of constrained dofs
        free_dof_indices_reduced=np.setdiff1d(np.arange(len(external_forces)), constrained_dofs)
        f_free = external_forces[free_dof_indices_reduced]
        return self.k_global[np.ix_(free_dof_indices_reduced, free_dof_indices_reduced)], f_free, free_dof_indices_reduced

# ___________________________________________________________________
#
#   HELPER FUNCTIONS
#
# ___________________________________________________________________
    
    #TODO: make it for a generic q, not only constantm should be easy, just need to also integrate it over l
    def __distribute_forces(self,element_forces,n_constraints):
        final_array=np.zeros(n_constraints)
        for i,p in enumerate(element_forces):
            print(i,p, int(i/2))
            nodes_in_elements=self.elements[int(i/2)]
            L=self.L[int(i/2)]
            #calculate Q
            n_nodes = len(nodes_in_elements)
            # Gauss quadrature points and weights
            # 2 points exact for linear, 3 points for quadratic
            
            degree_BtB = 2 * (n_nodes - 2)
            n_gauss = int(np.ceil((degree_BtB + 1) / 2))
            n_gauss = max(n_gauss, 1)  # at least 1 point
            
            points, weights = np.polynomial.legendre.leggauss(n_gauss)
            
            # integrate K_local = p*∫ N(ξ)(L/2) dξ
            N_func = self.shape_functions_array[n_nodes]
            Q=np.zeros(n_nodes)
            for xi, w in zip(points, weights):
                N = N_func(xi)                        # (n_nodes,)
                Q += w * N * (L/2)
            print(Q*p)
            destinations=(2*nodes_in_elements+(i%2))
            final_array[destinations]+=Q*p #Q (shape(3)) * p(scalar)
        print(final_array)
        return final_array
        
    def __calc_k_local(self,element_index):
        L = self.L[element_index]
        n_nodes = len(self.elements[element_index])
        
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

        if self.shape_func == "quadratic":
            # Static condensation: eliminate the internal mid-node (local index 1).
            # Local node ordering is [n1(0), mid(1), n2(2)].
            # Active (end) nodes: indices [0, 2]; internal (mid) node: index [1].
            active = [0, 2]
            internal = [1]

            K_aa = K_local[np.ix_(active, active)]      # 2x2: end nodes
            K_ai = K_local[np.ix_(active, internal)]    # 2x1: end -> mid
            K_ia = K_local[np.ix_(internal, active)]    # 1x2: mid -> end
            K_ii = K_local[np.ix_(internal, internal)]  # 1x1: mid -> mid

            # Condensed local stiffness (2x2): K_cond = K_aa - K_ai * K_ii^{-1} * K_ia
            K_ii_inv = np.linalg.inv(K_ii)
            K_local = K_aa - K_ai @ K_ii_inv @ K_ia
            return K_local,K_aa,K_ai, K_ii_inv, K_ia
        return K_local
    
    def __into_array_if_not(self,E) -> np.ndarray:
        if isinstance(E,(str, float, int)):
            return np.full(self.num_elements,E, dtype=np.float64)
        elif np.issubdtype(E.dtype, str) or np.issubdtype(E.dtype, np.number):
            return np.array(E, dtype=np.float64)
        else:
            raise ValueError("input value is neither a valid number nor an array of numbers")
        
    def __add_mid_node(self):
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
        
    @classmethod
    def __format_nodes(cls,structure):
        #TODO add the importance of ID
        nodes=[]
        constraints=[]
        inclined_support={}
        applied_forces={}
        for i,n in enumerate(structure["nodes"]):
            #position
            if "x" not in n or "y" not in n:
                raise ValueError(f"missing x or y from node number {i}")
            nodes.append((n["x"],n["y"]))
            #constraints
            if "constraints" in n:
                for c in n["constraints"]:
                    #add theta only if euler bernoully
                    if c!="ux" and c!="uy":
                        raise ValueError(f"{c} is not a valid constraint for this type of structure")
                    else:
                        constraints.append(i*2) if c=="ux" else constraints.append(i*2+1)
            #inclined support
            if "inclined_support" in n:
                inclined_support[i]=n["inclined_support"]
            #forces
            if "force" in n:
                    if n["force"][0]!=0:
                        applied_forces[i*2]=n["force"][0]
                    if n["force"][1]!=0:
                        applied_forces[i*2+1]=n["force"][1]
                        
        return nodes, constraints, inclined_support, applied_forces
    
    #TODO fix this, reconsider completely all possibilities to cover all edge cases
    #E.G i add a force on a middle node but i'm using a linear element
    def __reshape_force_vector(self,external_forces, total_dofs, free_dof_indices)-> np.ndarray:
        if isinstance(external_forces, dict):
            print("dict given")
            f_global = np.zeros(total_dofs)
            for dof, val in external_forces.items():
                f_global[dof] = val
            external_forces = f_global
        else:
            print("array given")
            external_forces = np.asarray(external_forces)
            # Backward-compatibility fallback: if input matches number of free DOFs
            if external_forces.ndim == 1 and external_forces.shape[0] == len(free_dof_indices):
                f_global = np.zeros(total_dofs)
                f_global[free_dof_indices] = external_forces
                external_forces = f_global
            elif external_forces.shape[0]!=self.num_nodes*2:
                raise ValueError(f"external_forces length ({external_forces.shape[0]}) must match "
                                 f"total DOFs ({total_dofs}) or free DOFs ({len(free_dof_indices)}).")
        return external_forces
    
# ___________________________________________________________________
#
#   PLOT FUNCTIONS
#
# ___________________________________________________________________

    def plot_plane_truss(self, show_node_indices=True, show_element_indices=True, show_deformed=False, scale_factor=1.0, num_samples=20, norm_type="linear"):
        """
        Function to plot the plane truss:
        - Elements are represented as lines connecting the nodes.
        - If displacements are available, plots the deformed shape by sampling points along the element.
        - Displays the stress along the elements using a symmetrical logarithmic colored gradient.
        - Draws arrows for any non-zero external forces stored in self.external_forces.
        - Draws triangle indicators (^ below, > left) to represent boundary constraints.
    
        Parameters:
        show_node_indices (bool): If True, display the node indices on the plot.
        show_element_indices (bool): If True, display the element indices on the plot.
        show_deformed (bool): If True, plots the deformed shape.
        scale_factor (float): Factor to scale displacements for visualization.
        num_samples (int): Number of points to sample along each element for smooth plotting.
        norm_type (str): "linear" or "log" for the stress colormap normalization.
        """
        from matplotlib.collections import LineCollection
        import matplotlib.colors as mcolors
    
        fig, ax = plt.subplots(figsize=(10, 6))
    
        # Check if truss has been solved
        has_disp = hasattr(self, 'displacements')
    
        xi_vals = np.linspace(-1, 1, num_samples)
        cmap = plt.get_cmap('seismic')
        
        # ── Global structural scale dimensions (used for arrows & boundary symbols) ──
        all_x = self.nodes[:, 0]
        all_y = self.nodes[:, 1]
        bbox_diag = np.hypot(np.ptp(all_x) or 1.0, np.ptp(all_y) or 1.0)
        support_offset = bbox_diag * 0.04  # Distance of the support symbol from the node
    
        if has_disp:
            all_stresses = []
            for i in range(self.num_elements):
                for xi in xi_vals:
                    all_stresses.append(self.get_element_stress(i, xi))

            max_abs_stress = np.max(np.abs(all_stresses))
    
            vmin_stress = -max_abs_stress
            vmax_stress = max_abs_stress
    
            if np.isclose(max_abs_stress, 0):
                max_abs_stress = 1e-6
                vmin_stress = -1e-6
                vmax_stress = 1e-6
    
            lin_threshold = max_abs_stress * 0.01 if max_abs_stress > 0 else 1.0
    
            if norm_type == "log":
                norm = mcolors.SymLogNorm(linthresh=lin_threshold, linscale=1.0,
                                        vmin=vmin_stress, vmax=vmax_stress, base=10)
            else:
                norm = mcolors.Normalize(vmin=vmin_stress, vmax=vmax_stress)
    
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
    
        # ── Loop through each element and plot ──────────────────────────────────
        for i in range(self.num_elements):
            nodes = self.elements[i]
    
            N_vals = np.array([self.shape_functions_array[len(nodes)](xi) for xi in xi_vals])
            orig_positions = N_vals @ self.nodes[nodes]
    
            if show_deformed and has_disp:
                disp = np.array([self.get_displacement_on_element(i, xi) for xi in xi_vals])
                plot_positions = orig_positions + disp * scale_factor
            else:
                plot_positions = orig_positions
    
            if has_disp:
                stresses = np.array([self.get_element_stress(i, xi) for xi in xi_vals])
            else:
                stresses = np.zeros(num_samples)
    
            points = plot_positions.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            seg_stresses = (stresses[:-1] + stresses[1:]) / 2.0
    
            if has_disp:
                lc = LineCollection(segments, cmap=cmap, norm=norm) # type: ignore
                lc.set_array(seg_stresses)
            else:
                lc = LineCollection(segments, colors='blue') # type: ignore
    
            lc.set_linewidth(4)
            ax.add_collection(lc)
    
            if show_element_indices:
                mid_idx = num_samples // 2
                mid_x, mid_y = plot_positions[mid_idx]
                ax.text(mid_x, mid_y, f"E{i}", color='black', fontsize=10, ha='center', va='center',
                        bbox={"facecolor":'white', "edgecolor":'black', "boxstyle":'round,pad=0.2', "alpha":0.8})
    
        # ── Original structure ghost ─────────────────────────────────────────────
        if show_deformed and has_disp:
            for i, e in enumerate(self.elements):
                n1, n2 = e[0], e[-1]
                x1, y1 = self.nodes[n1]
                x2, y2 = self.nodes[n2]
                ax.plot([x1, x2], [y1, y2], color='#5E5E5E', linestyle='--', linewidth=1.5, alpha=0.4,
                        label='Original' if i == 0 else "")
    
        # ── Node markers, labels, and boundary conditions ─────────────────────────
        node_positions = self.nodes.copy()
        if show_deformed and has_disp:
            node_positions += self.displacements_matrix * scale_factor
    
        for i, (nx, ny) in enumerate(node_positions):
            # Base node circle marker
            ax.plot(nx, ny, 'ko', markersize=5, zorder=4)
            
            if show_node_indices:
                ax.text(nx, ny + (0.05 * scale_factor if scale_factor != 1 else 0.05), f"N{i}",
                        color='black', fontsize=11, fontweight='bold', ha='center', va='bottom')
            
            # ── BOUNDARY CONDITION PLOTTING LOGIC ───────────────────────────────
            is_ux_constrained = False
            is_uy_constrained = False
            
            # 1. Fallback Option A: check if a node_constraints dictionary/list attribute exists
            # if hasattr(self, 'constrained_dofs') and self.constrained_dofs is not None:
            #     con = self.constrained_dofs[i]
            #     if 'ux' in con: is_ux_constrained = True
            #     if 'uy' in con: is_uy_constrained = True
            # 2. Fallback Option B: check global global DOF constraint arrays (2*i = ux, 2*i+1 = uy)
            if hasattr(self, 'constrained_dofs') and self.constrained_dofs is not None:
                if 2 * i in self.constrained_dofs: is_ux_constrained = True
                if 2 * i + 1 in self.constrained_dofs: is_uy_constrained = True
            
            # Draw vertical restraint (slides horizontally): '^' under the node
            if is_uy_constrained:
                ax.plot(nx, ny - support_offset, marker='^', color='grey', 
                        markersize=9, markeredgecolor='black', zorder=5)
                
            # Draw horizontal restraint (slides vertically): '>' on the left of the node
            if is_ux_constrained:
                ax.plot(nx - support_offset, ny, marker='>', color='grey', 
                        markersize=9, markeredgecolor='black', zorder=5)
    
        # ── Force arrows ─────────────────────────────────────────────────────────
        if hasattr(self, 'external_forces') and self.external_forces is not None:
            forces = np.array(self.external_forces)
    
            arrow_scale = bbox_diag * 0.08          # arrow length = 8 % of bounding diagonal
            max_force = np.max(np.abs(forces))
            if np.isclose(max_force, 0):
                max_force = 1.0
    
            
            
            for node_idx in range(self.num_nodes):
                dof_x = 2 * node_idx
                dof_y = 2 * node_idx + 1
    
                if dof_x >= len(forces):
                    break
    
                fx = forces[dof_x]
                fy = forces[dof_y] if dof_y < len(forces) else 0.0
    
                if np.isclose(fx, 0) and np.isclose(fy, 0):
                    continue
    
                nx, ny = self.nodes[node_idx]
    
                magnitude = np.hypot(fx, fy)
                length = arrow_scale * (magnitude / max_force)
    
                ux = fx / magnitude
                uy = fy / magnitude
    
                tip_x, tip_y = nx, ny
                tail_x = tip_x - ux * length
                tail_y = tip_y - uy * length
    
                ax.annotate(
                    "",
                    xy=(tip_x, tip_y),           
                    xytext=(tail_x, tail_y),      
                    arrowprops=dict(
                        arrowstyle="->,head_width=0.4,head_length=0.15",
                        color="darkorange",
                        lw=2.0,
                    ),
                )
    
                label = f"{magnitude:.3g} N"
                if fx != 0 and fy != 0:
                    label += f"\n({fx:.3g}, {fy:.3g})"
    
                label_offset_x = -ux * length * 0.25
                label_offset_y = -uy * length * 0.25
                ax.text(
                    tail_x + label_offset_x,
                    tail_y + label_offset_y,
                    label,
                    color="darkorange",
                    fontsize=9,
                    fontweight="bold",
                    ha="center",
                    va="center",
                    bbox=dict(facecolor="white", edgecolor="darkorange",
                            boxstyle="round,pad=0.2", alpha=0.85),
                )
    
        # ── Colorbar ─────────────────────────────────────────────────────────────
        if has_disp:
            cbar = fig.colorbar(sm, ax=ax) # type: ignore
            label_text = "Axial Stress [Pa] (Log Scale)" if norm_type == "log" else "Axial Stress [Pa]"
            cbar.set_label(label_text, rotation=270, labelpad=15)
    
        # ── Formatting ───────────────────────────────────────────────────────────
        title = "Plane Truss Structure (Deformed & Stress State)" if (show_deformed and has_disp) else "Plane Truss Structure"
        ax.set_title(title)
        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")
        ax.axis('equal')
        ax.grid(True, alpha=0.3, linestyle='--')
    
        if show_deformed and has_disp:
            ax.legend(loc='upper right')
    
        plt.tight_layout()
        plt.show()
