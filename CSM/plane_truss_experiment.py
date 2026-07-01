"""Module for solving 2D plane truss problems."""

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.collections import LineCollection
import matplotlib.colors as mcolors
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

    D = {
        2: lambda EA, EI: np.array([[EA]]),
        3: lambda EA, EI: np.array([EA,0],[0,EI])
    }
    shape_functions_matrix = {
        2: lambda x, l: np.array([
            [(1-x)/2,       0,   (x+1)/2,       0],  # u(x) interpolation
            [      0, (1-x)/2,         0, (x+1)/2]   # v(x) interpolation
        ]),
        3: lambda x, l: np.array([
            [0.5*x*(x-1),            0,   (1-x**2),            0,   0.5*x*(x+1),            0],
            [          0, 0.5*x*(x-1),           0,   (1-x**2),             0, 0.5*x*(x+1)]
        ])
    }
        #NOTE derived by hand, maybe there is a way to derive via python
    strain_dispacement_matrix = {
        2: lambda x, l: (2/l) * np.array([[
            -0.5,  0.0,   0.5,  0.0
        ]]),
        3: lambda x, l: (2/l) * np.array([[
            x - 0.5,  0.0,   -2*x,  0.0,   x + 0.5,  0.0
        ]])
    }

    element_type = 2 
# ___________________________________________________________________
#
#  INIT FUNCTIONS
#
# ___________________________________________________________________

    def __init__(self, nodes, elements, elasticity_modulus, cross_sectional_area, bending_modulus=None, shape_func="linear"):
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
        if bending_modulus is not None:
            self.I = self.__into_array_if_not(bending_modulus)
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

        # Store original (structural) node count before mid-nodes are added
        self.num_original_nodes = self.num_nodes

        #create middle points of elements for quadratic shape_func
        print(self.shape_func)
        if self.shape_func != "linear":
            self.__add_mid_node()

        self.num_nodes = len(self.nodes)
        self.nodes_per_elem = len(self.elements[0])

        #make rotation matrices
        self.R = np.empty(self.num_elements, dtype=np.ndarray)
        self.T = np.empty(self.num_elements, dtype=np.ndarray)
        self.L = np.zeros(self.num_elements)
        self.angles = np.zeros(self.num_elements)
        for i, e in enumerate(self.elements):
            x1, y1 = self.nodes[e[0]]
            x2, y2 = self.nodes[e[-1]]
            self.L[i] = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            # Compute the angles of the elements in rad
            self.angles[i] = np.arctan2(y2 - y1, x2 - x1)
            if self.angles[i] < 0:
                self.angles[i] += 2*np.pi

            self.R[i], self.T[i]=self.get_translation_matrix(i)
        

        # After static condensation the mid-nodes are eliminated from the global
        # system, so k_global is sized on the original structural nodes only.
        self.k_global = np.zeros((self.element_type * self.num_original_nodes, self.element_type * self.num_original_nodes))


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
        n_nodes_per_element=len(structure["elements"][0]["nodes"])
        #globals: elastic modulus and cross sectional area should be a list, since i prepare it here
        elastic_mod = np.full(n_elem,structure["defaults"]["E"])
        cross_sec = np.full(n_elem,structure["defaults"]["A"])
        shape_func = structure["defaults"]["shape_func"]

        bending_mod = None
        if "forces" in structure["defaults"]:
            F_elements=np.full(2*n_elem,structure["defaults"]["forces"])
        else:
            F_elements=np.zeros(2*n_elem)
        #force and type of struct tbd
        
        #formatting the nodes
        nodes, constraints, inclined_support, F_nodes = cls.__format_nodes(structure)


        #formatting the elements
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
            if "force" in e and len(e["force"])==2: #an element can only have 2 forces, axial and perpendicular (which get decomposed in also torque)
                F_elements[n_nodes_per_element*i:n_nodes_per_element*i+2]=e["force"]


        #per element's type tbd
        #remaining question: should i directly solve it? answer yes
        truss= PlaneEBBeamProblem(nodes=nodes,elements=elements,elasticity_modulus=elastic_mod,cross_sectional_area=cross_sec,bending_modulus=bending_mod,shape_func=shape_func)
        truss.assemble_global_stiffness()
        
        truss.solve(F_nodes, constraints, element_forces=F_elements, inclined_support=inclined_support)
        return truss
    

    def ElementStiffness(self, element_index):    
        if element_index < 0 or element_index >= self.num_elements:
            raise ValueError(self.MESSAGE_ELEMENT_IDX)
        
        #error correction because cos(1)!=0 in python, it gives a small number but it annoys me
        # if s>c:
        #     a=np.sqrt(1-np.power(s,2))
        #     c=a if a<c else c
        # else:
        #     a=np.sqrt(1-np.power(c,2))
        #     s=a if a<s else s

        if self.shape_func== "linear":
            K_local = self.__calc_k_local(element_index)
        else:
            K_local, _, _, _, _=self.__calc_k_local(element_index)

            # Build a 2-node transformation matrix for the two end nodes only
        T = self.T[element_index][:4,:4]
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
            dof_indices = [self.element_type * n + i for n in e for i in range(self.element_type)]
            for a, da in enumerate(dof_indices):
                for b, db in enumerate(dof_indices):
                    self.k_global[da, db] += k_elem[a, b]
        return self.k_global

#TODO make it generic for N mid nodes
    def _build_force_reduction_matrix(self):
        """
        Assemble the Guyan reduction operator C such that:
            f_reduced = C @ f_full
        
        C starts as [I | 0] (extract active DOFs), then for each element
        the mid-node condensation block is subtracted in:
            C[active_rows, mid_cols] -= factor * I_2x2
        """
        n_active = self.element_type * self.num_original_nodes
        n_total  = self.element_type * self.num_nodes
        C = np.eye(n_active, n_total)   # [I | 0]: identity on active DOFs, zero on mid-node DOFs

        for e_idx in range(self.num_elements):
            _, _, K_ai, K_ii_inv, _ = self.__calc_k_local(e_idx)
            condenser = K_ai @ K_ii_inv  # (4,1): Guyan allocation matrix

            n1, mid, n2 = self.elements[e_idx, [0, 1, -1]]
            f_n1 = condenser[0, 0]
            f_n2 = condenser[2, 0]

            # Subtract the 2x2 identity block scaled by each factor.
            # This distributes both x and y mid-node force components equally,
            # consistent with the isotropic axial-only condensation.
            C[np.ix_([2*n1,   2*n1+1], [2*mid, 2*mid+1])] -= f_n1 * np.eye(2)
            C[np.ix_([2*n2,   2*n2+1], [2*mid, 2*mid+1])] -= f_n2 * np.eye(2)

        return C

    def reduce_forces(self, external_forces):
        if self.shape_func == "linear":
            return external_forces[:self.element_type * self.num_original_nodes]

        if not hasattr(self, '_C_reduction'):
            self._C_reduction = self._build_force_reduction_matrix()

        return self._C_reduction @ external_forces


    def solve(self, external_forces, constrained_dofs, element_forces=[], inclined_support={}):
        """
        Solve the truss problem for the given external forces and constraints.

        Parameters:
        external_forces (np.ndarray or dict): External forces/moments applied to the nodes/DOFs.
        constrained_dofs (list of int): List of constrained degrees of freedom.
        element_forces (list): Optional distributed or element-level forces.
        inclined_support (dict): Optional dictionary defining inclined support angles.

        Returns:
        np.ndarray: Global displacement vector (length 3 * num_nodes).
        """
        # 1. Handle inclined supports if provided
        if inclined_support is not None and len(inclined_support) > 0:
            self.set_inclined_support(inclined_support)
            
        # 2. Define total degrees of freedom (3 DOFs per node: u, v, theta)
        total_dofs = self.element_type * self.num_nodes

        # 3. Partition the global stiffness matrix and force vector
        # Ensure your set_external_constraints_and_forces method handles total_dofs correctly!
        k_free, f_free, free_dof_indices = self.set_external_constraints_and_forces(constrained_dofs, external_forces, element_forces, total_dofs)
        # 4. Solve for displacements at unconstrained (free) DOFs

        free_displacements = np.linalg.solve(k_free, f_free)
        # 5. Reconstruct the full global displacements vector
        displacements = np.zeros(total_dofs)
        displacements[free_dof_indices] = free_displacements
        self.displacements_matrix=displacements.reshape(-1, self.element_type)
        if self.shape_func == "quadratic":
            for element_index in range(self.num_elements):
                nodes = self.elements[element_index]
                #TODO MAKE IT GENERIC FOR N NODES IN THE MIDDLE
                n1, mid, n2 = nodes[0], nodes[1], nodes[-1]
                
                # 1. Get full 4-element global displacements for end nodes
                destinations_active = [self.element_type * n + dof for n in [n1, n2] for dof in range(self.element_type)]
                d_global_active = displacements[destinations_active]
                
                # 2. Transform active displacements to local frame

                T= self.T[element_index][:4,:4]
                R = self.R[element_index]
                d_local_active = T @ d_global_active

                # 3. Rebuild K_local parts
                _, _, _, K_ii_inv, K_ia = self.__calc_k_local(element_index)

                # 4. Transform mid-node global forces to local frame (2 elements)
                f_mid_global = self.external_forces[self.element_type*mid:self.element_type*(mid+1)]
                f_i_local = R @ f_mid_global  # Shape (2,) -> [local_axial, local_transverse]

                # 5. Correct back-substitution (yields local axial displacement)
                f_axial_local = np.array([f_i_local[0]])  # Isolate only the local axial force component
                u_axial_local = K_ii_inv @ (f_axial_local - K_ia @ d_local_active)  # Shape (1,)
                
                # The local transverse displacement is simply the average of the end-nodes' transverse DOFs
                # d_local_active indices: [u1, v1, u2, v2] -> v1 is index 1, v2 is index 3
                v_transverse_local = 0.5 * (d_local_active[1] + d_local_active[3])
                
                # Combine back into a full 2D local mid-node displacement vector
                u_mid_local = np.array([u_axial_local[0], v_transverse_local])  # Shape (2,)

                # 6. Transform local mid-node displacements back to global frame
                d_mid_global = R.T @ u_mid_local  # (2,2) @ (2,) -> Shape (2,)

                displacements[self.element_type*mid:self.element_type*(mid+1)] = d_mid_global
        self.displacements_matrix=displacements.reshape(-1, self.element_type)
        self.displacements = displacements
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
        #2 because there are only the end nodes here, so is not representing the dofs
        return self.displacements[self.element_type*node_index], self.displacements[self.element_type*node_index + 1]
    

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
        
        L = self.L[element_index]
        nodes = self.elements[element_index]
        n_nodes = len(nodes)

        destinations= [self.element_type*n + dof for n in nodes for dof in range(self.element_type)]
        displacements = self.displacements[destinations]
        
        N = self.shape_functions_matrix[n_nodes](x, L)
        
        return self.R[element_index][:2, :2].T @ N @ self.T[element_index] @ displacements #global to local to global

    def get_reaction_forces(self):
        """
        Calculate the reaction forces.

        Returns:
        np.ndarray: Reaction forces.
        
        """
        reaction_forces = self.k_global @ self.displacements[:self.element_type*self.num_original_nodes]
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
        nodes = self.elements[element_index]
        n_nodes = len(nodes)
        destinations= [self.element_type*n + dof for n in nodes for dof in range(self.element_type)]
        d_local = self.T[element_index] @ self.displacements[destinations]

        B = self.strain_dispacement_matrix[n_nodes](xi, L)
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

    def get_position_over_element(self, element_index, xi):
        L = self.L[element_index]
        nodes = self.elements[element_index]
        n_nodes = len(nodes)

        #needed to expand the position and calculate the N of the end nodes, obviously they have 0 angle change at the beginning
        nodes_with_angles = np.zeros((n_nodes, self.element_type))
        for i, n in enumerate(nodes):
            nodes_with_angles[i, :2] = self.nodes[n]   # row per node, not column
        nodes_with_angles = nodes_with_angles.flatten()

        N = self.shape_functions_matrix[n_nodes](xi,L)


        return self.R[element_index][:2, :2].T @ N @ self.T[element_index] @ nodes_with_angles #return global position over of a point over the element

    def get_position_displaced(self,element_index, xi,k):
        return self.get_position_over_element(element_index,xi) + self.get_displacement_on_element(element_index,xi) * k
    
    def get_translation_matrix(self, element_index):
        theta = self.angles[element_index]
        c, s = np.cos(theta), np.sin(theta)
        element_type=self.element_type
        nodes_per_elem=self.nodes_per_elem
        T = np.zeros((element_type*nodes_per_elem,element_type*nodes_per_elem))
        R = np.array([[ c, s, 0],
                      [-s, c, 0],
                      [ 0, 0, 1]])
        for i in range(self.nodes_per_elem):
            T[element_type*i:element_type*(i+1), element_type*i:element_type*(i+1)] = R[:element_type, :element_type]
        return R[:element_type, :element_type], T

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
            
            transformation_matrix = np.eye(self.element_type * self.num_original_nodes)
            x = np.deg2rad(angle)  # Convert angle to radians)
            transformation_matrix[self.element_type*node_index, self.element_type*node_index] = np.cos(x)
            transformation_matrix[self.element_type*node_index, self.element_type*node_index + 1] = np.sin(x)
            transformation_matrix[self.element_type*node_index + 1, self.element_type*node_index] = -np.sin(x)
            transformation_matrix[self.element_type*node_index + 1, self.element_type*node_index + 1] = np.cos(x)
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
    def __distribute_forces(self, element_forces, n_constraints):
        final_array = np.zeros(n_constraints)
        for i, p in enumerate(element_forces.reshape(-1, 2)):
            # Skip unassigned or zero forces to maximize performance
            if np.allclose(p, 0.0):
                continue
                
            nodes_in_elements = self.elements[i]
            destinations= [self.element_type*n + dof for n in nodes_in_elements for dof in range(self.element_type)]
            L = self.L[i]
            nodes = self.elements[i]
            n_nodes = len(nodes)
            
            # 3 Gauss points are exact for polynomial shape function load integration
            degree_BtB = 2 * (n_nodes - 2)
            n_gauss = int(np.ceil((degree_BtB + 1) / 2))
            n_gauss = max(n_gauss, 1)  # at least 1 point
            points, weights = np.polynomial.legendre.leggauss(n_gauss)
            
            # Initialize local element equivalent nodal force vector (3 DOFs per node)
            Q = np.zeros((self.element_type*n_nodes,len(p)))
            N_func = self.shape_functions_matrix[n_nodes]

            for xi, w in zip(points, weights):
                N = N_func(xi, L)                        # (n_nodes,)
                Q += w * self.T[i].T @ N.T @ self.R[i][:2, :2] * (L / 2)
            final_array[destinations]+= Q @ p #p(2) -> final_array (6)
        return final_array
        
    def __calc_k_local(self,element_index):
        L = self.L[element_index]
        n_nodes = self.nodes_per_elem
        
        # Use 3 Gauss points for exact polynomial integration
        degree_BtB = 2 * (n_nodes - 2)
        n_gauss = int(np.ceil((degree_BtB + 1) / 2))
        n_gauss = max(n_gauss, 1)  # at least 1 point

        points, weights = np.polynomial.legendre.leggauss(n_gauss)
        
        EA = self.A[element_index] * self.E[element_index]
        EI=0
        if self.element_type == 3:# if euler bernoulli
            EI = self.I[element_index] * self.E[element_index]
        
        # Material property matrix (2x2)
        D =self.D[self.element_type](EA,EI)

        # Integration Loop over Gauss points
        total_dofs = self.element_type * n_nodes
        K_local = np.zeros((total_dofs, total_dofs))
        for xi, w in zip(points, weights):
            B = self.strain_dispacement_matrix[n_nodes](xi,L)
            # Perform matrix multiplication and accumulate with Jacobian scaling (L/2)
            K_local += w * (B.T @ D @ B) * (L / 2)
        print(K_local)
        if self.shape_func == "quadratic":
            # Static condensation: eliminate the internal mid-node (local index 1).
            # Local node ordering is [n1(0), mid(1), n2(2)].
            # Active (end) nodes: indices [0,1 , 4,5]; internal (mid) node: index [2], only axial forces are considered in the truss, the inclusion of [3] would cause singular matrix.
            active = [0, 1, 4, 5]
            # Internal condensed DOF: Node 1 Axial only (2)
            internal = [2]

            K_aa = K_local[np.ix_(active, active)]      # 4x4
            K_ai = K_local[np.ix_(active, internal)]    # 4x1
            K_ia = K_local[np.ix_(internal, active)]    # 1x4
            K_ii = K_local[np.ix_(internal, internal)]  # 1x1

            K_ii_inv = np.linalg.inv(K_ii)
            K_local = K_aa - K_ai @ K_ii_inv @ K_ia
            return K_local, K_aa, K_ai, K_ii_inv, K_ia
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
                    if c!="ux" and c!="uy" and c!="theta":
                        raise ValueError(f"{c} is not a valid constraint for this type of structure")
                    elif c=="ux":
                        constraints.append(i*cls.element_type)
                    elif c=="uy":
                        constraints.append(i*cls.element_type+1)
                    elif c!="theta" and cls.element_type==3:
                        constraints.append(i*cls.element_type+2)
            #inclined support
            if "inclined_support" in n:
                inclined_support[i]=n["inclined_support"]
            #forces
            if "force" in n:
                    for j,f in enumerate(n["force"]): 
                        if f!=0:
                            applied_forces[i*cls.element_type+j]=f
        return nodes, constraints, inclined_support, applied_forces
    
    #TODO fix this, reconsider completely all possibilities to cover all edge cases
    #E.G i add a force on a middle node but i'm using a linear element
    def __reshape_force_vector(self,external_forces, total_dofs, free_dof_indices)-> np.ndarray:
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
            elif external_forces.shape[0]!=self.num_nodes*self.element_type:
                raise ValueError(f"external_forces length ({external_forces.shape[0]}) must match "
                                 f"total DOFs ({total_dofs}) or free DOFs ({len(free_dof_indices)}).")
        return external_forces
    
# ___________________________________________________________________
#
#   PLOT FUNCTIONS
#
# ___________________________________________________________________

    def plot_plane_truss(self, show_node_indices=True, show_element_indices=True, show_deformed=False, scale_factor=1.0, num_samples=20, stress_to_plot="axial", norm_type="linear"):
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
        node_label_offset = bbox_diag * 0.03  # Robust static offset for node labels
    
        if has_disp:
            all_stresses = []
            for i in range(self.num_elements):
                L_i = self.L[i]
                for xi in xi_vals:
                    x_local = (xi + 1) * L_i / 2
                    stress_vec = self.get_element_stress(i, x_local)
                    
                    if stress_to_plot == "axial":
                        all_stresses.append(stress_vec[0])

                    elif stress_to_plot == "bending":
                        all_stresses.append(stress_vec[1])
                    else:
                        all_stresses.extend(stress_vec.tolist())
                        
            max_abs_stress = np.max(np.abs(all_stresses)) if len(all_stresses) > 0 else 0.0
    
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
            if show_deformed and has_disp:
                plot_positions = np.array([self.get_position_displaced(i, xi, scale_factor) for xi in xi_vals])
            else:
                plot_positions = np.array([self.get_position_over_element(i, xi) for xi in xi_vals])
    
            if has_disp and stress_to_plot=="axial":
                stresses = np.array([self.get_element_stress(i, xi)[0] for xi in xi_vals])  # axial only
            elif has_disp and stress_to_plot=="bending":
                stresses = np.array([self.get_element_stress(i, xi)[1] for xi in xi_vals])  # bending only
            else:
                stresses = np.zeros(num_samples)
    
            points = plot_positions.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            seg_stresses = (stresses[:-1] + stresses[1:]) / 2.0
    
            # Always draw a faint gray background line underneath the colored segments
            # This handles low/zero stress visibility issues beautifully
            ax.plot(plot_positions[:, 0], plot_positions[:, 1], color='lightgrey', linewidth=4, alpha=1, zorder=2)
    
            if has_disp:
                lc = LineCollection(segments, cmap=cmap, norm=norm, alpha=1, zorder=3) # type: ignore
                lc.set_array(seg_stresses)
            else:
                lc = LineCollection(segments, colors='blue', zorder=3) # type: ignore
    
            lc.set_linewidth(4)
            ax.add_collection(lc)
    
            if show_element_indices:
                mid_idx = num_samples // 2
                mid_x, mid_y = plot_positions[mid_idx]
                ax.text(mid_x, mid_y, f"E{i}", color='black', fontsize=10, ha='center', va='center',
                        bbox={"facecolor":'white', "edgecolor":'black', "boxstyle":'round,pad=0.2', "alpha":0.8}, zorder=6)
    
        # ── Original structure ghost (Undeformed) ─────────────────────────────────
        if show_deformed and has_disp:
            for i, e in enumerate(self.elements):
                n1, n2 = e[0], e[-1]
                x1, y1 = self.nodes[n1]
                x2, y2 = self.nodes[n2]
                ax.plot([x1, x2], [y1, y2], color='#5E5E5E', linestyle='--', linewidth=1.5, alpha=0.4,
                        label='Original' if i == 0 else "", zorder=1)
    
        # ── Node markers, labels, and boundary conditions ─────────────────────────
        node_positions = self.nodes.copy()
        if show_deformed and has_disp:
            node_positions += self.displacements_matrix[:,:2] * scale_factor
    
        for i, (nx, ny) in enumerate(node_positions):
            # Base node circle marker
            ax.plot(nx, ny, 'ko', markersize=5, zorder=4)
            
            # FIX: Used global robust offset bound instead of scale_factor product
            if show_node_indices:
                ax.text(nx, ny + node_label_offset, f"N{i}",
                        color='black', fontsize=11, fontweight='bold', ha='center', va='bottom', zorder=5)
            
            # ── BOUNDARY CONDITION PLOTTING LOGIC ───────────────────────────────
            is_ux_constrained = False
            is_uy_constrained = False
            
            if hasattr(self, 'constrained_dofs') and self.constrained_dofs is not None:
                if self.element_type * i in self.constrained_dofs: is_ux_constrained = True
                if self.element_type * i + 1 in self.constrained_dofs: is_uy_constrained = True
            
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
                dof_x = self.element_type * node_idx
                dof_y = self.element_type * node_idx + 1
    
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
                    arrowprops={
                        "arrowstyle":"->,head_width=0.4,head_length=0.15",
                        "color":"darkorange",
                        "lw":2.0,
                    },
                    zorder=5
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
                    bbox={"facecolor":"white", "edgecolor":"darkorange",
                            "boxstyle":"round,pad=0.2", "alpha":0.85},
                    zorder=6
                )
    
        # ── Colorbar ─────────────────────────────────────────────────────────────
        if has_disp:
            cbar = fig.colorbar(sm, ax=ax) # type: ignore
            label_text = f"{stress_to_plot.capitalize()} Stress [Pa] (Log Scale)" if norm_type == "log" else f"{stress_to_plot.capitalize()} Stress [Pa]"
            cbar.set_label(label_text, rotation=270, labelpad=15)
    
        # ── Formatting ───────────────────────────────────────────────────────────
        title = f"Plane Euler Bernoulli Beam Structure (Deformed & Stress {stress_to_plot} State)" if (show_deformed and has_disp) else "Plane Truss Structure"
        ax.set_title(title)
        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")
        ax.axis('equal')
        ax.grid(True, alpha=0.3, linestyle='--')
    
        if show_deformed and has_disp:
            ax.legend(loc='upper right')
    
        plt.tight_layout()
        plt.show()