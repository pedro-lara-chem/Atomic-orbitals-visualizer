import numpy as np
import numba
import pyvista as pv
import os
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap


def distance_matrix(coordinates, n_atoms):
    """
    Calculates the distance matrix between atoms in a molecule.

    This function computes the pairwise distances between all atoms
    given their coordinates. If the distance between two atoms is greater
    than a threshold (3 Bohr), it is set to 0, effectively ignoring
    the bond between them.

    Args:
        coordinates (numpy.ndarray): A 2D array of atom coordinates.
                                   Shape: (n_atoms, 3) where each row
                                   represents the (x, y, z) coordinates
                                   of an atom.
        n_atoms (int): The number of atoms in the molecule.

    Returns:
        numpy.ndarray: A 2D array representing the distance matrix.
                       Shape: (n_atoms, n_atoms). The diagonal elements
                       are zero, and off-diagonal elements (i, j) and (j, i)
                       represent the distance between atom i and atom j,
                       or 0 if the distance is above the threshold.
    """
    dist_mat = np.zeros((n_atoms, n_atoms))  # Initialize distance matrix with zeros
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):  # Loop through all unique pairs of atoms
            x = coordinates[j] - coordinates[i]  # Vector between atoms i and j
            dist = np.linalg.norm(x)  # Euclidean distance
            if dist > 3:  # Check against distance threshold (Bohr)
                dist = 0  # Set distance to 0 if above threshold

            dist_mat[i][j] = dist  # Store the distance
            dist_mat[j][i] = dist  # Distance matrix is symmetric
    return dist_mat

def parse_molden_file(filepath):
    """
    Parses a Molden file to extract atom information, GTO basis sets, and MO coefficients.

    Args:
        filepath (str): The path to the Molden file.

    Returns:
        tuple: A tuple containing three lists:
            - atoms_data (list): List of dictionaries, each representing an atom.
                                 Example: {'label': 'C', 'atomic_number': 6,
                                           'x': 0.0, 'y': 0.0, 'z': 0.0, 'unit': 'AU',
                                           'number_in_molden': 1}
            - gto_data (list): List of dictionaries, each representing the GTOs for an atom.
                               Example: {'atom_index': 0,  # 0-based index matching atoms_data
                                         'shells': [{'type': 's', 'scale_factor': 1.0,
                                                     'primitives': [{'exponent': 71.616837, 'coefficients': [0.154329]}]
                                                   }]
                                        }
            - mo_data (list): List of dictionaries, each representing a molecular orbital.
                              Example: {'symmetry': 'A1', 'energy': -10.25, 'spin': 'Alpha',
                                        'occupancy': 2.0,
                                        'coefficients': [(1, 0.99), (2, 0.05), ...]}
    """
    atoms_data = []
    gto_data = []
    mo_data = []

    current_section = None
    current_atom_gto_shells = []
    current_atom_gto_idx = -1 # 0-based index for GTO atom tracking
    current_mo_coeffs = []
    current_mo_details = {}
    atom_units = "AU" # Default, can be overridden by [Atoms] AU or [Atoms] Angs

    try:
        # It's often useful to have the file iterator available for `next(f)`
        # so we open it here and pass `f_iter` to the loop.
        with open(filepath, 'r') as f_iter:
            for line_number, raw_line in enumerate(f_iter, 1): # Add line number for better error reporting
                line = raw_line.strip()

                if not line: 
                    continue

                # Section headers
                if line.startswith('['):
                    # Finalize MO data when a new section starts
                    if current_section == "MO" and current_mo_details:
                        if current_mo_coeffs: 
                            current_mo_details['coefficients'] = list(current_mo_coeffs) # Use list() for copy
                            mo_data.append(current_mo_details)
                        current_mo_coeffs = []
                        current_mo_details = {}
                    
                    # Determine new section
                    if line.lower().startswith("[atoms]"):
                        # Finalize GTO if switching from GTO to Atoms (unlikely, but for safety)
                        if current_section == "GTO" and current_atom_gto_shells and current_atom_gto_idx != -1:
                            gto_data.append({'atom_index': current_atom_gto_idx, 'shells': list(current_atom_gto_shells)})
                            current_atom_gto_shells = []
                            current_atom_gto_idx = -1
                        current_section = "Atoms"
                        if "angs" in line.lower(): # Check if units are Angstroms
                            atom_units = "Angs"
                        else: # Default to Atomic Units
                            atom_units = "AU" 
                    elif line.lower().startswith("[gto]"): # Use startswith for "[GTO] (AU)"
                        # If switching from another section to GTO, no specific finalization needed here for GTO itself
                        current_section = "GTO"
                        current_atom_gto_idx = -1 # Reset for the first atom in this GTO block
                        current_atom_gto_shells = [] # Ensure it's fresh
                    elif line.lower().startswith("[mo]"):
                        # Finalize any pending GTO data for the last atom in GTO section
                        if current_section == "GTO" and current_atom_gto_shells and current_atom_gto_idx != -1:
                            gto_data.append({
                                'atom_index': current_atom_gto_idx,
                                'shells': list(current_atom_gto_shells) # Use list() for copy
                            })
                            current_atom_gto_shells = [] 
                            current_atom_gto_idx = -1 
                        current_section = "MO"
                    else: # Any other section (e.g., [5D], [Title], [N_ATOMS], [CHARGE])
                        # If we were in GTO section and now encounter something else (not MO), finalize GTO.
                        if current_section == "GTO" and current_atom_gto_shells and current_atom_gto_idx != -1:
                            gto_data.append({'atom_index': current_atom_gto_idx, 'shells': list(current_atom_gto_shells)})
                            current_atom_gto_shells = []
                            current_atom_gto_idx = -1
                        current_section = line # Store the name of the section (e.g., "[5D]")
                    continue

                # Parsing data based on current section
                if current_section == "Atoms":
                    parts = line.split()
                    if len(parts) == 6: # label, num, atomic_num, x, y, z
                        try:
                            atoms_data.append({
                                'label': parts[0],
                                'number_in_molden': int(parts[1]),
                                'atomic_number': int(parts[2]),
                                'x': float(parts[3]),
                                'y': float(parts[4]),
                                'z': float(parts[5]),
                                'unit': atom_units
                            })
                        except ValueError:
                            print(f"Warning (line {line_number}): Could not parse atom line in {filepath}: {line}")
                    else:
                        print(f"Warning (line {line_number}): Malformed atom line in {filepath}: {line}")

                elif current_section == "GTO":
                    parts = line.split()
                    # Check if it's an atom index line for GTO (e.g., "1 0" or just "1")
                    if len(parts) <= 2 and all(p.isdigit() for p in parts) and parts: # Ensure parts is not empty
                        # Finalize GTOs for the PREVIOUS atom if one was being processed
                        if current_atom_gto_shells and current_atom_gto_idx != -1 :
                            gto_data.append({
                                'atom_index': current_atom_gto_idx,
                                'shells': list(current_atom_gto_shells) # Use list() for copy
                            })
                        
                        atom_seq_num_gto = int(parts[0]) 
                        found_atom_idx = -1
                        for idx, atom_entry in enumerate(atoms_data):
                            if atom_entry['number_in_molden'] == atom_seq_num_gto:
                                found_atom_idx = idx
                                break
                        
                        if found_atom_idx != -1:
                            current_atom_gto_idx = found_atom_idx
                        # If the atom number from GTO section isn't found in the parsed atoms_data (e.g., due to ordering or partial file),
                        # fall back to using the GTO's atom sequence number directly, assuming 1-based to 0-based conversion.
                        else:
                            print(f"Warning (line {line_number}): GTO section in {filepath} refers to atom sequence number {atom_seq_num_gto} not found in [Atoms] section. Using raw sequence number '{atom_seq_num_gto-1}' as fallback index.")
                            current_atom_gto_idx = atom_seq_num_gto -1 


                        current_atom_gto_shells = [] # Start new list for this new atom
                        
                    # Check if it's a shell type definition line
                    # Format 1: type num_primitives scale_factor (e.g., "s  3  1.00")
                    # Format 2: type num_primitives (e.g., "s  9", scale_factor defaults to 1.0)
                    elif parts and parts[0].isalpha() and parts[0].lower() in ['s', 'p', 'sp', 'd', 'f', 'g', 'h', 'i']:
                        shell_type = parts[0].lower()
                        num_primitives = 0
                        scale_factor = 1.0  # Default scale factor

                        if len(parts) == 3: # type num_primitives scale_factor
                            try:
                                num_primitives = int(parts[1])
                                scale_factor = float(parts[2])
                            except ValueError:
                                print(f"Warning (line {line_number}): Could not parse shell definition (type N scale): '{line}' in {filepath}")
                                continue
                        elif len(parts) == 2: # type num_primitives
                            try:
                                num_primitives = int(parts[1])
                                # scale_factor remains 1.0 (default)
                            except ValueError:
                                print(f"Warning (line {line_number}): Could not parse shell definition (type N): '{line}' in {filepath}")
                                continue
                        else: # Malformed shell definition line
                            print(f"Warning (line {line_number}): Malformed shell definition line: '{line}' in {filepath}")
                            continue
                        
                        current_shell_primitives = []
                        if num_primitives > 0: 
                            for i_prim in range(num_primitives):
                                try:
                                    # Read next line from file iterator f_iter, not f
                                    primitive_line = next(f_iter).strip() 
                                except StopIteration:
                                    print(f"Warning (line {line_number}): Unexpected end of file while reading primitive {i_prim+1}/{num_primitives} for shell {shell_type} in {filepath}.")
                                    break 
                                prim_parts = primitive_line.split()
                                try:
                                    exponent = float(prim_parts[0])
                                    coefficients = [float(c) for c in prim_parts[1:]]
                                    current_shell_primitives.append({
                                        'exponent': exponent,
                                        'coefficients': coefficients
                                    })
                                except (ValueError, IndexError) as e:
                                    # line_number refers to the shell definition line, primitive is on subsequent lines
                                    print(f"Warning (near line {line_number+i_prim+1}): Could not parse GTO primitive line in {filepath}: '{primitive_line}'. Error: {e}")
                                    continue 
                        
                        if current_shell_primitives or num_primitives == 0: 
                            current_atom_gto_shells.append({
                                'type': shell_type,
                                'scale_factor': scale_factor,
                                'primitives': current_shell_primitives
                            })
                    elif line: # Non-empty line that doesn't match atom index or shell type
                        print(f"Warning (line {line_number}): Unrecognized line in GTO section of {filepath}: {line}")


                elif current_section == "MO":
                    if line.lower().startswith("sym="): 
                        if current_mo_details: 
                            if current_mo_coeffs: 
                                current_mo_details['coefficients'] = list(current_mo_coeffs) # Use list()
                                mo_data.append(current_mo_details)
                        
                        current_mo_details = {'symmetry': line.split('=')[1].strip()}
                        current_mo_coeffs = []
                    elif line.lower().startswith("ene="):
                        try:
                            current_mo_details['energy'] = float(line.split('=')[1].strip())
                        except (ValueError, KeyError): 
                             print(f"Warning (line {line_number}): Could not parse MO energy in {filepath}: {line}")
                    elif line.lower().startswith("spin="):
                        current_mo_details['spin'] = line.split('=')[1].strip()
                    elif line.lower().startswith("occup="):
                        try:
                            current_mo_details['occupancy'] = float(line.split('=')[1].strip())
                        except (ValueError, KeyError):
                            print(f"Warning (line {line_number}): Could not parse MO occupancy in {filepath}: {line}")
                    else: 
                        parts = line.split()
                        if len(parts) == 2:
                            try:
                                mo_coeff_val = float(parts[1])
                                current_mo_coeffs.append((mo_coeff_val))
                            except ValueError:
                                print(f"Warning (line {line_number}): Could not parse MO coefficient line in {filepath}: {line}")
                        elif line : 
                             print(f"Warning (line {line_number}): Unrecognized line in MO section of {filepath}: {line}")


            # After loop, finalize any pending GTO or MO data
            if current_section == "GTO" and current_atom_gto_shells and current_atom_gto_idx != -1:
                gto_data.append({
                    'atom_index': current_atom_gto_idx,
                    'shells': list(current_atom_gto_shells) # Use list()
                })
            if current_section == "MO" and current_mo_details and current_mo_coeffs: 
                current_mo_details['coefficients'] = list(current_mo_coeffs) # Use list()
                mo_data.append(current_mo_details)

    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return [], [], []
    except Exception as e:
        print(f"An unexpected error occurred while parsing {filepath}: {e}")
        import traceback
        traceback.print_exc() # Print full traceback for unexpected errors
        return [], [], []
        
    return atoms_data, gto_data, mo_data


def parse_directory_for_molden_files(directory_path):
    """
    Scans a directory for Molden files (ending with .molden or containing .molden)
    and parses each one.

    Args:
        directory_path (str): The path to the directory to scan.
    """
    found_molden_files = []
    if not os.path.isdir(directory_path):
        print(f"Error: Provided path '{directory_path}' is not a directory or does not exist.")
        return

    for filename in os.listdir(directory_path):
        # Flexible check for Molden files
        if "molden" in filename: # Handles cases like file.molden or file.type.molden
            if not filename.endswith((".zip", ".tar", ".gz", ".bz2")): # Avoid common archive extensions
                full_filepath = os.path.join(directory_path, filename)
                if os.path.isfile(full_filepath): # Ensure it's a file
                    found_molden_files.append(full_filepath)

    if not found_molden_files:
        print(f"No Molden files found in directory: {directory_path}")
        return

    print(f"\nFound {len(found_molden_files)} Molden file(s) in '{directory_path}'. Parsing...\n")
    return found_molden_files


### ------------------------------
### Main AO Evaluation Function
### ------------------------------
# --- Constants (Dependencies for compute_atomic_orbitals) ---
L_QUANTUM_NUMBERS_MAP = {'s': 0, 'p': 1, 'd': 2, 'f': 3, 'g': 4}
ANGULAR_LABELS = {
    0: ['s'],
    1: ['px', 'py', 'pz'],
    2: ['dxy', 'dyz', 'dz2', 'dxz', 'dx2y2'],
    3: ['fz3', 'fxz2', 'fyz2', 'fz(x2-y2)', 'fxyz', 'fx(x2-3y2)', 'fy(3x2-y2)']
}

# --- Numba JIT-compiled Helper Functions ---

@numba.njit(cache=True) # Added cache=True for faster subsequent runs
def numba_factorial2(n_val):
    """
    Numba-compatible version of double factorial (n!!).
    factorial2(-1) = 1
    factorial2(0) = 1
    factorial2(n) = n * (n-2) * ...
    For n < -1, behavior in scipy.special.factorial2 is 0.
    """
    if n_val < -1:
        return 0.0
    if n_val == -1 or n_val == 0: # factorial2(-1)=1, factorial2(0)=1
        return 1.0
    # For n=1, factorial2(1)=1, loop condition val > 0.5 handles this
    
    res = 1.0
    val = float(n_val) # Start with n
    while val > 0.5: # Handles both even (e.g. 2*0=0) and odd (e.g. 2*0+1=1)
        res *= val
        val -= 2.0
    return res

@numba.njit(cache=True)
def norm_primitive_numba(alpha, l_val):
    """
    Computes the normalization constant for a primitive Gaussian Type Orbital (GTO)
    in solid harmonic form, optimized with Numba.
    Assumes l_val >= 0.
    """
    if alpha < 1e-12: # Avoid issues with very small or zero alpha
        return 0.0

    # Calculate (2*l-1)!!
    # For l=0, (2*l-1) = -1, factorial2(-1) = 1.
    # For l > 0, (2*l-1) is a positive odd integer.
    double_fact_val = numba_factorial2(2 * l_val - 1)

    if abs(double_fact_val) < 1e-12: # Should not happen for valid l_val >= 0
        # This case implies an issue, e.g., if numba_factorial2 returned 0 unexpectedly.
        # For l_val=0, double_fact_val is 1.0.
        # For l_val >0, 2*l_val-1 is odd and >=1, so factorial2 should be >=1.
        # If it's near zero, it's problematic for division.
        return 0.0 # Or raise an error / handle as appropriate

    # (2 * alpha / np.pi)**(0.75)
    term1 = (2.0 * alpha / np.pi)**(0.75)
    
    # (4 * alpha)**(l_val / 2.0) can be written as (2 * sqrt(alpha))**l_val
    # This form might be slightly more stable or explicit.
    term2 = (2.0 * np.sqrt(alpha))**l_val
    
    norm_const = term1 * term2 / np.sqrt(double_fact_val) # abs() not strictly needed if double_fact_val is always >0
    return norm_const

@numba.njit(cache=True)
def compute_radial_part_numba(r_sq_grid_points, # 1D array of r^2 values for all grid points
                              exponents_arr,      # 1D array of primitive exponents for the shell
                              coeffs_arr,         # 1D array of contraction coeffs for this l-component
                              scale_factor_sq,    # shell scale_factor squared
                              l_val):             # angular momentum
    """
    Computes the summed contracted radial part of an AO using Numba.
    R_nl(r) = Sum_k d_k * N_k_prim * exp(-alpha_scaled_k * r^2)
    (The r^l term is applied later, this function returns the sum of Gaussians part)
    """
    num_grid_points = r_sq_grid_points.shape[0]
    sum_contracted_radial_part = np.zeros(num_grid_points, dtype=np.float64)

    for i in range(exponents_arr.shape[0]):
        exponent = exponents_arr[i]
        contraction_coeff = coeffs_arr[i]

        if abs(contraction_coeff) < 1e-15: # Skip if coefficient is negligible
            continue

        alpha_scaled = exponent * scale_factor_sq
        
        N_k_prim = norm_primitive_numba(alpha_scaled, l_val)
        if abs(N_k_prim) < 1e-15: # If normalization is effectively zero, skip
            continue
            
        exp_term = np.exp(-alpha_scaled * r_sq_grid_points) # Radial gaussian term for all grid points
        
        sum_contracted_radial_part += contraction_coeff * N_k_prim * exp_term
    
    return sum_contracted_radial_part

@numba.njit(cache=True) # Apply Numba to this function as well
def real_sph_harmonics_pyscf_order_numba(l, theta, phi):
    """
    Returns UNNORMALIZED angular components S_lm(theta, phi) for real solid harmonics.
    Order for l=1 is px, py, pz.
    Order for l=2 is dxy, dyz, dz2, dxz, dx2y2.
    Order for l=3 is fz3, fxz2, fyz2, fz(x2-y2), fxyz, fx(x2-3y2), fy(3x2-y2).
    Numba JIT compiled. theta and phi are expected to be NumPy arrays.
    """
    # Ensure inputs are arrays for Numba, though they usually will be from upstream
    # theta_arr = np.asarray(theta) # Not needed if already arrays
    # phi_arr = np.asarray(phi)

    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    sin_p = np.sin(phi)
    cos_p = np.cos(phi)
    sin_2p = np.sin(2*phi) # Used in fxyz alternative (2sinp cosp)
    cos_2p = np.cos(2*phi) # Used in fz(x2-y2) and d(x2-y2)
    sin_3p = np.sin(3*phi)
    cos_3p = np.cos(3*phi)

    # Numba requires list elements to be of compatible types (e.g. all 1D arrays)
    # We will build a list of arrays.
    
    if l == 0: 
        return [np.ones_like(theta)] # List containing one array
    elif l == 1: # Order: px, py, pz
        S_px = sin_t * cos_p  
        S_py = sin_t * sin_p
        S_pz = cos_t          
        return [S_px, S_py, S_pz] 
    elif l == 2: # PySCF d-orbital order: dxy, dyz, dz2, dxz, dx2y2
        sin_t_sq = sin_t**2
        cos_t_sq = cos_t**2
        
        S_dxy_comp   = sin_t_sq * cos_p * sin_p            # xy/r^2 (using cos_p * sin_p = 0.5 * sin_2p)
        S_dyz_comp   = sin_t * cos_t * sin_p               # yz/r^2
        S_dz2_comp   = (3.0 * cos_t_sq - 1.0) / 2.0        # (3z^2-r^2)/(2r^2)
        S_dxz_comp   = sin_t * cos_t * cos_p               # xz/r^2
        S_dx2y2_comp = sin_t_sq * (cos_p**2 - sin_p**2)    # (x^2-y^2)/r^2 (using cos_p**2 - sin_p**2 = cos_2p)
        
        return [S_dxy_comp, S_dyz_comp, S_dz2_comp, S_dxz_comp, S_dx2y2_comp]
    elif l == 3: # f-orbitals. Order: fz3, fxz2, fyz2, fz(x2-y2), fxyz, fx(x2-3y2), fy(3x2-y2)
        S_fz3 = cos_t * (5.0 * cos_t**2 - 3.0 * np.ones_like(cos_t)) / 2.0 
        S_fxz2 = sin_t * cos_p * (5.0 * cos_t**2 - 1.0)
        S_fyz2 = sin_t * sin_p * (5.0 * cos_t**2 - 1.0)
        S_fz_x2_y2 = cos_t * sin_t**2 * cos_2p
        S_fxyz = sin_t**2 * cos_t * sin_p * cos_p # sin_2p / 2.0 * cos_t * sin_t
        S_fx_x2_3y2 = sin_t**3 * cos_3p
        S_fy_3x2_y2 = sin_t**3 * sin_3p
        
        return [S_fz3, S_fxz2, S_fyz2, S_fz_x2_y2, S_fxyz, S_fx_x2_3y2, S_fy_3x2_y2]
    else:
        # Numba doesn't support raising NotImplementedError directly in nopython mode easily.
        # Return an empty list or a specific indicator if l is out of implemented range.
        # For now, let's assume l will be within the handled range.
        # If not, the calling Python code should check.
        # Or, one could return a list containing an array of NaNs or zeros.
        # Example: return [np.full_like(theta, np.nan)]
        # However, the original code raises an error, so the calling code expects valid lists.
        # This path should ideally not be hit if inputs are validated.
        # To make Numba happy if it must return something:
        empty_list_of_arrays = [np.empty(0, dtype=np.float64)] 
        return empty_list_of_arrays # Placeholder for unhandled l, adjust as needed
                                     # Or ensure l is always valid before calling.


# --- Main AO Evaluation Function (Optimized) ---
def compute_atomic_orbitals(grid_points, atoms_data, gto_data):
    """
    Computes atomic orbital values on a grid, optimized with Numba and pre-allocation.
    Handles generalized shell types like 's', 'p', 'd', 'f', 'sp'.
    Assumes the final AO is Sum_k d_k * (Normalized Primitive GTO).
    Normalized Primitive GTO = N_k_prim * r^l * S_lm(theta,phi) * exp(-alpha_k_scaled * r^2)
    This function computes: Sum_k [ d_k * N_k_prim * exp(-alpha_k_scaled * r^2) ] * r^l * S_lm(theta,phi)
    """
    num_grid_points = grid_points.shape[0]

    # 1. Pre-calculate the total number of atomic orbitals
    total_num_aos = 0
    ao_label_placeholders = [] # To store (atom_idx, n_val_label, orbital_type_label_component) for actual label construction later

    for atom_gto_idx, atom_gto_info in enumerate(gto_data):
        atom_props = atoms_data[atom_gto_info['atom_index']] # Get atom properties for labeling
        for shell_info in atom_gto_info['shells']:
            shell_type_full = shell_info['type'].lower()
            n_quantum_number_shell_label = shell_info.get('n_quantum_number', '') # For labeling like "1s", "2p"

            for current_coeff_idx, char_l in enumerate(shell_type_full):
                if char_l not in L_QUANTUM_NUMBERS_MAP:
                    continue
                l_val = L_QUANTUM_NUMBERS_MAP[char_l]
                if l_val not in ANGULAR_LABELS:
                    continue
                
                # Check if this component will actually be processed (i.e., has coefficients)
                # This check needs to be robust. Assume if primitives exist, and the first one has enough coeffs.
                can_process_component = False
                if shell_info['primitives']:
                    # Check if *any* primitive has enough coefficients for this component.
                    # A more accurate check would be if *at least one* primitive has a non-zero coeff for this component.
                    # For counting purposes, we check if the structure allows for this coefficient.
                    # The actual processing loop will skip primitives with zero coeffs.
                    if current_coeff_idx < len(shell_info['primitives'][0]['coefficients']):
                         can_process_component = True
                
                if can_process_component:
                    angular_momentum_label_suffixes = ANGULAR_LABELS[l_val]
                    total_num_aos += len(angular_momentum_label_suffixes)
                    for ang_label_suffix in angular_momentum_label_suffixes:
                        ao_label_placeholders.append(
                            (atom_gto_info['atom_index'], str(n_quantum_number_shell_label), ang_label_suffix)
                        )


    if total_num_aos == 0:
        return np.empty((num_grid_points, 0), dtype=np.float64), []

    ao_matrix = np.empty((num_grid_points, total_num_aos), dtype=np.float64)
    ao_labels = ["" for _ in range(total_num_aos)] # Pre-allocate Python list for string labels
    current_ao_matrix_idx = 0 # Index for ao_matrix columns

    # Construct actual labels (done once)
    for i in range(total_num_aos):
        atom_idx_for_label, n_val_str, orb_type_str = ao_label_placeholders[i]
        atom_label_prefix = atoms_data[atom_idx_for_label]['label'] + str(atom_idx_for_label)
        ao_labels[i] = f"{atom_label_prefix}_{n_val_str}{orb_type_str}"


    for atom_gto_info in gto_data:
        atom_idx = atom_gto_info['atom_index']
        atom_props = atoms_data[atom_idx]
        atom_center = np.array([atom_props['x'], atom_props['y'], atom_props['z']], dtype=np.float64)

        R_vectors = grid_points - atom_center  
        x_rel, y_rel, z_rel = R_vectors[:, 0], R_vectors[:, 1], R_vectors[:, 2]

        r_sq = x_rel**2 + y_rel**2 + z_rel**2
        # r_stable is sqrt(r_sq), used for r^l term and for theta calculation
        # Add small epsilon for r_for_division_in_theta to avoid division by zero if a grid point is at atom_center
        epsilon_r_div = 1e-12 
        r_stable = np.sqrt(r_sq) 
        
        # Avoid division by zero if r_stable is zero for theta calculation
        # np.clip ensures z_rel / r_for_division is within [-1, 1] for arccos
        # Create a mask for points where r_stable is very small
        zero_r_mask = r_stable < epsilon_r_div
        
        # For points where r_stable is zero, theta can be defined as 0 (along z-axis)
        # or handle as appropriate. Here, we ensure no division by zero.
        # If r_stable is zero, z_rel is also zero. z_rel / (r_stable + eps) will be 0.
        # arccos(0) is pi/2. If z_rel is also 0, this is fine.
        # If r_stable is truly 0, x_rel, y_rel, z_rel are 0.
        # phi is undefined at the origin (r_stable=0). arctan2(0,0) is 0.
        # theta (polar angle from z-axis) is also somewhat arbitrary at origin.
        # By convention, if z_rel is 0 and r_stable is 0, arccos(0) = pi/2.
        # If x_rel, y_rel, z_rel are all zero, r_stable is zero.
        # For these points, the r^l term (if l>0) will make the AO zero anyway.
        # If l=0 (s-orbital), it's non-zero. The angular part S_00 is constant.

        theta_vals = np.arccos(np.clip(z_rel / (r_stable + epsilon_r_div), -1.0, 1.0))
        phi_vals = np.arctan2(y_rel, x_rel) 
        
        for shell_info in atom_gto_info['shells']:
            shell_type_full = shell_info['type'].lower()
            scale_factor = shell_info['scale_factor']
            primitives_in_shell = shell_info['primitives']
            n_quantum_number_shell_label = str(shell_info.get('n_quantum_number', '')) # For labeling

            if not primitives_in_shell: # Skip shell if it has no primitives
                continue

            # Iterate through each angular momentum component defined in the shell type string
            # e.g., for "sp", char_l will be 's' then 'p'. coeff_idx will be 0 then 1.
            for current_coeff_idx, char_l in enumerate(shell_type_full):
                if char_l not in L_QUANTUM_NUMBERS_MAP:
                    # print(f"Warning: Unknown angular momentum char '{char_l}' in shell '{shell_type_full}'. Skipping.")
                    continue
                l_val = L_QUANTUM_NUMBERS_MAP[char_l]
                
                if l_val not in ANGULAR_LABELS:
                    # print(f"Warning: Angular labels not defined for l={l_val} (from '{char_l}'). Skipping.")
                    continue
                
                # Prepare data for Numba: extract exponents and relevant coefficients
                # for the current l_val (determined by current_coeff_idx)
                exponents_list = []
                coeffs_list_for_l_component = []

                for prim_data in primitives_in_shell:
                    if current_coeff_idx < len(prim_data['coefficients']):
                        # Only include if this primitive has a coefficient for the current l-component
                        exponents_list.append(prim_data['exponent'])
                        coeffs_list_for_l_component.append(prim_data['coefficients'][current_coeff_idx])
                    # else:
                        # This primitive doesn't contribute to this specific l-component of a general (e.g. 'sp') shell
                        # print(f"Debug: Primitive in shell '{shell_type_full}' for atom {atom_idx} lacks coeff at index {current_coeff_idx} for l-component '{char_l}'.")

                if not exponents_list: # No primitives contribute to this specific l-component
                    # print(f"Debug: No primitives found for l-component '{char_l}' (l={l_val}, coeff_idx={current_coeff_idx}) in shell '{shell_type_full}' for atom {atom_idx}. Skipping component.")
                    continue

                exponents_np = np.array(exponents_list, dtype=np.float64)
                coeffs_np_for_l = np.array(coeffs_list_for_l_component, dtype=np.float64)

                # Call Numba function for the radial part (sum of Gaussians)
                # This part is Sum_k [ d_k * N_k_prim * exp(-alpha_k_scaled * r^2) ]
                sum_gaussians_part = compute_radial_part_numba(
                    r_sq,
                    exponents_np,
                    coeffs_np_for_l,
                    scale_factor**2,
                    l_val
                )
                
                # Angular part S_lm(theta, phi)
                # Ensure real_sph_harmonics_pyscf_order_numba returns a list of arrays
                Slm_angular_parts_list = real_sph_harmonics_pyscf_order_numba(l_val, theta_vals, phi_vals)

                if not Slm_angular_parts_list or (len(Slm_angular_parts_list) == 1 and Slm_angular_parts_list[0].shape[0] == 0) :
                    # This indicates an issue with real_sph_harmonics for the given l_val (e.g. l > 3)
                    # print(f"Warning: real_sph_harmonics_pyscf_order_numba returned empty or invalid for l={l_val}. Skipping.")
                    continue

                # Radial prefactor r^l
                # If r_stable is zero at some points, and l_val > 0, then r_pow_l will be zero.
                # If l_val is 0, r_pow_l is 1.
                # np.power(0.0, 0) is 1.0, which is correct for s-orbitals (l=0).
                r_pow_l = np.power(r_stable, l_val)
                # For points where r_stable was zero (masked points), r_pow_l will be zero if l_val > 0.
                # If l_val is 0, r_pow_l is 1. This is correct.

                angular_momentum_label_suffixes = ANGULAR_LABELS[l_val]
                for i, ang_part_Slm_theta_phi in enumerate(Slm_angular_parts_list):
                    if current_ao_matrix_idx >= total_num_aos:
                        # This should not happen if total_num_aos was calculated correctly
                        print("Error: Exceeded pre-allocated AO matrix size. Check total_num_aos calculation.")
                        break 
                    
                    # Full AO component: Sum_Gaussians * r^l * S_lm
                    ao_component_values = sum_gaussians_part * r_pow_l * ang_part_Slm_theta_phi
                    
                    # Assign to the pre-allocated matrix
                    ao_matrix[:, current_ao_matrix_idx] = ao_component_values
                    
                    # Label was already pre-calculated and stored in ao_labels
                    # orbital_type_label_component = angular_momentum_label_suffixes[i]
                    # label = f"{atom_props['label']}{atom_idx}_{n_quantum_number_shell_label}{orbital_type_label_component}"
                    # ao_labels[current_ao_matrix_idx] = label # This is now done once at the beginning

                    current_ao_matrix_idx += 1
            
            if current_ao_matrix_idx > total_num_aos: # Safety break if something went wrong with indexing
                break
        if current_ao_matrix_idx > total_num_aos:
            break
            
    # Final check if the number of generated AOs matches the pre-calculated count
    if current_ao_matrix_idx != total_num_aos:
        print(f"Warning: Number of generated AOs ({current_ao_matrix_idx}) does not match pre-calculated count ({total_num_aos}). Trimming matrix if necessary.")
        # If fewer AOs were generated than expected, trim the matrix and labels
        ao_matrix = ao_matrix[:, :current_ao_matrix_idx]
        ao_labels = ao_labels[:current_ao_matrix_idx]


    return ao_matrix, ao_labels

####Plotter######
def plot3D(coordinates, n_atoms, symbols, dist_mat):
    """
    Visualizes the molecule structure in 3D with bond-specific color gradients.

    This function generates a 3D plot of the molecule. Atoms are spheres,
    and each bond is a tube colored with a unique gradient corresponding
    only to the two atoms it connects.

    Args:
        coordinates (numpy.ndarray): A 2D array of atom coordinates.
        n_atoms (int): The number of atoms in the molecule.
        symbols (list): A list of element symbols for each atom.
    """

    # --- Color and Radius Definitions ---
    color_dict_rgb = {
        'C': 'black', 'H': 'gainsboro', 'O': 'red', 'F': 'cyan',
        'Cl': 'green', 'N': 'blue', 'S': 'yellow', 'Br': 'darkred',
        'B': 'pink', 'Al': 'brown', 'Fe': 'orange'
    }
    radius_data = {
        'C': 0.70, 'H': 0.31, 'O': 0.48, 'F': 0.42, 'Cl': 0.79,
        'N': 0.56, 'S': 1.00, 'Br': 1.14, 'B': 0.85, 'Al': 1.43, 'Fe': 1.26
    }
    default_color = 'gray'
    default_radius = 0.5

    # --- Atom Plotting ---
    for i in range(n_atoms):
        color = color_dict_rgb.get(symbols[i], default_color)
        radius = radius_data.get(symbols[i], default_radius)
        sphere = pv.Sphere(radius=radius, center=coordinates[i])
        plotter.add_mesh(sphere, color=color)
    # Define scalar bar arguments once to hide the color legend
    sargs = dict(n_labels=0)

    # Iterate through each unique pair of atoms
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            # Check if a bond exists between atom i and atom j
            if dist_mat[i][j] != 0:
                # 1. Get the colors for the two specific atoms in the bond
                color_i = mcolors.to_rgb(color_dict_rgb.get(symbols[i], default_color))
                color_j = mcolors.to_rgb(color_dict_rgb.get(symbols[j], default_color))

                # 2. Create a dedicated 2-color map for this bond only
                cmap_bond = ListedColormap([color_i, color_j])

                # 3. Create a PolyData object for this single bond line
                points = np.array([coordinates[i], coordinates[j]])
                lines = np.array([2, 0, 1])  # Connects the two points
                bond_poly = pv.PolyData(points, lines=lines)

                # 4. Add scalars [0, 1] to the endpoints of the line.
                # This maps directly to the two colors in cmap_bond.
                bond_poly.point_data['bond_scalars'] = np.arange(2)

                # 5. Create a tube from the line and add it to the plotter
                tube = bond_poly.tube(radius=0.15) # You can adjust the radius here
                plotter.add_mesh(
                    tube,
                    scalars='bond_scalars',
                    cmap=cmap_bond,
                    scalar_bar_args=sargs, # Hides the scalar bar
                    smooth_shading=True
                )

    plotter.set_background('white')


def plot_orbitals(mo_coeff, ao_values, iso_value, grid_x, grid_y, grid_z):
    """
    Plots molecular orbitals as 3D isosurfaces.

    This function visualizes molecular orbitals by creating isosurfaces
    corresponding to positive and negative values of the orbital.

    Args:
        mo_coeff (numpy.ndarray): Molecular orbital coefficients.
                                Shape: (n_atomic_orbitals,).
        ao_values (numpy.ndarray): Atomic orbital values on the grid.
                                 Shape: (n_grid_points, n_atomic_orbitals).
        iso_value (float): Isovalue for the surfaces.
    """

    # Calculate MO values on the grid
    mo_values = np.dot(ao_values, mo_coeff)

    # Create a structured grid for the MO data
    grid = pv.StructuredGrid(grid_x, grid_z, grid_y)
    grid.point_data['mo_values'] = mo_values.real  # Assign MO values to the grid

    # Create isosurfaces
    iso_surface_pos = grid.contour([iso_value], scalars='mo_values')  # Positive isosurface
    iso_surface_neg = grid.contour([-iso_value], scalars='mo_values')  # Negative isosurface

    # Add the surfaces to the plotter
    plotter.add_mesh(iso_surface_pos, color='red', smooth_shading=True, specular=1, specular_power=15, opacity=1)
    plotter.add_mesh(iso_surface_neg, color='darkblue', smooth_shading=True, specular=1, specular_power=15, opacity=1)

### Main Script ####

if __name__ == '__main__':
    # Get the current working directory (where the script is run from)
    current_directory = os.getcwd()
    
    print(f"Scanning for Molden files in the current directory: '{current_directory}'")
    molden_files = parse_directory_for_molden_files(current_directory)


for file in molden_files:
    file = str(file)
    molecule = file.split('.')[0]
    molecule_upper = molecule.upper()
    print(f'CALCULATIONS FOR {molecule_upper}.')
    atoms, gtos, mos = parse_molden_file(file)
    coordinates = []
    mo_energies = []
    mo_occup = []
    mo_coeffs = []
    symbols = []
    n_atoms = int(len(atoms))
    for i in range(len(atoms)):
        symbol = atoms[i]['label']
        symbol = symbol[0]
        symbols.append(symbol)
        x = atoms[i]['x']
        y = atoms[i]['y']
        z = atoms[i]['z']
        coordinates.append((x,y,z))

    for i in range(len(mos)):
        mo_energies.append(mos[i]['energy'])
        mo_occup.append(mos[i]['occupancy'])
        mo_coeffs.append(mos[i]['coefficients'])
    coordinates = np.array(coordinates)
    mo_energies = np.array(mo_energies)
    mo_coeffs = np.array(mo_coeffs)
    mo_occup = np.array(mo_occup)
    HOMO_index = np.where(mo_occup>=1)
    HOMO_index = HOMO_index[-1][-1] +1

    # User input for plot settings
    resolution = int((input(f'Enter an integer value for number of grid points (default value = 61): ')) or '61')
    iso_value = float((input(f'Enter a value for the iso value (default value = 0.01): ')) or '0.01')
        # Extract atomic coordinates
    X = coordinates[:, 0]
    Y = coordinates[:, 1]
    Z = coordinates[:, 2]

    # Calculate the maximum extent of the molecule
    max_extent = max(np.max(X) - np.min(X), np.max(Y) - np.min(Y), np.max(Z) - np.min(Z))
    buffer = 5  # Buffer to add around the molecule
    grid_range = max_extent / 2 + buffer  # Define the range for the grid

    # Define the 1D grid along each axis
    x = np.linspace(-grid_range, grid_range, resolution)
    y = np.linspace(-grid_range, grid_range, resolution)
    z = np.linspace(-grid_range, grid_range, resolution)

    # Create a 3D grid
    grid_x, grid_y, grid_z = np.meshgrid(x, y, z)
    points = np.vstack((grid_x.ravel(), grid_y.ravel(), grid_z.ravel())).T  # Stack grid points into (N, 3) array

    ao_values , ao_labels = compute_atomic_orbitals(points, atoms, gtos)
    
    # --- Distance matrix calculation ---
    dist_mat = distance_matrix(coordinates, n_atoms)

    # Get the range of MOs to plot from user input
    print(f"The HOMO orbital is the orbital number {HOMO_index}")
    range_orbitals = str(input("Enter the MOs to plot. \nTwo integers for range(format MO_start MO_final) or one integer for only one MO (0 for None MO to be plotted): \n"))
    range_orbitals = range_orbitals.split()

    # Parse the input range
    if len(range_orbitals) == 2:
        MO_first = int(range_orbitals[0])
        MO_last = int(range_orbitals[1])
    else:
        MO_first = int(range_orbitals[0])
        MO_last = int(range_orbitals[0])

    # Plot specified molecular orbitals
    for i in range(MO_first, MO_last + 1):
        selected_mo = i
        if selected_mo != 0:
            plotter = pv.Plotter(off_screen=True)  # Initialize plotter here
            plot3D(coordinates, n_atoms, symbols, dist_mat)
            mo_coeff = mo_coeffs[selected_mo - 1, :]
            plot_orbitals(mo_coeff, ao_values, iso_value, grid_x, grid_y, grid_z)
            print(f'Saving MO{selected_mo} as a gltf file.')
            plotter.export_gltf(f"{molecule}_MO{selected_mo}_E{mo_energies[selected_mo - 1]:.2f}_Occup{mo_occup[selected_mo - 1]:.2f}.gltf")
            plotter.close()  # Close the plotter after exporting
        elif selected_mo == 0:
            plotter = pv.Plotter(off_screen=True)
            plot3D(coordinates, n_atoms, symbols, dist_mat)
            plotter.export_gltf(f"{molecule}.gltf")
            plotter.close()
    plotter = None
    print(f'Finished the plot for {molecule}')


