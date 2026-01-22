# Atomic-orbitals-visualizer
A high-performance Python tool for visualizing molecular structures and atomic orbitals. It parses **Molden** format files, computes orbital values on a grid using **Numba JIT compilation** for speed, and exports interactive 3D models (GLTF) using **PyVista**.
# Numba-Accelerated Atomic Orbital Visualizer

![License](https://img.shields.io/badge/license-MIT-blue.svg) ![Python](https://img.shields.io/badge/python-3.8+-blue.svg) ![Numba](https://img.shields.io/badge/numba-accelerated-green)

A high-performance Python tool for visualizing molecular structures and atomic orbitals. It parses **Molden** format files, computes orbital values on a grid using **Numba JIT compilation** for speed, and exports interactive 3D models (GLTF) using **PyVista**.

## 🌟 Features

* **Fast Computation:** Uses `@numba.njit` to compile calculation-heavy functions (factorial, radial part, spherical harmonics) to machine code.
* **3D Visualization:** Generates isosurfaces for Molecular Orbitals (MOs) and renders atoms/bonds with element-specific colors.
* **Interactive Output:** Exports files to `.gltf` format, ready for web viewers, Blender, or PowerPoint 3D.
* **Molden Support:** Compatible with standard output from computational chemistry packages (Gaussian, ORCA, Molpro, etc.) converted to Molden format.

## 🛠 Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/Atomic-Orbital-Visualizer.git](https://github.com/YOUR_USERNAME/Atomic-Orbital-Visualizer.git)
    cd Atomic-Orbital-Visualizer
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## 🚀 Usage

The script scans the **current directory** for any `.molden` files and processes them automatically.

1.  **Place your `.molden` files** in the project root (or copy the script to your data folder).
2.  **Run the script:**
    ```bash
    python src/Atomic_orbs_numba.py
    ```
3.  **Follow the interactive prompts:**
    * **Grid Points:** Define resolution (Default: 61). Higher = smoother but slower.
    * **Iso Value:** Define the surface density threshold (Default: 0.01).
    * **MO Range:** Select which orbitals to plot (e.g., `20 21` for HOMO/LUMO, or `0` for just the geometry).

### Example with provided data
We provide a dummy $H_2$ file in `data/`. Copy it to the root to test:
```bash
cp data/example.molden .
python src/Atomic_orbs_numba.py
```
## 📂 Outputs
The script generates **.gltf** files in the same directory:

* **MoleculeName.gltf** (Geometry only)

* **MoleculeName_MO1_E-0.50.gltf** (Molecular Orbital 3D models)

## 📄 License
Distributed under the MIT License. See LICENSE for more information.
