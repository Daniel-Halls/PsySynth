# Mathematical Audit Report: PsySynth Meta-Analysis Pipeline

This report details a rigorous mathematical and logical audit of the conjugate Beta-Binomial update equations, empirical prior estimations, and coordinate translation mappings in `run_meta_analysis.py` and `app.py` within the PsySynth workspace.

---

## 1. Executive Summary

A comprehensive audit of the statistical and spatial logic of the PsySynth codebase has revealed several critical mathematical inconsistencies, coordinate query mismatches, and path-resolution bugs:
1. **Empirical Prior and Study Count ($N$) Mismatch**: The Dash application (`app.py`) hardcodes a study count of $N = 73$ for the functional modality and applies it globally, whereas the loaded functional metadata actually contains $68$ studies.
2. **Global Prior Application to Structural Data**: Prior parameters are computed once at startup using functional data (or fall back to hardcoded functional parameters) and are applied to both structural and functional queries. When structural data is viewed, the app queries functional studies and uses functional priors, resulting in incorrect stats.
3. **Distance Metric Discretization Mismatch**: There is a logical discrepancy between how spheres are drawn during map generation (`run_meta_analysis.py`) and how contributing studies are queried in the UI (`app.py`). This causes **2,092 mismatched voxel-peak combinations in just the first 100 peaks** analyzed.
4. **Broken File Paths**: Recent directory reorganization moved data files to `data/anorexia/`, but `app.py` remains hardcoded to look for them directly under the root or in a `functional/` subfolder, causing the app to fall back to blank templates and zero-filled arrays.
5. **Hardcoded Voxel Boundary Clipping**: Clipping limits in `app.py` are hardcoded to `(90, 108, 90)` rather than dynamically adapting to the shape of the loaded image.

---

## 2. Conjugate Beta-Binomial Update Equations

The Bayesian meta-analysis models voxel activation as a sequence of independent Bernoulli trials across $N$ studies.

### Mathematical Formulation
For each voxel:
* **Prior**: $\theta \sim \text{Beta}(\alpha_0, \beta_0)$, where $\theta$ is the activation probability.
* **Likelihood**: $k \sim \text{Binomial}(N, \theta)$, where $k$ is the number of active studies at that voxel.
* **Posterior**: $\theta \mid k \sim \text{Beta}(\alpha_{\text{post}}, \beta_{\text{post}})$

The conjugate update formulas are:
$$\alpha_{\text{post}} = \alpha_0 + k$$
$$\beta_{\text{post}} = \beta_0 + N - k$$

The posterior mean $E[\theta \mid k]$ is:
$$E[\theta \mid k] = \frac{\alpha_{\text{post}}}{\alpha_{\text{post}} + \beta_{\text{post}}} = \frac{\alpha_0 + k}{\alpha_0 + \beta_0 + N}$$

### Implementation Consistency Check
* **`run_meta_analysis.py`**: The updates are mathematically consistent and correctly implemented:
  ```python
  alpha_post = alpha_0 + k_map
  beta_post = beta_0 + N - k_map
  posterior_mean = alpha_post / (alpha_post + beta_post)
  ```
  Since $\alpha_0 + \beta_0 = \sqrt{N}$, the denominator simplifies to $\sqrt{N} + N$, which matches the theory.
* **`app.py`**: The formulas are implemented correctly in code (line 242):
  ```python
  post_mean = (alpha_0 + k) / (alpha_0 + beta_0 + N_formula)
  ```
  However, the underlying parameters ($\alpha_0, \beta_0, N_{\text{formula}}$) are mathematically mismatched (see Section 3).

---

## 3. Empirical Prior Estimation and Hardcoding Issues

The empirical prior parameters $\alpha_0$ and $\beta_0$ are computed from the Global Background Rate ($GBR$) and a prior strength $W = \sqrt{N}$ (equivalent to $\sqrt{N}$ studies):
$$GBR = \frac{\text{Total Hits in Mask}}{\text{Total Voxels in Mask} \times N}$$
$$\alpha_0 = GBR \times \sqrt{N}$$
$$\beta_0 = (1.0 - GBR) \times \sqrt{N}$$

### Inconsistencies and Hardcoded Parameters in `app.py`

#### 1. Mismatch in Functional Study Count
* **Metadata file**: `data/anorexia/functional/phase3_metadata.json` contains $N = 68$ studies.
* **Hardcoded formula count**: `app.py` hardcodes `N_formula = 73` (line 74).
* **Prior calculation error**: During startup, `app.py` loops over the $68$ functional studies to compute `total_hits` but divides by $73$ in the $GBR$ formula:
  ```python
  GBR = total_hits / (total_voxels * N_formula) # divides by total_voxels * 73
  ```
  This artificially underestimates the $GBR$ by about $7\%$. It then uses $W = \sqrt{73}$ instead of $\sqrt{68}$.
* **Posterior Mean Underestimation**: In `query_coordinate_stats`, the posterior mean is computed as:
  ```python
  post_mean = (alpha_0 + k) / (alpha_0 + beta_0 + N_formula) # Denominator = sqrt(73) + 73
  ```
  Since there are only $68$ functional studies in the database, the maximum possible value for $k$ is $68$. Thus, even if a voxel is active in all $68$ studies, the reported posterior mean can never reach its correct maximum value, as it is scaled by a denominator that assumes $73$ studies.

#### 2. Hardcoded Fallback Priors
* If the prior computation raises an exception, the app falls back to hardcoded parameters (lines 126-127):
  ```python
  alpha_0 = 0.153213
  beta_0 = 8.390790
  ```
  These correspond to $N = 73$ and a $GBR \approx 0.017932$. They do not adapt if different datasets or study counts are loaded.

#### 3. Cross-Modality Prior Mismatch
* The app allows the user to toggle between "Functional MRI" and "Structural MRI" (line 281).
* However, the app **only loads the functional studies metadata** at startup. It does not load the structural studies (`data/anorexia/structure.json`, $N=24$, or `structure_metadata.json`, $N=43$).
* As a result, when the user views the Structural MRI modality, the "Overlapping Studies" list and "Bayesian Posterior Mean" in the UI are calculated using the **functional studies and functional prior parameters** ($N = 73$), which are completely mismatched with the structural maps (which were generated with $N = 24$ and $\alpha_0 = 0.042788, \beta_0 = 4.856192$).

---

## 4. Coordinate Mapping and Distance Query Discrepancies

### 1. Distance Metric Mismatch
There is a fundamental discrepancy in how coordinates are evaluated in physical space:
* **Map Generation (`run_meta_analysis.py`)**:
  1. Rounds the exact peak coordinate $P = (p_x, p_y, p_z)$ to the nearest voxel index $V_{\text{peak}} = \text{round}(M^{-1} P)$ where $M$ is the affine matrix.
  2. Identifies the physical center of this voxel $C(V_{\text{peak}}) = M \cdot V_{\text{peak}}$.
  3. Precomputes relative voxel offsets $\vec{o}$ whose physical distance from $(0,0,0)$ is $\le 10$ mm: $\| M \vec{o} \|_2 \le 10$ mm.
  4. Marks voxel $V$ active if it lies in $V_{\text{peak}} + \vec{o}$, which translates to:
     $$\| C(V) - C(V_{\text{peak}}) \|_2 \le 10.0\text{ mm}$$
* **UI Query (`app.py`)**:
  Checks if the distance from the queried voxel center $C(V)$ to the **exact peak coordinates** $P$ is $\le 10$ mm:
  $$\| C(V) - P \|_2 \le 10.0\text{ mm}$$

Because $P \neq C(V_{\text{peak}})$ in general, these two conditions are not logically equivalent. This discretization discrepancy causes significant mismatches:
* **Under-counting (Voxel is active, but query says no)**: A voxel is marked active in the generated NIfTI map, but the UI report lists $0$ contributing studies.
* **Over-counting (Voxel is inactive, but query says yes)**: A voxel is inactive in the map, but the UI query lists the study as contributing.

#### Discrepancy Analysis (Anorexia Functional Dataset)
Running a comparative script on the first $100$ peaks in `phase3_metadata.json` yields **2,092 mismatched voxel-peak combinations**. Below are concrete examples of this error:

* **Mismatch Example 1 (Under-counting)**:
  * **Peak**: $(36.0, -12.0, -27.0)$ mm, which maps to voxel center $[36.0, -12.0, -28.0]$ mm.
  * **Voxel**: $[23, 55, 20]$ with center $[44.0, -16.0, -32.0]$ mm.
  * **Map Distance (Center-to-Center)**: $9.80$ mm ($\le 10$ mm) $\rightarrow$ **Active in Map (True)**.
  * **Query Distance (Center-to-Exact)**: $10.25$ mm ($> 10$ mm) $\rightarrow$ **Not counted in UI (False)**.
  
* **Mismatch Example 2 (Over-counting)**:
  * **Peak**: $(36.0, -12.0, -27.0)$ mm, which maps to voxel center $[36.0, -12.0, -28.0]$ mm.
  * **Voxel**: $[23, 56, 25]$ with center $[44.0, -14.0, -22.0]$ mm.
  * **Map Distance (Center-to-Center)**: $10.20$ mm ($> 10$ mm) $\rightarrow$ **Inactive in Map (False)**.
  * **Query Distance (Center-to-Exact)**: $9.64$ mm ($\le 10$ mm) $\rightarrow$ **Counted in UI (True)**.

### 2. Hardcoded Clipping Boundaries
In `app.py`, the `mni_to_voxel` function clips index coordinates to hardcoded values (lines 151-153):
```python
i = np.clip(i, 0, 90)
j = np.clip(j, 0, 108)
k = np.clip(k, 0, 90)
```
While this matches the $91 \times 109 \times 91$ shape of the standard $2$mm template, hardcoding these numbers prevents the application from working correctly with other template resolutions (e.g., $1$mm templates with shape $182 \times 218 \times 182$). The clipping limits should be bound dynamically to the loaded template shape (i.e., `shape[0]-1`, `shape[1]-1`, `shape[2]-1`).

### 3. File Path Mismatch / Loading Failures
Due to recent file organization, the data files were moved to `data/anorexia/`. However, `app.py` still contains:
```python
BASE_DIR = "/Users/mszdjh3/code/PsySynth"
brain_path = os.path.join(BASE_DIR, "functional/MNI152_T1_2mm_brain.nii.gz")
map_paths = {
    ('structure', 'exceedance'): os.path.join(BASE_DIR, "structure_thresholded_exceedance_map.nii.gz"),
    ...
}
metadata_path = os.path.join(BASE_DIR, "functional/phase3_metadata.json")
```
None of these paths resolve. This causes:
1. The anatomical template loading to fail, falling back to a zero-filled array.
2. All four modality maps to fail to load, falling back to zero-filled arrays.
3. The metadata JSON loading to fail, meaning no studies are loaded for cross-referencing, and the coordinate query returns empty results.

---

## 5. Recommendations for Corrective Actions

1. **Unify the Distance Metrics**:
   To align map generation and UI coordinate queries, `app.py`'s distance query should match `run_meta_analysis.py`'s center-to-center heuristic. Alternatively, both should be updated to compute exact Euclidean distance between each voxel center and the exact peak coordinates (which is more mathematically precise but requires updating the 3D map generation logic).
2. **Dynamic Prior Loading by Modality**:
   Modify `app.py` to dynamically load structural and functional study metadata files depending on the selected modality, and recompute the prior parameters ($\alpha_0, \beta_0, N$) dynamically using the length of the active metadata list rather than relying on hardcoded constants.
3. **Resolve Path Structures**:
   Update `BASE_DIR` or the paths in `app.py` to correctly point to the `data/anorexia/` folder, or add support for selecting different datasets (e.g., anorexia, depression, anxiety) via the UI.
4. **Dynamic Clipping Bounds**:
   Use `brain_data.shape` to dynamically define boundary clipping in `mni_to_voxel`.
