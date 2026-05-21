# PsySynth

PsySynth is an automated neuroimaging meta-analysis pipeline that downloads PMC articles, extracts spatial activation coordinates and study metadata, normalizes coordinates to MNI space, and performs a Bayesian meta-analysis using a Beta-Binomial conjugate update with an Empirical Prior.

---

## Dependencies & Requirements

To run the pipeline, ensure the following Python packages are installed:
- `numpy`
- `scipy`
- `nibabel`
- `beautifulsoup4`
- `lxml`
- `requests`

If using the local environment:
```bash
python -m pip install numpy scipy nibabel beautifulsoup4 lxml requests
```

For Phase 3 metadata extraction, you must also have a local instance of [Ollama](https://ollama.com/) running with the `llama3` model:
```bash
ollama run llama3
```

---

## Execution Pipeline

Follow these steps sequentially to extract coordinates and generate the meta-analysis maps.

### Step 1: Download PMC Full-Text XMLs
Fetch open-access articles from Europe PMC using a search query (e.g., searching for fMRI studies related to anorexia).
```bash
python3 fetch_pmc_xmls.py --query "anorexia AND fMRI AND OPEN_ACCESS:Y" --target_dir xmls
```
*(Alternatively, if the package is installed, you can run: `fetch-pmc-xmls --query "anorexia AND fMRI AND OPEN_ACCESS:Y" --target_dir xmls`)*

### Step 2: Extract Peak Coordinates & Statistics
Extract the coordinates ($X, Y, Z$) and peak statistic values from the downloaded XML tables.
```bash
python3 extract_coordinates.py
```
*(Alternatively, you can run: `extract-coordinates`)*
> [!NOTE]
> This command writes extracted data to `phase2_spatial_matrices_v2.json`. To proceed to coordinate normalization, copy or rename this file to `phase2_spatial_matrices.json`:
> ```bash
> cp phase2_spatial_matrices_v2.json phase2_spatial_matrices.json
> ```

### Step 3: Spatial Coordinate Normalization
Harmonize the dataset by converting all Talairach coordinates into MNI (ICBM) space.
```bash
python3 normalize_coordinates.py
```
*(Alternatively, you can run: `normalize-coordinates`)*
This reads from `phase2_spatial_matrices.json` and updates the coordinates in-place.

### Step 4: Extract Study Metadata (LLM)
Query the local Ollama instance running `llama3` to extract the anorexia group sample size ($N$), imaging modality, and contrast description for each study.
```bash
python3 extract_metadata.py
```
*(Alternatively, you can run: `extract-metadata`)*
This reads `phase2_spatial_matrices.json` and outputs the merged dataset `phase3_metadata.json`.

### Step 5: Run the Bayesian Meta-Analysis
Run the meta-analysis using the standard MNI152 brain mask. This computes the Empirical Prior (based on the dataset's Global Base Rate) and applies a fixed Exceedance Probability decision rule ($\tau = 0.10$, confidence level $\ge 0.95$).
```bash
python3 run_meta_analysis.py
```

---

## Generated Maps

Running Step 5 successfully creates the following NIfTI maps in the current directory:
- `posterior_mean_map.nii.gz`: Unthresholded posterior mean map of activation probabilities.
- `exceedance_probability_map_sparsity_t15.nii.gz`: Unthresholded exceedance probability map.
- `thresholded_exceedance_map.nii.gz`: Thresholded exceedance probability map (only showing voxels with exceedance prob $\ge 0.95$).
- `thresholded_posterior_mean.nii.gz`: Thresholded posterior mean map (only showing voxels with exceedance prob $\ge 0.95$).

---

## Mathematical Formulation

The following mathematical formulations describe the operations carried out by the pipeline:

### 1. Spatial Coordinate Normalization (Talairach to MNI)
Coordinates declared in Talairach space are transformed to MNI space using the Lancaster transform matrix (Lancaster et al., 2007). This is the inverse of the standard ICBM-SPM matrix:
$$T = \begin{pmatrix} 0.9254 & 0.0024 & -0.0118 & -1.0207 \\ -0.0048 & 0.9316 & -0.0871 & -1.7667 \\ 0.0152 & 0.0883 & 0.8924 & 4.0926 \\ 0.0000 & 0.0000 & 0.0000 & 1.0000 \end{pmatrix}^{-1}$$

The transformation of homogeneous coordinate vector $v_{\text{Talairach}} = [x, y, z, 1]^T$ to MNI space is:
$$v_{\text{MNI}} = T \cdot v_{\text{Talairach}}$$

### 2. Likelihood Summation
Let $X_s(v) \in \{0, 1\}$ be the binary indicator representing whether study $s$ has a peak coordinate within a 10mm sphere neighborhood of voxel $v$:
$$X_s(v) = \begin{cases} 1 & \text{if } \min_{p \in \mathcal{P}_s} \|v - p\|_2 \le 10\text{ mm} \\ 0 & \text{otherwise} \end{cases}$$
where $\mathcal{P}_s$ is the set of MNI coordinates reported in study $s$.

The total success count $k(v)$ at voxel $v$ across $N$ studies is:
$$k(v) = \sum_{s=1}^N X_s(v)$$

### 3. Data-Driven Empirical Prior
We calculate the Global Base Rate (GBR) of coordinate hits across the standard MNI152 brain mask:
$$\text{GBR} = \frac{\sum_{v \in \text{mask}} k(v)}{V_{\text{mask}} \times N}$$
where $V_{\text{mask}}$ is the number of valid voxels in the brain mask.

We scale the prior weight to $\sqrt{N}$ to define the prior Beta distribution parameters:
$$\alpha_0 = \text{GBR} \times \sqrt{N}$$
$$\beta_0 = (1.0 - \text{GBR}) \times \sqrt{N}$$

### 4. Beta-Binomial Conjugate Update
By combining the Binomial likelihood $k(v) \sim \text{Binomial}(N, \theta(v))$ and the conjugate prior $\theta(v) \sim \text{Beta}(\alpha_0, \beta_0)$, the posterior distribution for activation probability $\theta(v)$ at voxel $v$ is:
$$\theta(v) | k(v) \sim \text{Beta}(\alpha_{\text{post}}(v), \beta_{\text{post}}(v))$$
where:
$$\alpha_{\text{post}}(v) = \alpha_0 + k(v)$$
$$\beta_{\text{post}}(v) = \beta_0 + N - k(v)$$

The posterior mean map is then computed as:
$$\mathbb{E}[\theta(v) | k(v)] = \frac{\alpha_{\text{post}}(v)}{\alpha_{\text{post}}(v) + \beta_{\text{post}}(v)}$$

### 5. Exceedance Probability & Decision Rule
The exceedance probability map $P_{\text{exc}}(v)$ represents the posterior probability that the voxel's true activation probability exceeds a fixed biological threshold $\tau = 0.10$:
$$P_{\text{exc}}(v) = P(\theta(v) > 0.10 \mid k(v)) = \int_{0.10}^{1} f_{\text{Beta}}(t; \alpha_{\text{post}}(v), \beta_{\text{post}}(v)) \, dt$$

The final decision mask is defined by thresholding the exceedance probability at a $95\%$ confidence level:
$$M_{\text{decision}}(v) = \mathbb{I}(P_{\text{exc}}(v) \ge 0.95)$$

We apply this mask to construct the thresholded maps:
$$P_{\text{exc}}^{\text{thresh}}(v) = P_{\text{exc}}(v) \times M_{\text{decision}}(v)$$
$$\mathbb{E}^{\text{thresh}}[\theta(v) | k(v)] = \mathbb{E}[\theta(v) | k(v)] \times M_{\text{decision}}(v)$$