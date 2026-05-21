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