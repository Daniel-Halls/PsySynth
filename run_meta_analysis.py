#!/usr/bin/env python3
import os
import json
import logging
import numpy as np
import scipy.stats as stats
import nibabel as nib

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_brain_mask():
    """Loads standard MNI152 brain mask and returns shape, affine, and data."""
    # Preferred path and fallback options
    mask_paths = [
        "/Users/mszdjh3/fsl/data/standard/MNI152_T1_2mm_brain_mask.nii.gz",
        "/Users/mszdjh3/fsl/pkgs/fsl-data_standard-2208.0-0/data/standard/MNI152_T1_2mm_brain_mask.nii.gz"
    ]
    
    mask_img = None
    for path in mask_paths:
        if os.path.exists(path):
            logging.info(f"Loading standard brain mask from: {path}")
            mask_img = nib.load(path)
            break
            
    if mask_img is None:
        raise FileNotFoundError("Standard MNI152 brain mask file not found in any expected location.")
        
    mask_data = mask_img.get_fdata() > 0
    return mask_img.shape, mask_img.affine, mask_data

def precompute_sphere_offsets(affine, radius_mm=10.0):
    """
    Precomputes the grid of voxel offsets within a physical distance (radius_mm) from the origin.
    Handles general affine transformations.
    """
    # Get voxel sizes (resolutions) along the three axes
    voxel_sizes = nib.affines.voxel_sizes(affine)
    logging.info(f"Detected voxel sizes: {voxel_sizes} mm")
    
    # Calculate search boundary box in voxel coordinate offsets
    max_offsets = np.ceil(radius_mm / voxel_sizes).astype(int)
    logging.info(f"Voxel offset bounding box limits: {max_offsets}")
    
    # Generate relative grid coordinates
    rx = np.arange(-max_offsets[0], max_offsets[0] + 1)
    ry = np.arange(-max_offsets[1], max_offsets[1] + 1)
    rz = np.arange(-max_offsets[2], max_offsets[2] + 1)
    
    grid_i, grid_j, grid_k = np.meshgrid(rx, ry, rz, indexing='ij')
    grid_offsets = np.stack([grid_i.ravel(), grid_j.ravel(), grid_k.ravel()], axis=1)
    
    # Map offsets to physical space using linear part of affine (top-left 3x3)
    phys_offsets = grid_offsets @ affine[:3, :3].T
    dists = np.linalg.norm(phys_offsets, axis=1)
    
    # Keep only those offsets within the physical radius
    valid_offsets = grid_offsets[dists <= radius_mm]
    logging.info(f"Precomputed sphere neighborhood: {len(valid_offsets)} voxels out of {len(grid_offsets)} grid coordinates.")
    
    return valid_offsets

def run_meta_analysis():
    # 1. Standard Space Initialization
    shape, affine, mask_data = load_brain_mask()
    inv_affine = np.linalg.inv(affine)
    
    # Precompute the spherical kernel voxel offsets (r = 10mm)
    sphere_offsets = precompute_sphere_offsets(affine, radius_mm=10.0)
    
    # Load input coordinates
    json_path = "phase3_metadata.json"
    logging.info(f"Loading studies metadata from: {json_path}")
    with open(json_path, 'r') as f:
        studies = json.load(f)
        
    N = len(studies)
    logging.info(f"Total number of studies: {N}")
    
    # 2. Draw Spherical Kernels (Create Bernoulli Trials)
    # Initialize 4D array of shape (X, Y, Z, N) to hold 3D indicator maps for each study
    all_maps = np.zeros(shape + (N,), dtype=np.uint8)
    
    for idx, study in enumerate(studies):
        pmid = study.get("PMID", "Unknown")
        peaks = study.get("Peaks", [])
        
        # If there are no peaks, the map remains all zeros
        if not peaks:
            continue
            
        study_voxels = []
        for peak in peaks:
            x, y, z = peak.get("x"), peak.get("y"), peak.get("z")
            if x is None or y is None or z is None:
                continue
                
            # Convert physical MNI (X, Y, Z) to voxel index (i, j, k)
            coord_homg = np.array([x, y, z, 1.0])
            voxel_idx = np.round(inv_affine @ coord_homg)[:3].astype(int)
            
            # Check if center voxel is inside the 3D bounding box
            if (0 <= voxel_idx[0] < shape[0]) and \
               (0 <= voxel_idx[1] < shape[1]) and \
               (0 <= voxel_idx[2] < shape[2]):
                
                # Center + offsets gives all voxels in the sphere
                sphere_voxels = voxel_idx + sphere_offsets
                study_voxels.append(sphere_voxels)
                
        if study_voxels:
            # Concatenate all sphere voxels for this study
            study_voxels = np.concatenate(study_voxels, axis=0)
            
            # Filter voxels to ensure they lie within the 3D array boundaries
            in_bounds = (
                (study_voxels[:, 0] >= 0) & (study_voxels[:, 0] < shape[0]) &
                (study_voxels[:, 1] >= 0) & (study_voxels[:, 1] < shape[1]) &
                (study_voxels[:, 2] >= 0) & (study_voxels[:, 2] < shape[2])
            )
            valid_voxels = study_voxels[in_bounds]
            
            # Assign binary indicator (overlap capped at 1)
            all_maps[valid_voxels[:, 0], valid_voxels[:, 1], valid_voxels[:, 2], idx] = 1
            
    logging.info("Finished drawing binary spherical kernels for all studies.")
    
    # 3. The Likelihood Summation
    logging.info("Summing maps across the 4th dimension to compute success count k map.")
    k_map = np.sum(all_maps, axis=3)
    
    # 4. Beta-Binomial Conjugate Update
    total_voxels = np.sum(mask_data)
    total_trials = total_voxels * N
    total_hits = np.sum(k_map[mask_data])
    GBR = total_hits / total_trials
    
    W = np.sqrt(N)
    alpha_0 = GBR * W
    beta_0 = (1.0 - GBR) * W
    
    logging.info(f"Calculated Empirical Prior - GBR: {GBR:.8f}, alpha_0: {alpha_0:.6f}, beta_0: {beta_0:.6f}")
    logging.info("Executing Beta-Binomial conjugate update.")
    alpha_post = alpha_0 + k_map
    beta_post = beta_0 + N - k_map
    
    # 5. Posterior Mean Map Calculation
    logging.info("Calculating Posterior Mean Map.")
    posterior_mean = alpha_post / (alpha_post + beta_post)
    posterior_mean = posterior_mean * mask_data
    
    # Export Posterior Mean Map
    mean_filename = "posterior_mean_map.nii.gz"
    logging.info(f"Saving Posterior Mean Map to: {mean_filename}")
    nib.save(nib.Nifti1Image(posterior_mean, affine), mean_filename)
        
    # 6. Map Extraction & Export (Phase 5 with fixed tau = 0.10)
    tau = 0.10
    logging.info(f"Calculating Exceedance Probability Map with fixed threshold tau = {tau:.2f}.")
    exceedance_prob = stats.beta.sf(tau, alpha_post, beta_post) * mask_data
    
    output_filename = "exceedance_probability_map_sparsity_t15.nii.gz"
    logging.info(f"Saving Exceedance Probability Map to: {output_filename}")
    out_img = nib.Nifti1Image(exceedance_prob, affine)
    nib.save(out_img, output_filename)
    
    # 7. Exceedance Probability Decision Rule
    # Create the decision mask by explicitly thresholding the confidence level at 0.95
    decision_mask = (exceedance_prob >= 0.95).astype(int)
    
    # Apply decision mask to both exceedance map and posterior mean map
    logging.info("Applying binary decision mask to enforce threshold.")
    thresholded_exceedance = exceedance_prob * decision_mask
    thresholded_mean = posterior_mean * decision_mask
    
    # Export thresholded maps
    thresholded_exceedance_filename = "thresholded_exceedance_map.nii.gz"
    thresholded_mean_filename = "thresholded_posterior_mean.nii.gz"
    logging.info(f"Saving thresholded maps to: {thresholded_exceedance_filename} and {thresholded_mean_filename}")
    nib.save(nib.Nifti1Image(thresholded_exceedance, affine), thresholded_exceedance_filename)
    nib.save(nib.Nifti1Image(thresholded_mean, affine), thresholded_mean_filename)
    
    logging.info("Successfully completed Exceedance Probability Decision Thresholding.")
    logging.info(f"Posterior success map k ranges from {k_map.min()} to {k_map.max()} active studies per voxel.")
    logging.info(f"Exceedance probability ranges from {exceedance_prob.min():.5f} to {exceedance_prob.max():.5f}.")
    logging.info(f"Thresholded exceedance probability ranges from {thresholded_exceedance.min():.5f} to {thresholded_exceedance.max():.5f}.")
    logging.info(f"Thresholded posterior mean ranges from {thresholded_mean.min():.5f} to {thresholded_mean.max():.5f}.")

if __name__ == "__main__":
    run_meta_analysis()
