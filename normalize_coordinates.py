import os
import json
import logging
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def get_tal2icbm_matrix():
    """
    Returns the standard Lancaster transform matrix (tal2icbm) to convert
    Talairach coordinates to MNI (ICBM) coordinates.
    The forward icbm_spm matrix transforms MNI -> Talairach.
    The inverse transforms Talairach -> MNI.
    (Lancaster et al., 2007)
    """
    # icbm2tal transformation matrix for SPM
    icbm_spm = np.array([
        [ 0.9254,  0.0024, -0.0118, -1.0207],
        [-0.0048,  0.9316, -0.0871, -1.7667],
        [ 0.0152,  0.0883,  0.8924,  4.0926],
        [ 0.0000,  0.0000,  0.0000,  1.0000]
    ])
    
    # Invert to get tal2icbm
    tal2icbm = np.linalg.inv(icbm_spm)
    return tal2icbm

def convert_tal_to_mni(x, y, z, transform_matrix):
    """
    Applies the 4x4 affine transformation matrix to a 1x3 coordinate.
    """
    # Convert to 1x4 vector (homogenous coordinates)
    coord_vector = np.array([x, y, z, 1.0])
    
    # Compute the dot product
    mni_coord = np.dot(transform_matrix, coord_vector)
    
    # Extract the new X, Y, Z values and round to 2 decimal places
    return round(mni_coord[0], 2), round(mni_coord[1], 2), round(mni_coord[2], 2)

def main():
    input_file = 'phase2_spatial_matrices.json'
    output_file = 'phase2_spatial_matrices.json'
    
    if not os.path.exists(input_file):
        logging.error(f"Input file {input_file} not found. Please run Phase 2 first.")
        return

    with open(input_file, 'r') as f:
        data = json.load(f)

    # Initialize the transformation matrix
    tal2icbm_matrix = get_tal2icbm_matrix()
    
    total_pmids = len(data)
    total_mni = 0
    total_converted = 0

    for entry in data:
        space = entry.get('Space', '').upper() if entry.get('Space') else ''
        
        # If the space is explicitly Talairach, apply the transformation
        if 'TAL' in space or 'TALAIRACH' in space:
            for peak in entry.get('Peaks', []):
                new_x, new_y, new_z = convert_tal_to_mni(peak['x'], peak['y'], peak['z'], tal2icbm_matrix)
                peak['x'] = new_x
                peak['y'] = new_y
                peak['z'] = new_z
                total_converted += 1
            
            # Update the Space audit trail
            entry['Space'] = 'MNI (Converted)'
            
        else:
            # Leave the coordinates as MNI (or unknown spaces which are assumed MNI natively in typical pipelines)
            total_mni += len(entry.get('Peaks', []))

    # Export fully harmonized dataset
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)

    # Console Summary
    logging.info("=" * 40)
    logging.info("Phase 2.5: Spatial Normalization Complete")
    logging.info(f"Total PMIDs processed: {total_pmids}")
    logging.info(f"Total peak coordinates left as MNI: {total_mni}")
    logging.info(f"Total peak coordinates successfully converted from Talairach: {total_converted}")
    logging.info(f"Harmonized data exported to {output_file}")
    logging.info("=" * 40)

if __name__ == "__main__":
    main()
