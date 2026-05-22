import os
import json
import shutil
from . import run_meta_analysis

# Set FSLDIR so it can find the brain mask if it exists
if "FSLDIR" not in os.environ and os.path.exists("/Users/mszdjh3/fsl"):
    os.environ["FSLDIR"] = "/Users/mszdjh3/fsl"

def main():
    # Load root phase3_metadata.json
    metadata_path = "phase3_metadata.json"
    if not os.path.exists(metadata_path):
        print(f"Error: {metadata_path} not found.")
        return

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    # Split into structure and function
    structure_studies = [s for s in metadata if s.get("type") == "structure"]
    function_studies = [s for s in metadata if s.get("type") == "function"]

    print(f"Found {len(structure_studies)} structure studies and {len(function_studies)} function studies.")

    # Write temporary json files
    with open("structure_metadata.json", "w") as f:
        json.dump(structure_studies, f, indent=4)

    with open("function_metadata.json", "w") as f:
        json.dump(function_studies, f, indent=4)

    # Run meta-analysis for structural
    print("Running meta-analysis for structural studies...")
    run_meta_analysis.run_meta_analysis("structure_metadata.json")

    # Run meta-analysis for functional
    print("Running meta-analysis for functional studies...")
    run_meta_analysis.run_meta_analysis("function_metadata.json")

    # Rename outputs to requested names
    shutil.copy("structure_metadata_thresholded_exceedance_map.nii.gz", "structure_thresholded_exceedance_map.nii.gz")
    shutil.copy("structure_metadata_thresholded_posterior_mean.nii.gz", "structure_thresholded_posterior_mean.nii.gz")
    shutil.copy("function_metadata_thresholded_exceedance_map.nii.gz", "function_thresholded_exceedance_map.nii.gz")
    shutil.copy("function_metadata_thresholded_posterior_mean.nii.gz", "function_thresholded_posterior_mean.nii.gz")

    print("Successfully generated all modality maps!")

if __name__ == "__main__":
    main()
