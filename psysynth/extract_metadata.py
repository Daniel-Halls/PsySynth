import os
import json
import logging
import requests
from bs4 import BeautifulSoup

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def extract_text_from_xml(file_path):
    """
    Parses an XML file and extracts text from the <abstract> and 
    method/participant-related <sec> tags.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f, 'lxml-xml')

    # Extract abstract
    abstract = ""
    abstract_tag = soup.find('abstract')
    if abstract_tag:
        abstract = abstract_tag.get_text(separator=' ', strip=True)

    # Extract methods/participants sections
    methods_text = ""
    for sec in soup.find_all('sec'):
        title = sec.find('title')
        if title:
            title_text = title.get_text().lower()
            # Match keywords typically associated with methods and subjects
            if any(kw in title_text for kw in ['method', 'participant', 'subject', 'procedure', 'material', 'sample']):
                methods_text += " " + sec.get_text(separator=' ', strip=True)

    return f"ABSTRACT:\n{abstract}\n\nMETHODS:\n{methods_text}"

def query_ollama(text, model="llama3"):
    """
    Sends extracted text to a local Ollama instance to extract structured metadata.
    """
    system_prompt = (
               "You are an expert neuroimaging data extractor. "
        "Your task is to analyze the provided text and extract metadata. "
        "You MUST output ONLY a raw JSON object with absolutely no markdown formatting, no backticks, and no conversational filler. "
        "The JSON MUST contain exactly three keys:\n"
        "- \"sample_size\": an integer representing the total sample size (N) of the anorexia group. If unknown, output null.\n"
        "- \"modality\": a string (e.g., \"task-fMRI\", \"rs-fMRI\", \"VBM\"). If unknown, output null.\n"
        "- \"type\": a string classifying the imaging type. It MUST be exactly \"structure\" (e.g., VBM, thickness, volume, gyrification, surface area, DTI, structural MRI), \"function\" (e.g., task-fMRI, rs-fMRI, BOLD, PET, SPECT, ReHo, ALFF, functional MRI), or \"unknown\". Do NOT output any other values for this key.\n"
        "- \"contrast\": a string summarizing the main task contrast, structural comparison, or primary finding. If unknown, output null."
    )

    url = "http://localhost:11434/api/generate"
    
    # We truncate the text slightly to ensure we don't blow past standard context windows
    # 4000 words is a safe margin for 8K context limits.
    truncated_text = " ".join(text.split()[:4000])
    
    payload = {
        "model": model,
        "system": system_prompt,
        "prompt": f"Extract the metadata from the following text:\n\n{truncated_text}",
        "stream": False,
        "format": "json"  # Forces Ollama to strictly adhere to a JSON schema output if supported
    }

    try:
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
        result = response.json()
        
        response_text = result.get('response', '').strip()
        
        # Clean up if the model accidentally included markdown backticks despite instructions
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
            
        return json.loads(response_text.strip())
        
    except requests.exceptions.RequestException as e:
        logging.error(f"Ollama API request failed: {e}")
        return None
    except json.JSONDecodeError as e:
        logging.error(f"Failed to parse LLM JSON output: {e}\nRaw output: {response_text}")
        return None

def main():
    phase2_file = 'phase2_spatial_matrices.json'
    xml_dir = 'xmls'
    output_file = 'phase3_metadata.json'

    # Check dependencies
    if not os.path.exists(phase2_file):
        logging.error(f"Could not find {phase2_file}. Make sure to run Phase 2 first.")
        return

    # Load previously extracted spatial matrices
    with open(phase2_file, 'r') as f:
        phase2_data = json.load(f)

    # Determine unique PMIDs to process
    pmids = set(item['PMID'] for item in phase2_data)
    logging.info(f"Found {len(pmids)} unique PMIDs in Phase 2 results.")

    metadata_cache = {}

    for i, pmid in enumerate(pmids, 1):
        xml_path = os.path.join(xml_dir, f"{pmid}.xml")
        if not os.path.exists(xml_path):
            logging.warning(f"XML file for PMID {pmid} not found at {xml_path}")
            continue

        logging.info(f"[{i}/{len(pmids)}] Processing PMID {pmid} with Ollama...")
        extracted_text = extract_text_from_xml(xml_path)
        
        # Query Ollama
        metadata = query_ollama(extracted_text, model="llama3")
        
        if metadata:
            metadata_cache[pmid] = metadata
            logging.info(f"Extracted: {metadata}")
        else:
            logging.warning(f"Could not extract metadata for PMID {pmid}")
            # Fallback empty structure
            metadata_cache[pmid] = {"sample_size": None, "modality": None, "contrast": None}

    # Merge metadata into the Phase 2 dataset
    final_data = []
    for item in phase2_data:
        pmid = item['PMID']
        meta = metadata_cache.get(pmid, {"sample_size": None, "modality": None, "contrast": None})
        
        new_item = item.copy()
        new_item.update(meta)
        final_data.append(new_item)

    # Export merged results
    with open(output_file, 'w') as f:
        json.dump(final_data, f, indent=4)

    logging.info("=" * 40)
    logging.info("Phase 3 Metadata Extraction Complete")
    logging.info(f"Successfully processed {len(metadata_cache)} documents.")
    logging.info(f"Final merged dataset exported to {output_file}")
    logging.info("=" * 40)

if __name__ == "__main__":
    main()
