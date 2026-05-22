import os
import time
import logging
import argparse
import requests
from requests.exceptions import RequestException

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pmc_download.log"),
        logging.StreamHandler()
    ]
)

# Europe PMC rate limit guideline allows up to 10 requests/second.
# We use a conservative 0.3s delay (approx 3 req/sec).
RATE_LIMIT_DELAY = 0.3
SEARCH_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
FULLTEXT_URL_TEMPLATE = "https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"

def search_europe_pmc(query, max_results=None):
    """
    Search Europe PMC and yield tuples of (PMID, PMCID).
    If max_results is None, fetch all matching records.
    """
    params = {
        "query": query,
        "format": "json",
        "cursorMark": "*",
        "pageSize": 1000,
        "resultType": "lite"
    }
    
    count = 0
    while True:
        try:
            response = requests.get(SEARCH_URL, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
        except RequestException as e:
            logging.error(f"Failed to fetch search results: {e}")
            break
            
        results = data.get("resultList", {}).get("result", [])
        if not results:
            break
            
        for item in results:
            pmid = item.get("pmid")
            pmcid = item.get("pmcid")
            if pmid:
                yield pmid, pmcid
                count += 1
                if max_results and count >= max_results:
                    return
        
        next_cursor = data.get("nextCursorMark")
        # Break if pagination is complete or unchanged
        if not next_cursor or next_cursor == params["cursorMark"]:
            break
            
        params["cursorMark"] = next_cursor
        time.sleep(RATE_LIMIT_DELAY)

def download_fulltext_xml(pmcid, pmid, target_dir):
    """
    Download the full-text XML for a given PMCID and save it as {PMID}.xml.
    """
    if not pmcid:
        logging.warning(f"No PMCID available for PMID: {pmid}. Skipping.")
        return False
        
    url = FULLTEXT_URL_TEMPLATE.format(pmcid=pmcid)
    file_path = os.path.join(target_dir, f"{pmid}.xml")
    
    # Skip if file was already downloaded
    if os.path.exists(file_path):
        logging.info(f"File already exists: {file_path}. Skipping.")
        return True
        
    try:
        response = requests.get(url, timeout=20)
        
        # 404 typically means the full-text XML is not available in PMC
        if response.status_code == 404:
            logging.warning(f"Full-text XML not found for PMCID: {pmcid} (PMID: {pmid}).")
            return False
            
        response.raise_for_status()
        
        with open(file_path, "wb") as f:
            f.write(response.content)
            
        logging.info(f"Successfully downloaded full-text for PMID: {pmid} (PMCID: {pmcid}).")
        return True
        
    except RequestException as e:
        logging.error(f"HTTP request failed for PMCID: {pmcid} (PMID: {pmid}). Error: {e}")
        return False
    except IOError as e:
        logging.error(f"File system error saving PMID: {pmid}. Error: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Download Full-Text XMLs from Europe PMC based on a search query.")
    parser.add_argument("--query", required=True, help="Search query string (e.g., 'schizophrenia AND fMRI'). Add 'AND OPEN_ACCESS:Y' to guarantee XML availability.")
    parser.add_argument("--target_dir", required=True, help="Directory to save the XML files.")
    parser.add_argument("--max_results", type=int, default=None, help="Maximum number of articles to process.")
    
    args = parser.parse_args()
    
    # Ensure target directory exists
    os.makedirs(args.target_dir, exist_ok=True)
    
    logging.info(f"Starting PMC XML acquisition.")
    logging.info(f"Query: '{args.query}'")
    logging.info(f"Target Directory: '{args.target_dir}'")
    
    success_count = 0
    failure_count = 0
    
    for pmid, pmcid in search_europe_pmc(args.query, args.max_results):
        # Rate limit before making the individual XML request
        time.sleep(RATE_LIMIT_DELAY)
        if download_fulltext_xml(pmcid, pmid, args.target_dir):
            success_count += 1
        else:
            failure_count += 1
            
    logging.info("="*40)
    logging.info("Acquisition Complete.")
    logging.info(f"Successfully downloaded: {success_count}")
    logging.info(f"Failed or missing: {failure_count}")
    logging.info("="*40)

if __name__ == "__main__":
    main()
