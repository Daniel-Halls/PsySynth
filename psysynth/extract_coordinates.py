import os
import re
import json
import logging
from bs4 import BeautifulSoup

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def clean_text_for_numbers(text):
    """
    Cleans typographic noise and extracts all numeric values (integers or floats).
    Converts en-dashes and typographic minus signs to standard hyphens.
    """
    if not text:
        return []
    text = text.replace('–', '-').replace('−', '-')
    matches = re.findall(r'-?\d+\.?\d*', text)
    return [float(m) if '.' in m else int(m) for m in matches]

def get_normalized_headers(rows):
    """
    Handles rowspans and colspans in table headers to generate a flat list of 
    column headers that correctly map 1:1 with the data columns.
    """
    max_cols = 0
    for r in rows:
        cols = sum(int(cell.get('colspan', 1)) for cell in r.find_all(['th', 'td']))
        if cols > max_cols: 
            max_cols = cols
            
    grid = [[''] * max_cols for _ in range(len(rows))]
    
    for r_idx, r in enumerate(rows):
        c_idx = 0
        for cell in r.find_all(['th', 'td']):
            while c_idx < max_cols and grid[r_idx][c_idx] != '':
                c_idx += 1
            if c_idx >= max_cols: 
                break
                
            colspan = int(cell.get('colspan', 1))
            rowspan = int(cell.get('rowspan', 1))
            text = cell.get_text(separator=' ').strip().lower().replace('-', ' ').replace('/', ' ')
            
            for i in range(rowspan):
                for j in range(colspan):
                    if r_idx + i < len(rows) and c_idx + j < max_cols:
                        grid[r_idx + i][c_idx + j] = text
            c_idx += colspan
            
    final_headers = []
    for c in range(max_cols):
        # Merge the vertical columns to create a single comprehensive header
        h = " ".join(dict.fromkeys([grid[r][c] for r in range(len(rows)) if grid[r][c]]))
        final_headers.append(h)
        
    return final_headers

def classify_columns(headers):
    """
    Given a list of normalized headers, identifies the indices for X, Y, Z and Stat.
    """
    indices = {'x': -1, 'y': -1, 'z': -1, 'xyz': -1, 'stat': -1, 'stat_type': 'unknown'}
    
    # Exact word match for x, y, z to prevent overlap
    x_cands = [i for i, h in enumerate(headers) if re.search(r'\b(x)\b', h) and not re.search(r'\b(y)\b', h)]
    y_cands = [i for i, h in enumerate(headers) if re.search(r'\b(y)\b', h) and not re.search(r'\b(x)\b', h)]
    z_cands = [i for i, h in enumerate(headers) if re.search(r'\b(z)\b', h) and not re.search(r'\b(x)\b', h)]
    
    if x_cands and y_cands and z_cands:
        indices['x'] = x_cands[-1]
        indices['y'] = y_cands[-1]
        indices['z'] = z_cands[-1]
        
        # If there are multiple 'z' columns, one is likely the z-statistic
        for z_idx in z_cands:
            if z_idx != indices['z']:
                indices['stat'] = z_idx
                indices['stat_type'] = 'z'
                break
    else:
        # Check for combined xyz coordinate column
        for i, h in enumerate(headers):
            if re.search(r'\b(coordinates|coord|mni|talairach|tal|x y z|xyz)\b', h):
                indices['xyz'] = i
                break

    # Identify stat columns if not already found
    for i, h in enumerate(headers):
        if i in [indices['x'], indices['y'], indices['z'], indices['xyz']]:
            continue
            
        if re.search(r'\b(t)\b|\bt value\b|\bt score\b|\btmax\b|\bpeak t\b', h):
            indices['stat'] = i
            indices['stat_type'] = 't'
            break
        elif re.search(r'\b(z)\b|\bz value\b|\bz score\b|\bzmax\b|\bpeak z\b', h):
            if indices['stat'] == -1: 
                indices['stat'] = i
                indices['stat_type'] = 'z'
            break
        elif re.search(r'\b(f)\b|\bf value\b|\bf score\b|\bfmax\b|\bpeak f\b', h):
            indices['stat'] = i
            indices['stat_type'] = 'f'
            break
        elif re.search(r'\b(statistic|stat|peak|max|maximum)\b', h):
            if indices['stat'] == -1:
                indices['stat'] = i
                indices['stat_type'] = 'unknown'

    return indices

def parse_xml_for_coordinates_and_stats(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f, 'lxml-xml')

    pmid = os.path.splitext(os.path.basename(file_path))[0]
    results = []

    for table_wrap in soup.find_all(['table-wrap', 'table']):
        text_content = table_wrap.get_text().lower()
        space = None
        if 'mni' in text_content:
            space = 'MNI'
        elif 'talairach' in text_content or 'tal' in text_content:
            space = 'Talairach'

        table = table_wrap.find('table')
        if not table:
            table = table_wrap if table_wrap.name == 'table' else None
        if not table:
            continue

        thead = table.find('thead')
        rows = thead.find_all('tr') if thead else table.find_all('tr')[:2]
        
        headers = get_normalized_headers(rows)
        indices = classify_columns(headers)
        
        has_coords = (indices['x'] != -1 and indices['y'] != -1 and indices['z'] != -1) or indices['xyz'] != -1
        
        # We can relax the strict requirement on `indices['stat'] != -1` slightly, 
        # because the stat might be embedded directly next to the coordinate.
        if not has_coords:
            continue

        tbody = table.find('tbody')
        body_rows = tbody.find_all('tr') if tbody else table.find_all('tr')[len(rows):]
        
        peaks = []
        for tr in body_rows:
            cells = tr.find_all(['th', 'td'])
            if not cells:
                continue
            
            # Normalize row cells to account for colspans 
            # (though rare in data bodies, it ensures 1:1 mapping with headers)
            cell_texts = []
            for c in cells:
                colspan = int(c.get('colspan', 1))
                text = c.get_text().strip()
                cell_texts.extend([text] * colspan)
            
            coords = []
            
            # Scenario A: Combined XYZ
            if indices['xyz'] != -1 and indices['xyz'] < len(cell_texts):
                raw = cell_texts[indices['xyz']]
                nums = clean_text_for_numbers(raw)
                if len(nums) >= 3:
                    coords = nums[:3]
            
            # Scenario B: Separate X, Y, Z
            elif indices['x'] != -1 and indices['y'] != -1 and indices['z'] != -1:
                if max(indices['x'], indices['y'], indices['z']) < len(cell_texts):
                    x_n = clean_text_for_numbers(cell_texts[indices['x']])
                    y_n = clean_text_for_numbers(cell_texts[indices['y']])
                    z_n = clean_text_for_numbers(cell_texts[indices['z']])
                    if x_n and y_n and z_n:
                        coords = [x_n[0], y_n[0], z_n[0]]
            
            # Extract Statistic Value
            stat_val = None
            stat_type = indices['stat_type']
            
            # Try to get stat from designated column
            if indices['stat'] != -1 and indices['stat'] < len(cell_texts):
                raw_stat_text = cell_texts[indices['stat']]
                s_nums = clean_text_for_numbers(raw_stat_text)
                if s_nums:
                    stat_val = s_nums[0]
                    
                # The cell itself might declare the type (e.g. "t=4.5")
                lower_text = raw_stat_text.lower()
                if re.search(r'\bt\s*=', lower_text): stat_type = 't'
                elif re.search(r'\bz\s*=', lower_text): stat_type = 'z'
                elif re.search(r'\bf\s*=', lower_text): stat_type = 'f'
                
            # Fallback: scan all OTHER cells for embedded stats like "t=4.5"
            if stat_val is None:
                for i, raw_text in enumerate(cell_texts):
                    if i in [indices['x'], indices['y'], indices['z'], indices['xyz']]:
                        continue
                    lower_text = raw_text.lower()
                    if re.search(r'\b(t|z|f)\s*=\s*-?\d+', lower_text):
                        if 't' in lower_text: stat_type = 't'
                        elif 'z' in lower_text: stat_type = 'z'
                        elif 'f' in lower_text: stat_type = 'f'
                        
                        s_nums = clean_text_for_numbers(raw_text)
                        if s_nums:
                            stat_val = s_nums[0]
                            break

            if len(coords) == 3 and stat_val is not None:
                peaks.append({
                    "x": float(coords[0]),
                    "y": float(coords[1]),
                    "z": float(coords[2]),
                    "stat_type": stat_type,
                    "stat_value": float(stat_val)
                })
                
        if peaks:
            results.append({
                'PMID': pmid,
                'Space': space,
                'Peaks': peaks
            })

    return results

def main():
    xml_dir = 'xmls'
    if not os.path.exists(xml_dir):
        logging.error(f"Target directory '{xml_dir}' not found.")
        return

    all_results = []
    processed_tables = 0
    success_pmids = set()

    for filename in os.listdir(xml_dir):
        if not filename.endswith('.xml'):
            continue
            
        file_path = os.path.join(xml_dir, filename)
        try:
            results = parse_xml_for_coordinates_and_stats(file_path)
            for res in results:
                processed_tables += 1
                success_pmids.add(res['PMID'])
                all_results.append(res)
        except Exception as e:
            logging.error(f"Error parsing {filename}: {e}")

    out_file = 'phase2_spatial_matrices_v2.json'
    with open(out_file, 'w') as f:
        json.dump(all_results, f, indent=4)

    logging.info("=" * 40)
    logging.info("Phase 2 Extraction (v3 logic) Complete")
    logging.info(f"PMIDs successfully yielding coordinates AND statistics: {len(success_pmids)}")
    logging.info(f"Total valid coordinate tables processed: {processed_tables}")
    logging.info(f"Structured data exported to {out_file}")
    logging.info("=" * 40)

if __name__ == "__main__":
    main()
