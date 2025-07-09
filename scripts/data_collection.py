import pandas as pd
import json
import os
# from pdfplumber import open as pdf_open  # Uncomment if you want to parse PDF text

# Paths
EXCEL_PATH = os.path.join('data', 'sdg_raw', 'Global-Indicator-Framework-after-2025-review-English (1).xlsx')
PDF_PATH = os.path.join('data', 'sdg_raw', '2030Agenda.pdf')
OUTPUT_PATH = os.path.join('data', 'sdg_structured.json')

# 1. Load SDG Indicator List (Excel)
def load_indicator_list(excel_path):
    # Try to find the correct sheet and columns
    xl = pd.ExcelFile(excel_path)
    # Print sheet names for debugging
    print('Available sheets:', xl.sheet_names)
    # Try to load the first sheet (adjust as needed)
    df = xl.parse(xl.sheet_names[0])
    print('Columns:', df.columns)
    return df

# 2. (Optional) Parse SDG Goals/Targets from PDF (placeholder)
def parse_sdg_pdf(pdf_path):
    # Placeholder: In production, use pdfplumber or PyPDF2 to extract text
    # For now, return an empty dict
    return {}

# 3. Structure Data

def structure_sdg_data(df, pdf_data=None):
    # This function assumes the Excel has columns like 'Goal', 'Target', 'Indicator', 'Indicator Description'
    sdg = {}
    for _, row in df.iterrows():
        goal_num = str(row.get('Goal', '')).strip()
        goal_name = row.get('Goal description', '')
        target_code = str(row.get('Target', '')).strip()
        target_desc = row.get('Target description', '')
        indicator_code = str(row.get('Indicator', '')).strip()
        indicator_desc = row.get('Indicator description', '')
        if not goal_num or not target_code or not indicator_code:
            continue
        if goal_num not in sdg:
            sdg[goal_num] = {
                'name': goal_name,
                'targets': {}
            }
        if target_code not in sdg[goal_num]['targets']:
            sdg[goal_num]['targets'][target_code] = {
                'description': target_desc,
                'indicators': []
            }
        sdg[goal_num]['targets'][target_code]['indicators'].append({
            'code': indicator_code,
            'description': indicator_desc
        })
    # Optionally merge in PDF data for richer descriptions
    if pdf_data:
        pass  # TODO: Merge PDF goal/target descriptions
    return sdg

if __name__ == '__main__':
    df = load_indicator_list(EXCEL_PATH)
    pdf_data = parse_sdg_pdf(PDF_PATH)
    sdg_structured = structure_sdg_data(df, pdf_data)
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(sdg_structured, f, indent=2, ensure_ascii=False)
    print(f'Saved structured SDG data to {OUTPUT_PATH}')
