import pandas as pd
import json
import os

CSV_PATH = os.path.join('data', 'raw', 'Global-Indicator-Framework-after-2025-review-English - A.RES.71.313 Annex.csv')
OUTPUT_PATH = os.path.join('data', 'sdg_structured.json')

def load_indicator_list(csv_path):
    df = pd.read_csv(csv_path, skiprows=2, header=0, usecols=[1, 2, 3])
    df.columns = ['goal_target', 'indicator', 'unsd_code']
    return df

def structure_sdg_data(df):
    sdg = {}
    current_goal = None
    current_goal_name = None
    current_target = None
    for _, row in df.iterrows():
        goal_target = str(row['goal_target']).strip() if not pd.isna(row['goal_target']) else ''
        indicator = str(row['indicator']).strip() if not pd.isna(row['indicator']) else ''
        unsd_code = str(row['unsd_code']).strip() if not pd.isna(row['unsd_code']) else ''

        # Detect new Goal
        if goal_target.startswith('Goal'):
            parts = goal_target.split('.', 1)
            if len(parts) == 2:
                current_goal = parts[0].replace('Goal', '').strip()
                current_goal_name = parts[1].strip()
                sdg[current_goal] = {
                    'name': current_goal_name,
                    'targets': {}
                }
            continue

        # Detect new Target
        if goal_target and not goal_target.startswith('Goal'):
            target_code = goal_target.split(' ', 1)[0]
            target_desc = goal_target[len(target_code):].strip()
            current_target = target_code
            if current_goal and current_target:
                sdg[current_goal]['targets'][current_target] = {
                    'description': target_desc,
                    'indicators': []
                }
            continue

        # Detect Indicator
        if not goal_target and indicator:
            ind_parts = indicator.split(' ', 1)
            if len(ind_parts) == 2:
                ind_code = ind_parts[0]
                ind_desc = ind_parts[1]
                if current_goal and current_target:
                    sdg[current_goal]['targets'][current_target]['indicators'].append({
                        'code': ind_code,
                        'description': ind_desc,
                        'unsd_code': unsd_code
                    })
            continue
    return sdg

if __name__ == '__main__':
    df = load_indicator_list(CSV_PATH)
    sdg_structured = structure_sdg_data(df)
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(sdg_structured, f, indent=2, ensure_ascii=False)
    print(f'Saved structured SDG data to {OUTPUT_PATH}')
