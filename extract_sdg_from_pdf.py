#!/usr/bin/env python3
"""
Extract SDG information from the 2030 Agenda PDF and create a structured dataset.

This script extracts SDG goals, targets, and indicators from the official 
2030 Agenda for Sustainable Development PDF document.
"""

import pdfplumber
import PyPDF2
import json
import pandas as pd
import re
from pathlib import Path
import argparse


def extract_text_from_pdf(pdf_path):
    """Extract text from PDF using pdfplumber."""
    print(f"Extracting text from {pdf_path}")
    
    text_content = []
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            print(f"PDF has {len(pdf.pages)} pages")
            
            for page_num, page in enumerate(pdf.pages, 1):
                if page_num % 10 == 0:
                    print(f"Processing page {page_num}...")
                
                text = page.extract_text()
                if text:
                    text_content.append({
                        'page': page_num,
                        'text': text.strip()
                    })
    
    except Exception as e:
        print(f"Error with pdfplumber: {e}")
        # Fallback to PyPDF2
        return extract_text_pypdf2(pdf_path)
    
    return text_content


def extract_text_pypdf2(pdf_path):
    """Fallback method using PyPDF2."""
    print("Using PyPDF2 as fallback...")
    
    text_content = []
    
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            print(f"PDF has {len(pdf_reader.pages)} pages")
            
            for page_num, page in enumerate(pdf_reader.pages, 1):
                if page_num % 10 == 0:
                    print(f"Processing page {page_num}...")
                
                text = page.extract_text()
                if text:
                    text_content.append({
                        'page': page_num,
                        'text': text.strip()
                    })
    
    except Exception as e:
        print(f"Error with PyPDF2: {e}")
        return []
    
    return text_content


def parse_sdg_content(text_content):
    """Parse and structure SDG content from extracted text."""
    print("Parsing SDG content...")
    
    sdgs = {}
    current_sdg = None
    current_target = None
    
    # Combined text for pattern matching
    full_text = "\n".join([page['text'] for page in text_content])
    
    # Patterns for different SDG elements
    sdg_pattern = r'Goal\s+(\d+)\.?\s*([^\n]+)'
    target_pattern = r'(\d+\.\d+)\s+(.+?)(?=\d+\.\d+|\nGoal|\n\n|$)'
    indicator_pattern = r'(\d+\.\d+\.\d+)\s+(.+?)(?=\d+\.\d+\.\d+|\d+\.\d+|\nGoal|\n\n|$)'
    
    # Find all SDG goals
    sdg_matches = re.finditer(sdg_pattern, full_text, re.IGNORECASE | re.MULTILINE)
    
    for sdg_match in sdg_matches:
        sdg_num = int(sdg_match.group(1))
        sdg_title = sdg_match.group(2).strip()
        
        if sdg_num <= 17:  # Only process SDGs 1-17
            sdgs[sdg_num] = {
                'number': sdg_num,
                'title': sdg_title,
                'targets': {},
                'full_text': '',
                'page_numbers': []
            }
    
    # Extract more detailed content for each SDG
    for page_data in text_content:
        page_text = page_data['text']
        page_num = page_data['page']
        
        # Check which SDG this page might belong to
        for sdg_num in sdgs.keys():
            if f"Goal {sdg_num}" in page_text or f"SDG {sdg_num}" in page_text:
                sdgs[sdg_num]['full_text'] += f"\n{page_text}"
                if page_num not in sdgs[sdg_num]['page_numbers']:
                    sdgs[sdg_num]['page_numbers'].append(page_num)
    
    # Find targets and indicators
    target_matches = re.finditer(target_pattern, full_text, re.DOTALL)
    
    for target_match in target_matches:
        target_num = target_match.group(1)
        target_text = target_match.group(2).strip()
        
        # Determine which SDG this target belongs to
        sdg_num = int(target_num.split('.')[0])
        
        if sdg_num in sdgs:
            sdgs[sdg_num]['targets'][target_num] = {
                'number': target_num,
                'description': target_text,
                'indicators': {}
            }
    
    # Find indicators
    indicator_matches = re.finditer(indicator_pattern, full_text, re.DOTALL)
    
    for indicator_match in indicator_matches:
        indicator_num = indicator_match.group(1)
        indicator_text = indicator_match.group(2).strip()
        
        # Determine which SDG and target this indicator belongs to
        parts = indicator_num.split('.')
        if len(parts) >= 2:
            sdg_num = int(parts[0])
            target_num = f"{parts[0]}.{parts[1]}"
            
            if sdg_num in sdgs and target_num in sdgs[sdg_num]['targets']:
                sdgs[sdg_num]['targets'][target_num]['indicators'][indicator_num] = {
                    'number': indicator_num,
                    'description': indicator_text
                }
    
    return sdgs


def create_datasets(sdgs, output_dir):
    """Create different dataset formats from the parsed SDG data."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # 1. Full structured SDG dataset (JSON)
    structured_data = {}
    for sdg_num, sdg_data in sdgs.items():
        structured_data[str(sdg_num)] = {
            'number': sdg_data['number'],
            'title': sdg_data['title'],
            'targets': {},
            'page_numbers': sdg_data['page_numbers']
        }
        
        for target_num, target_data in sdg_data['targets'].items():
            structured_data[str(sdg_num)]['targets'][target_num] = {
                'number': target_data['number'],
                'description': target_data['description'],
                'indicators': target_data['indicators']
            }
    
    json_path = output_path / "sdg_2030_agenda_structured.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(structured_data, f, indent=2, ensure_ascii=False)
    print(f"Saved structured SDG data to: {json_path}")
    
    # 2. Flat dataset for ML training (CSV)
    ml_data = []
    
    # SDG goals
    for sdg_num, sdg_data in sdgs.items():
        ml_data.append({
            'type': 'goal',
            'sdg_number': sdg_num,
            'identifier': f"Goal {sdg_num}",
            'text': sdg_data['title'],
            'parent_sdg': sdg_num,
            'parent_target': None,
            'level': 'goal'
        })
    
    # Targets
    for sdg_num, sdg_data in sdgs.items():
        for target_num, target_data in sdg_data['targets'].items():
            ml_data.append({
                'type': 'target',
                'sdg_number': sdg_num,
                'identifier': target_num,
                'text': target_data['description'],
                'parent_sdg': sdg_num,
                'parent_target': target_num,
                'level': 'target'
            })
            
            # Indicators
            for indicator_num, indicator_data in target_data['indicators'].items():
                ml_data.append({
                    'type': 'indicator',
                    'sdg_number': sdg_num,
                    'identifier': indicator_num,
                    'text': indicator_data['description'],
                    'parent_sdg': sdg_num,
                    'parent_target': target_num,
                    'level': 'indicator'
                })
    
    # Convert to DataFrame and save
    ml_df = pd.DataFrame(ml_data)
    csv_path = output_path / "sdg_2030_agenda_ml_dataset.csv"
    ml_df.to_csv(csv_path, index=False, encoding='utf-8')
    print(f"Saved ML dataset to: {csv_path}")
    
    # 3. Text chunks for embedding/similarity tasks
    text_chunks = []
    chunk_id = 0
    
    for sdg_num, sdg_data in sdgs.items():
        # Goal text
        chunk_id += 1
        text_chunks.append({
            'chunk_id': chunk_id,
            'sdg_number': sdg_num,
            'type': 'goal',
            'identifier': f"Goal {sdg_num}",
            'text': sdg_data['title'],
            'word_count': len(sdg_data['title'].split()),
            'char_count': len(sdg_data['title'])
        })
        
        # Target texts
        for target_num, target_data in sdg_data['targets'].items():
            chunk_id += 1
            text_chunks.append({
                'chunk_id': chunk_id,
                'sdg_number': sdg_num,
                'type': 'target',
                'identifier': target_num,
                'text': target_data['description'],
                'word_count': len(target_data['description'].split()),
                'char_count': len(target_data['description'])
            })
            
            # Indicator texts
            for indicator_num, indicator_data in target_data['indicators'].items():
                chunk_id += 1
                text_chunks.append({
                    'chunk_id': chunk_id,
                    'sdg_number': sdg_num,
                    'type': 'indicator',
                    'identifier': indicator_num,
                    'text': indicator_data['description'],
                    'word_count': len(indicator_data['description'].split()),
                    'char_count': len(indicator_data['description'])
                })
    
    # Save text chunks
    chunks_df = pd.DataFrame(text_chunks)
    chunks_path = output_path / "sdg_2030_agenda_text_chunks.csv"
    chunks_df.to_csv(chunks_path, index=False, encoding='utf-8')
    print(f"Saved text chunks to: {chunks_path}")
    
    # 4. Statistics
    stats = {
        'total_sdgs': len(sdgs),
        'total_targets': sum(len(sdg_data['targets']) for sdg_data in sdgs.values()),
        'total_indicators': sum(
            sum(len(target_data['indicators']) for target_data in sdg_data['targets'].values())
            for sdg_data in sdgs.values()
        ),
        'total_text_chunks': len(text_chunks),
        'sdg_breakdown': {}
    }
    
    for sdg_num, sdg_data in sdgs.items():
        stats['sdg_breakdown'][f'sdg_{sdg_num}'] = {
            'title': sdg_data['title'],
            'targets_count': len(sdg_data['targets']),
            'indicators_count': sum(len(target_data['indicators']) for target_data in sdg_data['targets'].values()),
            'pages': sdg_data['page_numbers']
        }
    
    stats_path = output_path / "sdg_2030_agenda_stats.json"
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"Saved statistics to: {stats_path}")
    
    return stats


def main():
    parser = argparse.ArgumentParser(description='Extract SDG data from 2030 Agenda PDF')
    parser.add_argument('--input', '-i', 
                        default='data/raw/2030_Agenda.pdf',
                        help='Input PDF file path')
    parser.add_argument('--output', '-o', 
                        default='data/processed',
                        help='Output directory')
    
    args = parser.parse_args()
    
    print("=== SDG 2030 Agenda PDF Extractor ===")
    print(f"Input PDF: {args.input}")
    print(f"Output directory: {args.output}")
    print()
    
    # Extract text from PDF
    text_content = extract_text_from_pdf(args.input)
    
    if not text_content:
        print("ERROR: Could not extract text from PDF")
        return
    
    print(f"Extracted text from {len(text_content)} pages")
    
    # Parse SDG content
    sdgs = parse_sdg_content(text_content)
    
    if not sdgs:
        print("WARNING: No SDG content found in PDF")
        return
    
    print(f"Found {len(sdgs)} SDGs")
    
    # Create datasets
    stats = create_datasets(sdgs, args.output)
    
    print("\n=== Extraction Summary ===")
    print(f"Total SDGs: {stats['total_sdgs']}")
    print(f"Total Targets: {stats['total_targets']}")
    print(f"Total Indicators: {stats['total_indicators']}")
    print(f"Total Text Chunks: {stats['total_text_chunks']}")
    
    print("\n=== Extraction completed successfully! ===")


if __name__ == "__main__":
    main()
