# Week 1 Research Log

## Daily Progress & Notes
### Day 1 (2024-07-08)
- Set up Python virtual environment and installed all requirements
- Cleaned up the project structure, removed irrelevant files, and organized directories
- Initialized a new GitHub repository (sdg-qa-framework)
- Moved SDG indicator CSV to data/raw/
- Developed and debugged a script to parse SDG indicators and structure them as hierarchical JSON
- Verified and fixed data extraction logic for the CSV format
- Discussed and planned for research logging and methodology documentation

## Methodology Notes
- Used pandas for data parsing and cleaning
- Designed a hierarchical JSON structure: Goal → Target → Indicator
- Parsing logic adapted to handle non-standard SDG indicator CSV format

## Experiments & Tests
- Ran the data extraction script and checked the output for the first few SDGs
- Iteratively debugged parsing logic to ensure correct population of the JSON structure

## Reflections & Next Steps
- The data extraction step is now robust for the provided CSV
- Next: Manual/augmented QA dataset creation and continued research logging 