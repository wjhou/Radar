python mimic-code/create_section_files.py \
    --reports_path sections \
    --output_path sections

python mimic-code/extract.py \
    --input_file sections/mimic_cxr_sectioned.csv \
    --output_file data/clinical_context.json