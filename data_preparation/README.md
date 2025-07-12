# Data Preparation

## Prepare `annotation.json`

To run the annotation format transformation code, specify the input file paths (e.g., `libra_findings_section_train.json`, `libra_findings_section_valid.json`, and `libra_findings_section_eval.json`) and the output file path (e.g., `annotation.json`).

Here is an example:

```
python annotation_libra2radar.py \
    data/libra_findings_section_train.json \
    data/libra_findings_section_valid.json \
    data/libra_findings_section_eval.json \
    data/annotation.json
```

A script for running the above code is also provided:

```
chmod +x script/run_annotation.sh
./script/run_annotation.sh
```

## Prepare `clinical_context.json`

To extract clinical context from MIMIC-CXR data:

- Download MIMIC-CXR sections from [MIMIC-CXR](https://physionet.org/content/mimic-cxr/2.1.0/)
- Specify `--reports_path` (e.g., `sections`)
- Specify `--output_path` (e.g., `sections`)

```
python mimic-code/create_section_files.py --reports_path sections --output_path sections
```

The above command will produce `sections/mimic_cxr_sectioned.csv`. You can then transform it into the required format using:

```
python mimic-code/extract.py --input_file sections/mimic_cxr_sectioned.csv --output_file data/clinical_context.json
```

A script to run the above code is also provided:

```
chmod +x script/run_clinical_context.sh
./script/run_clinical_context.sh
```
