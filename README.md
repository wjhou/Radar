# <img src="figure/radar.png?raw=true" alt="Alt" height="38" style="vertical-align:middle;"> <span style="font-variant:small-caps;">RADAR</span>: Enhancing Radiology Report Generation with Supplementary Knowledge Injection

This repository is the implementation of [RADAR: Enhancing Radiology Report Generation with Supplementary Knowledge Injection](https://arxiv.org/abs/2505.14318). Before running the code, please install the prerequisite libraries, and follow our instructions to replicate the experiments.

## Overview

Large language models (LLMs) have demonstrated remarkable capabilities in various domains, including radiology report generation. Previous approaches have attempted to utilize multimodal LLMs for this task, enhancing their performance through the integration of domain-specific knowledge retrieval. However, these approaches often overlook the knowledge already embedded within the LLMs, leading to redundant information integration. To address this limitation, we propose Radar, a framework for enhancing radiology report generation with supplementary knowledge injection. Radar improves report generation by systematically leveraging both the internal knowledge of an LLM and externally retrieved information. Specifically, it first extracts the model's acquired knowledge that aligns with expert image-based classification outputs. It then retrieves relevant supplementary knowledge to further enrich this information. Finally, by aggregating both sources, Radar generates more accurate and informative radiology reports. Extensive experiments on MIMIC-CXR, CheXpert-Plus, and IU X-ray demonstrate that our model outperforms state-of-the-art LLMs in both language quality and clinical accuracy
![Alt text](figure/framework.png?raw=true "Title")

## Requirements

### Basic Requirements

- `python>=3.9.0`
- `torch==2.2.1`
- `transformers==4.41.1`

### Other Requirements

Please install dependencies by using the following command:

```
conda env create -f environment.yml # Untested
conda activate radar
```

## Data Preparation and Preprocessing

### 1. Semi-Structured Report as Knowledge

Please specific `folder` that contains `annotation.json`

```
./script/annotate.sh folder
```

Example: `./script/annotate.sh ./data/mimic_cxr`

### 2. Extract Clinical Context from Datasets

Please follow [MIMIC-Code](https://github.com/MIT-LCP/mimic-code/tree/main/mimic-iv-cxr) to extract clinical context from the MIMIC-CXR dataset.

In addition, we provide the files of CheXpert Plus and IU X-ray in the `data` folder.

## Stage I: Preliminary Findings Generation

### Stage 1.1 Fine-tuning BLIP-3 on the MIMIC-CXR dataset

Four parameters are required to run the code of the Stage 2:

- debug: whether debugging the code (0 for debugging and 1 for running)
- date: date of running the code (checkpoint identifier)
- stage: stage of current fune-tuning (0 for Stage I and 1 for Stage II)
- topk: number of retrieved knowledge snippets

```
./script/run_mimic_cxr.sh debug date stage topk
```

Example: `./script/run_mimic_cxr.sh 0 20250727 0 0`

### Stage 1.2 Preliminary Findings Extraction with CheXbert

Generate Findings and Annotate Observations with [CheXbert](https://github.com/stanfordmlgroup/CheXbert):

```
./script/inference.sh 1 20250727 0 # Generate Findings
./script/annotate_pf.sh # Annotate Observations of Preliminary Findings
```

## Stage II: Supplementary Findings Augmentation

### Stage 2.1 Supplementary Knowledge Retrieval

```
./retrieval/retrieve.sh folder
```

### Stage 2.2 Supplementary Knowledge Extraction & Enhanced Radiology Report Generation

```
./script/run_mimic_cxr.sh debug date topk
```

Example: `./script/run_mimic_cxr.sh 1 20250727 2`

### Data Format

#### Annotation

```json
{
    "train": {
        "02aa804e-bde0afdd-112c0b34-7bc16630-4e384014": {
            "image_path": "p10/p10000032/s50414267/02aa804e-bde0afdd-112c0b34-7bc16630-4e384014.jpg",
            "prior_image_path": "......",
            "findings": "......"
        },
        ......
    },
    "val": {},
    "test": {}
}
```

#### Clinical Context

```json
{
    "s50414267": {
        "Indication": "......",
        "History": "......",
        "Comparison": "......",
        "Technique": "......",
    },
    ......
}
```

## Citation

If you use the Radar, please cite our paper:

```bibtex
@misc{hou2025radarenhancingradiologyreport,
      title={RADAR: Enhancing Radiology Report Generation with Supplementary Knowledge Injection},
      author={Wenjun Hou and Yi Cheng and Kaishuai Xu and Heng Li and Yan Hu and Wenjie Li and Jiang Liu},
      year={2025},
      eprint={2505.14318},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2505.14318},
}
```

## Acknowledges

- We use the same samples of the MIMIC-CXR dataset provided by [Libra](https://github.com/X-iZhang/Libra).
