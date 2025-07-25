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

Here is the minimum requirement to run the code:

```
conda create -n radar python=3.9.19
conda activate radar
conda install nvidia/label/cuda-12.1.0::cuda-toolkit
conda install pytorch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install transformers==4.41.1
pip install deepspeed==0.14.2
pip install numpy==1.26.3
pip install einops==0.8.0 einops-exts==0.0.4
pip install peft==0.10.0
pip install evaluate==0.4.3
pip install absl-py==2.1.0 rouge-score==0.1.2 nltk==3.8.1
pip install sentencepiece==0.2.0
pip install protobuf==4.25.3
pip install flash-attn==2.5.8 --no-build-isolation
```

## Checkpoints

The checkpoints of this repository are available at [houwenjun060/Radar](https://huggingface.co/houwenjun060/Radar).

## Data Preparation and Preprocessing

### 1. Semi-Structured Report as Knowledge

- Download the CheXbert checkpoint `chexbert.pth` into the `./checkpoints/` folder
- Put `annotation.json` into the `./data/mimic_cxr/` folder
- Run the following script:

```
./script/annotate_reference.sh
```

This produces two annotations:

- `observation.json`: contains the observations of each study
- `sentence_observation.json`: contains sentence-observation paris for each report

### 2. Extract Clinical Context from Datasets

Please follow [MIMIC-Code](https://github.com/MIT-LCP/mimic-code/tree/main/mimic-iv-cxr) to extract clinical context from the MIMIC-CXR dataset.

## Stage I: Preliminary Findings Generation

### Stage 1.1 Fine-tuning BLIP-3 on the MIMIC-CXR dataset

Four parameters are required to run the code of the Stage 1/2:

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
# Generate Preliminary Findings
chmod +x ./script/inference.sh
./script/inference.sh 1 20250727 0

# Annotate Observations of Preliminary Findings
chmod +x ./script/annotate_hypothesis.sh
./script/annotate_hypothesis.sh

# Annotate Observations of CXRs
chmod +x ./script/annotate_image_mimic_cxr.sh
./script/annotate_image_mimic_cxr.sh
```

## Stage II: Supplementary Findings Augmentation

### Stage 2.1 Supplementary Knowledge Retrieval

Run the following scripts to retrieve knowledge for the MIMIC-CXR dataset:

```
chmod +x ./script/retrieve_knowledge_mimic_cxr.sh
./script/retrieve_knowledge_mimic_cxr.sh
```

### Stage 2.2 Supplementary Knowledge Extraction & Enhanced Radiology Report Generation

```
./script/run_mimic_cxr.sh debug date stage topk
```

Example: `./script/run_mimic_cxr.sh 1 20250727 1 2`

## Data Format

We provide [code](data_preparation/README.md) to convert the data format from [Libra](https://github.com/X-iZhang/Libra) to the format used in our work.

### Annotation

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

### Clinical Context

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

- We use the same samples from the MIMIC-CXR dataset as provided by [Libra](https://github.com/X-iZhang/Libra).
