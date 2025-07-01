import transformers
import json
from torch.utils.data import Dataset
from typing import Optional
from PIL import Image
import os
from conversation_utils import (
    preprocess_phi_3_new,
    SYSTEM_PROMPT,
    INSTRUCTION_PROMPT,
    INSTRUCTION_PRIOR_PROMPT,
    OBSERVATION_PROMPT,
)
import torch
import copy

MAX_LEN = 512

CONDITIONS = [
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
    "No Finding",
]


class RadarDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        tokenizer: transformers.PreTrainedTokenizer,
        data_args,
        image_processor,
        split: Optional[str],
        sft_checkpoint=None,
    ):
        super(RadarDataset, self).__init__()
        self.split = split
        self.data_args = data_args
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        stage = data_args.stage
        data = json.load(open(data_path, "r", encoding="utf-8"))

        self.retrieved_ids = {}
        self.retrieval_query = {}
        if stage > 0:
            docs = data["train"]
            self.study2doc = {
                key: docs[key]["findings"] for key in docs if "findings" in docs[key]
            }
            knowledge_path = data_args.knowledge_path.replace(
                "[placeholder]", split.replace("valid", "val")
            )

            for line in open(knowledge_path, "r", encoding="utf-8"):
                line = json.loads(line)
                self.retrieved_ids[line["id"]] = line["retrieved"]
            self.image_annotation = json.load(
                open(data_args.image_annotation_path, "r")
            )

            # Stage 1 checkpoint folder
            previous_query_path = f"{sft_checkpoint}/{split}_results.json"
            self.retrieval_query = {
                key: val["hyp"]
                for key, val in json.load(open(previous_query_path, "r")).items()
            }
            self.retrieval_query_observation = json.load(
                open(f"{sft_checkpoint}/observation.json", "r")
            )
            previous_query_path = f"{sft_checkpoint}/{split}_sentence_observation.json"
            self.query_report2sentence = json.load(open(previous_query_path, "r"))

            self.report2sentence = json.load(
                open(data_args.sentence_knowledge_path, "r")
            )
            self.gt_report_observation = json.load(
                open(data_args.gt_observation_path, "r")
            )

        data = data[split.replace("valid", "val")]
        clinical_context = json.load(open(data_args.clinical_context_path, "r"))
        self.list_data_dict = []

        self.id2report = {
            idx: data[idx]["findings"] for idx in data if "findings" in data[idx]
        }
        for idx in data:
            sample = data[idx]
            report = sample.get("findings", None)
            image_path = sample["image_path"]
            prior_image_path = sample["prior_image_path"]
            study_id = image_path.split("/")[2]
            indications = clinical_context.get(study_id, {})
            if prior_image_path != image_path:
                prior_id = prior_image_path.split("/")[-1].split(".")[0]
                prior_findings = self.id2report.get(prior_id, "")
                indications["Prior Findings"] = prior_findings[:MAX_LEN]
            if report is not None:
                self.list_data_dict.append(
                    {
                        "id": idx,
                        "study": study_id,
                        "image_path": image_path,
                        "report": report,
                        "indications": indications,
                        "prior_image_path": (
                            prior_image_path if prior_image_path != image_path else None
                        ),
                        "task": "RRG",
                    }
                )

            if self.data_args.debug_model:
                if len(self.list_data_dict) >= 32:
                    break

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, idx):
        data_dict = self.list_data_dict[idx]
        return {"RRG": self.getitem_rrg}[data_dict["task"]](data_dict)

    def getitem_rrg(self, data_dict):
        idx = data_dict["id"]
        report = data_dict["report"]

        # truncate the report to 512 characters for training to avoid OOM
        if self.split == "train":
            report = report[:MAX_LEN]

        image_path = data_dict["image_path"]
        raw_image = Image.open(
            os.path.join(self.data_args.image_folder, image_path)
        ).convert("RGB")

        vision_inputs = self.image_processor(
            [raw_image],
            return_tensors="pt",
            image_aspect_ratio="pad",
        ).pixel_values[0]
        has_prior = 0

        prior_vision_inputs = torch.zeros_like(vision_inputs)
        if data_dict["prior_image_path"] is not None:
            has_prior = 1
            prior_image_path = data_dict["prior_image_path"]
            prior_image = Image.open(
                os.path.join(self.data_args.image_folder, prior_image_path)
            ).convert("RGB")

            prior_vision_inputs = self.image_processor(
                [prior_image],
                return_tensors="pt",
                image_aspect_ratio="pad",
            ).pixel_values[0]

        query, report = self.construct_conversation(
            (data_dict["id"], data_dict["study"]),
            data_dict["indications"],
            report,
            stage=self.data_args.stage,
            has_prior=True if has_prior == 1 else False,
        )

        system_round = {
            "from": "system",
            "value": SYSTEM_PROMPT,
        }

        conversation = [
            system_round,
            {
                "from": "human",
                "value": f"<image>\n{query}",
            },
        ]
        if self.split == "train":
            conversation.append({"from": "gpt", "value": report})

        conv = preprocess_phi_3_new([conversation], self.tokenizer)

        input_ids = conv["input_ids"][0]
        if self.split == "train":
            labels = conv["labels"][0]
        else:
            idx = data_dict["id"]
            labels = self.tokenizer(
                [f"{idx}@{report}"],
                return_tensors="pt",
                add_special_tokens=True,
            ).input_ids[0]

        inputs = dict(
            input_ids=input_ids,
            labels=labels,
            pixel_values=vision_inputs,
            prior_pixel_values=prior_vision_inputs,
            has_prior=has_prior,
            padding_side="right" if self.split == "train" else "left",
        )
        return inputs

    def construct_clinical_context(self, indications):
        clinical_context = ""
        context = []
        keys = ("Indication", "History", "Comparison", "Technique", "Prior Findings")
        for key in keys:
            val = indications.get(key, "")
            if len(val) > 0:
                c = f"{key}: {val}"
                context.append(c)
        if len(context) > 0:
            clinical_context = "\n".join(context)
            clinical_context = f"{clinical_context}\n"
        return clinical_context

    def construct_conversation(
        self, idx, indications, report, stage=0, has_prior=False
    ):
        idx, study_id = idx
        clinical_context = self.construct_clinical_context(indications)
        instruction = INSTRUCTION_PRIOR_PROMPT if has_prior else INSTRUCTION_PROMPT
        query = instruction
        if stage <= 0:
            return f"{clinical_context}{query}", report

        retrieval_findings = self.retrieval_query.get(idx, "")
        retrieved_ids = self.retrieved_ids[idx]
        retrieval_docs = [
            self.study2doc[study] for study in retrieved_ids if study in self.study2doc
        ][: self.data_args.topk]

        prel_findings, supp_findings, verified_findings = self.enhance_clinical_context(
            idx,
            retrieval_findings,
            self.retrieval_query_observation.get(idx, []),
            retrieval_docs,
            self.gt_report_observation.get(idx, []),
        )
        retrieval_query = (
            f"Preliminary Findings: {prel_findings}\n" if len(prel_findings) > 0 else ""
        )
        if len(supp_findings) > 0:
            supp_findings = f"Supplementary Findings:\n{supp_findings}\n"
            retrieved_context = f"{retrieval_query}{supp_findings}"
        else:
            retrieved_context = retrieval_query

        query = f"{clinical_context}{retrieved_context}{query} {OBSERVATION_PROMPT}"

        if self.split == "train":
            verified_findings = f"Identified Observations:\n{verified_findings}"
            report = f"{verified_findings}\nOverall Findings:\n{report}"
        return query, report

    def enhance_clinical_context(
        self,
        idx,
        relevant_findings,
        relevant_observations,
        retrieved_findings,
        gt_observations=None,
    ):
        relevant_observations = set(relevant_observations)
        image_observations = set(self.image_annotation.get(idx, []))
        prel_observations = relevant_observations.intersection(image_observations)
        supp_observations = set(CONDITIONS[:-1]) - prel_observations

        prel_findings = self.collect_findings(
            prel_observations,
            [relevant_findings],
            self.query_report2sentence,
        )
        prel_findings = "\n".join(prel_findings)

        supp_findings = self.collect_findings(
            supp_observations,
            retrieved_findings,
            self.report2sentence,
        )
        supp_findings = "\n".join(supp_findings)

        identified_findings = copy.deepcopy(gt_observations)
        identified_findings = sorted(
            identified_findings, key=lambda x: CONDITIONS.index(x)
        )

        identified_findings = "\n".join(identified_findings)
        if len(identified_findings) == 0:
            identified_findings = "N/A"
        return prel_findings, supp_findings, identified_findings

    def collect_findings(self, observations, reports, database):
        findings = []
        for report in reports:
            sentences = database.get(report, [])
            supp_findings = []
            for sentence in sentences:
                sentence_obs = set(sentence["observations"])
                agreement = observations.intersection(sentence_obs)
                if len(agreement) > 0:
                    supp_findings.append(sentence["sentence"])
            supp_findings = " ".join(supp_findings)[:MAX_LEN]
            if len(supp_findings) == 0:
                continue
            findings.append(supp_findings)
        return findings
