import json
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn.functional as F
from PIL import Image
from modeling_expert_model import ExpertModel
from safetensors.torch import load_file
from tqdm import tqdm
import os
from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    AutoTokenizer,
    AutoModel,
)
import argparse


os.environ["TOKENIZERS_PARALLELISM"] = "false"
cp_class_names = [
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


def construct_clinical_context(indications):
    clinical_context = ""
    context = []
    keys = (
        "Indication",
        "History",
        "Comparison",
        "Technique",
        "Prior Findings",
    )
    for key in keys:
        val = indications.get(key, "")
        if len(val) > 0:
            c = f"{key}: {val}"
            context.append(c)
    if len(context) > 0:
        clinical_context = "\n".join(context)
    return clinical_context


class Dataset(torch.utils.data.Dataset):
    def __init__(self, root_path, idxs, image_paths, hyps, tokenizer, processor):
        self.root_path = root_path
        self.idxs = idxs
        self.image_paths = image_paths
        self.hyps = hyps
        self.tokenizer = tokenizer
        self.processor = processor

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        study_idx = image_path.split("/")[2]
        image_path = Path(self.root_path) / image_path
        image = Image.open(image_path).convert("RGB")
        image = processor(images=image, return_tensors="pt")["pixel_values"]
        hyp = self.hyps[idx]
        return image, hyp, idx, self.idxs[idx], study_idx


def collate_fn(batch):
    images, hyps, idxs, idxs_, study_idxs = zip(*batch)
    images = torch.cat(images)
    return images, hyps, idxs, idxs_, study_idxs


def func(
    root_path,
    annotation_path,
    output_path,
    clinical_context_path,
    gt_observation_path,
):
    gt_observation = json.load(open(gt_observation_path, "r"))
    annotation = json.load(open(annotation_path, "r"))
    clinical_contexts = json.load(open(clinical_context_path, "r"))
    image_paths = {}
    hyps = {}
    refs = {}
    idxs = {}
    target_ids = []
    gt_masks = {}
    for split in annotation:
        hyps[split] = []
        refs[split] = []
        image_paths[split] = []
        idxs[split] = []
        data = annotation[split]
        id2report = {
            idx: data[idx]["findings"] for idx in data if "findings" in data[idx]
        }
        for idx in tqdm(annotation[split], desc=split, total=len(annotation[split])):
            if "findings" not in annotation[split][idx]:
                continue
            image_path = (
                annotation[split][idx]["image_path"]
                if "chexpert_plus" not in root_path
                else annotation[split][idx]["image_path"].replace(".jpg", ".png")
            )
            image_paths[split].append(image_path)
            idxs[split].append(idx)

            study_id = (
                annotation[split][idx]["image_path"].split("/")[2]
                if "iu_xray" not in image_path and "chexpert" not in image_path
                else idx
            )
            clinical_context = clinical_contexts.get(study_id, {})
            sample = annotation[split][idx]
            image_path = sample["image_path"]
            prior_image_path = sample["prior_image_path"]
            if prior_image_path != image_path:
                prior_id = prior_image_path.split("/")[-1].split(".")[0]
                prior_findings = id2report.get(prior_id, "")
                clinical_context["Prior Findings"] = prior_findings[:512]
            hyps[split].append(construct_clinical_context(clinical_context))

            if split == "train":
                mask = [0] * 14
                gt_obs = gt_observation.get(idx, [])
                for o in gt_obs:
                    mask[cp_class_names.index(o)] = 1
                gt_masks[idx] = mask

    train_dataset = Dataset(
        root_path,
        idxs["train"],
        image_paths["train"],
        hyps["train"],
        tokenizer,
        processor,
    )
    val_dataset = Dataset(
        root_path,
        idxs["val"],
        image_paths["val"],
        hyps["val"],
        tokenizer,
        processor,
    )
    test_dataset = Dataset(
        root_path,
        idxs["test"],
        image_paths["test"],
        hyps["test"],
        tokenizer,
        processor,
    )

    batch_size = 128
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=16,
        collate_fn=collate_fn,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=16,
        collate_fn=collate_fn,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=16,
        collate_fn=collate_fn,
    )
    data_loaders = {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
    }
    model.cuda()
    model.eval()
    model.half()

    target_logits = []
    all_logits = {}
    all_embeds = {}
    all_idxs = {}
    all_study_idxs = {}
    target_logits = []
    for split in image_paths:
        all_logits[split] = []
        all_embeds[split] = []
        all_idxs[split] = []
        all_study_idxs[split] = []

        total = len(image_paths[split]) // batch_size
        if len(image_paths[split]) % batch_size != 0:
            total += 1

        data_loader = data_loaders[split]
        for i, batch in tqdm(enumerate(data_loader), total=total):
            (
                images,
                batch_hyps,
                batch_idxs,
                batch_idxs_,
                batch_study_idxs,
            ) = batch

            text_inputs = tokenizer(
                batch_hyps,
                padding="longest",
                max_length=512,
                return_tensors="pt",
                truncation=True,
            )

            with torch.no_grad():
                images = images.cuda().half()
                text_inputs = {k: v.cuda() for k, v in text_inputs.items()}
                logits = model(
                    images,
                    input_ids=text_inputs["input_ids"],
                    attention_mask=text_inputs["attention_mask"],
                )
                all_logits[split].append(logits)
                all_idxs[split].extend(batch_idxs_)
                all_study_idxs[split].extend(batch_study_idxs)

        all_logits[split] = torch.cat(all_logits[split], dim=0)
    target_ids = all_idxs["train"]
    target_study_ids = all_study_idxs["train"]
    target_masks = [gt_masks[idx] for idx in target_ids]
    target_masks = torch.tensor(target_masks).cuda()

    target_logits = all_logits["train"]
    targets = torch.sigmoid(target_logits)
    targets = targets.masked_fill(target_masks == 1, 1)
    targets = targets / targets.sum(dim=-1, keepdim=True)
    targets = torch.log(targets)

    topn = 4
    for split in all_logits:
        inputs = all_logits[split]
        image_annotation = {}
        for i, inp in tqdm(enumerate(inputs), total=len(inputs), desc=split):
            idx = all_idxs[split][i]
            study_idx = all_study_idxs[split][i]
            inp = torch.sigmoid(inp).unsqueeze(0)
            inp = inp / inp.sum(dim=-1, keepdim=True)
            score = F.kl_div(targets, inp, reduction="none").sum(dim=-1)

            num_topk = topn + (1 if split == "train" else 0)
            num_topk = min(num_topk, len(score))
            topk = torch.topk(score, num_topk, largest=False)[1]
            topk = topk.cpu().numpy().tolist()
            inp_target_ids = target_ids
            topk_ids = [
                inp_target_ids[t]
                for t in topk
                if inp_target_ids[t] != idx and target_study_ids[t] != study_idx
            ]
            image_annotation[idx] = topk_ids[:topn]
        with open(output_path.replace("[placeholder]", split), "w") as f:
            for idx in image_annotation:
                sample = {"id": idx, "retrieved": image_annotation[idx]}
                f.write(json.dumps(sample) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path", type=str, default="./checkpoints/expert_model.safetensors"
    )
    parser.add_argument("--image_path", type=str, default="./data/mimic_cxr/images/")
    parser.add_argument(
        "--annotation_path", type=str, default="./data/mimic_cxr/annotation.json"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="data/mimic_cxr/knowledge_[placeholder].jsonl",
    )
    parser.add_argument(
        "--clinical_context_path",
        type=str,
        default="./data/mimic_cxr/clinical_context.json",
    )
    parser.add_argument(
        "--gt_observation_path",
        type=str,
        default="./data/mimic_cxr/observation.json",
    )
    args = parser.parse_args()

    vison_model_name = "microsoft/swinv2-large-patch4-window12to16-192to256-22kto1k-ft"
    text_model_name = "emilyalsentzer/Bio_ClinicalBERT"
    processor = AutoFeatureExtractor.from_pretrained(vison_model_name)
    config = AutoConfig.from_pretrained(vison_model_name)
    config.num_observation = 14
    config.pretrained_visual_extractor = vison_model_name
    text_model = AutoModel.from_pretrained(text_model_name)
    model = ExpertModel(config=config, text_model=text_model)
    state_dict = load_file(args.model_path)
    model.load_state_dict(state_dict)
    tokenizer = AutoTokenizer.from_pretrained(text_model_name)

    func(
        root_path=args.image_path,
        annotation_path=args.annotation_path,
        output_path=args.output_path,
        clinical_context_path=args.clinical_context_path,
        gt_observation_path=args.gt_observation_path,
    )
