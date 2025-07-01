from modeling_expert_model import ExpertModel
import json
from pathlib import Path
from tqdm import tqdm
import torch
from PIL import Image
from safetensors.torch import load_file
from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    AutoTokenizer,
    AutoModel,
)
import argparse

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
        image_path = Path(self.root_path) / image_path
        image = Image.open(image_path).convert("RGB")
        image = processor(images=image, return_tensors="pt")["pixel_values"]
        hyp = self.hyps[idx]
        return image, hyp, self.idxs[idx]


def collate_fn(batch):
    images, hyps, idxs = zip(*batch)
    images = torch.cat(images)
    return images, hyps, idxs


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


def func(
    root_path,
    clinical_context,
    annotation_path,
    output_path,
):

    annotation = json.load(open(annotation_path, "r"))
    image_paths = []
    hyps = []
    idxs = []
    clinical_contexts = json.load(open(clinical_context, "r"))
    for split in annotation:
        data = annotation[split]
        id2report = {
            idx: data[idx]["findings"] for idx in data if "findings" in data[idx]
        }
        for idx in annotation[split]:
            if "findings" not in annotation[split][idx]:
                continue
            image_path = (
                annotation[split][idx]["image_path"]
                if "chexpert_plus" not in root_path
                else annotation[split][idx]["image_path"].replace(".jpg", ".png")
            )
            image_paths.append(image_path)
            idxs.append(idx)
            study_id = (
                annotation[split][idx]["image_path"].split("/")[2]
                if "iu_xray" not in clinical_context
                and "chexpert" not in clinical_context
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
            hyps.append(construct_clinical_context(clinical_context))

    train_dataset = Dataset(
        root_path,
        idxs,
        image_paths,
        hyps,
        tokenizer,
        processor,
    )
    batch_size = 64
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        collate_fn=collate_fn,
    )

    print(len(image_paths), len(hyps))
    model.cuda()
    model.eval()
    model.half()
    total = len(image_paths) // batch_size
    if len(image_paths) % batch_size != 0:
        total += 1

    image_annotation = {}
    for i, batch in tqdm(enumerate(train_loader), total=total):
        images, batch_hyps, batch_idxs = batch

        inputs = tokenizer(
            batch_hyps,
            padding="longest",
            max_length=512,
            truncation=True,
            return_tensors="pt",
        )

        with torch.no_grad():
            images = images.cuda().half()
            logits = model(
                images,
                input_ids=inputs["input_ids"].cuda(),
                attention_mask=inputs["attention_mask"].cuda(),
            )
            preds = (logits > 0) + 0.0
            preds = preds.cpu().numpy()
            findings = [
                [
                    find_
                    for j, find_ in enumerate(cp_class_names)
                    if pred[j] == 1 and find_ != "No Finding"
                ]
                for pred in preds
            ]
            for idx, finding in zip(batch_idxs, findings):
                image_annotation[idx] = finding

    with open(output_path, "w") as f:
        json.dump(image_annotation, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--expert_model_path", type=str, default=None)
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--annotation_path", type=str, required=True)
    parser.add_argument("--clinical_context_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()

    vision_model_name = "microsoft/swinv2-large-patch4-window12to16-192to256-22kto1k-ft"
    text_model_name = "emilyalsentzer/Bio_ClinicalBERT"
    processor = AutoFeatureExtractor.from_pretrained(vision_model_name)
    config = AutoConfig.from_pretrained(vision_model_name)
    config.pretrained_visual_extractor = vision_model_name
    text_model = AutoModel.from_pretrained(text_model_name)
    tokenizer = AutoTokenizer.from_pretrained(text_model_name)
    model = ExpertModel(config=config, text_model=text_model)
    state_dict = load_file(args.expert_model_path)
    model.load_state_dict(state_dict)

    func(
        root_path=args.image_path,
        annotation_path=args.annotation_path,
        clinical_context=args.clinical_context_path,
        output_path=args.output_path,
    )
