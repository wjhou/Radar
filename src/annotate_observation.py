import json
from chexbert_util import load_chexbert, CONDITIONS
from transformers import BertTokenizer
from tqdm import tqdm
import pandas as pd
import torch
import numpy as np
from nltk.tokenize import sent_tokenize
import sys
import os


def annotate(chexbert, tokenizer, reports, batch_size):
    observations = []
    for i in tqdm(range(0, len(reports), batch_size), desc="Annotate reports"):
        batch = reports[i : i + batch_size]
        inputs = tokenizer.batch_encode_plus(
            batch, return_tensors="pt", padding=True, truncation=True, max_length=512
        )
        inputs = {k: v.cuda() for k, v in inputs.items()}
        logits = chexbert(
            source_padded=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )
        preds = [logit.argmax(dim=1) for logit in logits]
        preds = torch.stack(preds).cpu().numpy().T
        observations.append(preds)
    observations = np.concatenate(observations, axis=0)
    assert len(observations) == len(reports)
    return observations


def main(folder):
    chexbert = load_chexbert("../CheXbert/chexbert.pth")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    all_data = json.load(
        open(os.path.join(folder, "annotation.json"), "r", encoding="utf-8")
    )

    reports = set()
    sentences = set()
    report2sentences = {}
    id2report = {}
    for split in all_data:
        data = all_data[split]
        for idx in data:
            if "findings" not in data[idx]:
                continue
            report = data[idx]["findings"]
            reports.add(report)
            report_sentences = sent_tokenize(report)
            sentences.update(report_sentences)
            report2sentences[report] = report_sentences
            id2report[idx] = report

    reports = list(reports)
    batch_size = 4
    observations = annotate(chexbert, tokenizer, reports, batch_size)
    df = pd.DataFrame(observations, columns=CONDITIONS)
    df.replace(0, np.nan, inplace=True)
    df.replace(3, -1, inplace=True)
    df.replace(2, 0, inplace=True)

    df["Report"] = reports
    report2obs = {}
    for i, row in df.iterrows():
        report = row["Report"]
        obs = {condition for condition in CONDITIONS if row[condition] == 1}
        if len(obs) == 0:
            continue
        report2obs[report] = obs

    id2obs = {}
    for idx, report in id2report.items():
        obs = report2obs.get(report, [])
        if len(obs) == 0:
            continue
        id2obs[idx] = list(obs)
    json.dump(
        id2obs,
        open(os.path.join(folder, "observation.json"), "w", encoding="utf-8"),
        indent=4,
    )

    sentences = list(sentences)
    batch_size = 4
    observations = annotate(chexbert, tokenizer, sentences, batch_size)
    df = pd.DataFrame(observations, columns=CONDITIONS)
    df.replace(0, np.nan, inplace=True)
    df.replace(3, -1, inplace=True)
    df.replace(2, 0, inplace=True)
    df["Sentence"] = sentences[: len(df)]
    sentence2obs = {}
    for i, row in df.iterrows():
        sentence = row["Sentence"]
        obs = {
            condition
            for condition in CONDITIONS
            if row[condition] == 1 and condition != "No Finding"
        }
        sentence2obs[sentence] = obs

    report2obs = {}

    for report, sentences in report2sentences.items():
        report2obs[report] = []
        for sentence in sentences:
            if len(sentence2obs[sentence]) == 0:
                continue
            report2obs[report].append(
                {"sentence": sentence, "observations": list(sentence2obs[sentence])}
            )

    json.dump(
        report2obs,
        open(os.path.join(folder, "sentence_observation.json"), "w", encoding="utf-8"),
        indent=4,
    )


if __name__ == "__main__":
    folder = sys.argv[1]
    main(folder)
