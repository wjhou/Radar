import json
from clinical_metric import load_chexbert
from transformers import BertTokenizer
import argparse
from annotate_reference import main
import os


def main_pf(args):
    id2observation = {}
    for split in ["train", "valid", "test"]:
        filename = f"{split}_results.json"
        data = json.load(
            open(os.path.join(args.annotation_folder, filename), "r", encoding="utf-8")
        )
        info = {}
        for idx in data:
            report = data[idx]["hyp"]
            info[idx] = {"findings": report}
        all_data = {split: info}
        id2obs, report2obs = main(args, all_data)
        id2observation.update(id2obs)

        with open(
            os.path.join(args.annotation_folder, f"{split}_sentence_observation.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(report2obs, f, indent=4)

    with open(
        os.path.join(args.annotation_folder, "observation.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(id2observation, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--annotation_folder", type=str)
    parser.add_argument("--chexbert_path", type=str)
    parser.add_argument("--output_observation_path", type=str)
    parser.add_argument("--output_sentence_observation_path", type=str)
    args = parser.parse_args()
    chexbert = load_chexbert(args.chexbert_path)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    main_pf(args)
