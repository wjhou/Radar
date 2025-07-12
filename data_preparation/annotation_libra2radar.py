import json
import sys

if __name__ == "__main__":
    input_train_file = sys.argv[1] # data/libra_findings_section_train.json
    input_valid_file = sys.argv[2] # data/libra_findings_section_valid.json
    input_eval_file = sys.argv[3] # data/libra_findings_section_eval.json
    output_file = sys.argv[4] # data/annotation.json

    output = {}
    for input_file, split in zip(
        [input_train_file, input_valid_file, input_eval_file], ["train", "val", "test"]
    ):
        with open(input_file, "r") as f:
            data = [json.loads(line) for line in f if line.strip()]

        output[split] = {}
        for line in data:
            images = line["image"]
            prior_image_path, image_path = images[1], images[0]
            case_id = image_path.split("/")[-1].split(".")[0]
            findings = line["findings"]
            sample = {
                "image_path": image_path,
                "prior_image_path": prior_image_path,
                "findings": findings,
            }
            output[split][case_id] = sample
    with open(output_file, "w") as f:
        json.dump(output, f, indent=4, ensure_ascii=False)
