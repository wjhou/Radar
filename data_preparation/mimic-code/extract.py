import pandas as pd
import argparse
import json


parser = argparse.ArgumentParser()
parser.add_argument("--input_file", required=True)
parser.add_argument("--output_file", required=True)

if __name__ == "__main__":
    args = parser.parse_args()
    input_data = pd.read_csv(args.input_file)
    output_data = {}

    # get the columns
    columns = input_data.columns.tolist()
    for i in range(len(input_data)):
        row = input_data.iloc[i]
        sample = {
            "Indication": row["indication"],
            "History": row["history"],
            "Comparison": row["comparison"],
            "Technique": row["technique"],
        }
        output_data[row["study"]] = sample
    # save to json
    with open(args.output_file, "w") as f:
        json.dump(output_data, f, indent=4)
