#!/usr/bin/env python3
import jsonlines, json
import argparse



parser = argparse.ArgumentParser(description="Data converting code(test,dev)")
parser.add_argument("--input-file", type=str, help="Filename to convert(jsonl)", default="nikluge-ea-2023-test.jsonl")
parser.add_argument("--output-file", type=str, help="Filename to save(json)", default="test_data_clean.json")
args = parser.parse_args()

data = []

map = []
with jsonlines.open(args.input_file) as f:
    for line in f.iter():
        sen = line["input"]["form"]
        target = line["input"]["target"]["form"]

        data.append({
            "text": sen,
            "target": target
        })


print(len(data))

with open(args.output_file, "w", encoding='utf-8') as json_file:
    json.dump(data, json_file, indent=2, ensure_ascii=False)