#!/usr/bin/env python3
import jsonlines, json
import argparse



parser = argparse.ArgumentParser(description="Data converting code")
parser.add_argument("--input-file", type=str, help="Filename to convert(jsonl)", default="nikluge-ea-2023-train.jsonl")
parser.add_argument("--output-file", type=str, help="Filename to save(json)", default="train_data_clean.json")
args = parser.parse_args()

data = []
count = [0,0,0,0,0,0,0,0]

map = []
with jsonlines.open(args.input_file) as f:
    for line in f.iter():
        sen = line["input"]["form"]
        target = line["input"]["target"]["form"]
        output = []
        if (line["output"]["joy"] == "True"):
            output = "기쁨"
            count[0] += 1
            data.append({
                "text": sen,
                "target": target,
                "emo": output
            })
        if (line["output"]["anticipation"] == "True"):
            output = "기대"
            count[1] += 1
            data.append({
                "text": sen,
                "target": target,
                "emo": output
            })
        if (line["output"]["trust"] == "True"):
            output = "신뢰"
            count[2] += 1
            data.append({
                "text": sen,
                "target": target,
                "emo": output
            })
        if (line["output"]["surprise"] == "True"):
            output = "당황"
            count[3] += 1
            data.append({
                "text": sen,
                "target": target,
                "emo": output
            })
        if (line["output"]["disgust"] == "True"):
            output = "혐오"
            count[4] += 1
            data.append({
                "text": sen,
                "target": target,
                "emo": output
            })
        if (line["output"]["fear"] == "True"):
            output = "공포"
            count[5] += 1
            data.append({
                "text": sen,
                "target": target,
                "emo": output
            })
        if (line["output"]["anger"] == "True"):
            output = "분노"
            count[6] += 1
            data.append({
                "text": sen,
                "target": target,
                "emo": output
            })
        if (line["output"]["sadness"] == "True"):
            output = "슬픔"
            count[7] += 1
            data.append({
                "text": sen,
                "target": target,
                "emo": output
            })


print(len(data))
print(count)

with open(args.output_file, "w", encoding='utf-8') as json_file:
    json.dump(data, json_file, indent=2, ensure_ascii=False)