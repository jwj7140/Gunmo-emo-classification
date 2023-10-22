#!/usr/bin/env python3
import jsonlines, json
import argparse

parser = argparse.ArgumentParser(description="Percent simulation code")
parser.add_argument("--predict-file", type=str, help="name of prediction file(json)", default="dev_data_clean_predict.json")
parser.add_argument("--answer-file", type=str, help="name of original datafile(jsonl)", default="nikluge-ea-2023-dev.jsonl")
args = parser.parse_args()


with open(args.predict_file, encoding='utf-8') as f:
    predict = json.load(f)


for point in range(1000):
    with jsonlines.open(args.answer_file) as f:
        num = 0
        count = {
            "joy":0,
            "anticipation":0,
            "trust":0,
            "surprise":0,
            "disgust":0,
            "fear":0,
            "anger":0,
            "sadness":0
        }
        for line in f.iter():
            for emo in list(count.keys()):
                if (predict[num][emo] >= point/10 and line["output"][emo] == "True"):
                    count[emo] += 1
                elif (predict[num][emo] < point/10 and line["output"][emo] == "False"):
                    count[emo] += 1

            num += 1
        print(f"{point/10}%: {count}")