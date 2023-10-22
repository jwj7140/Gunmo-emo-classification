#!/usr/bin/env python3
import jsonlines, json
import argparse


parser = argparse.ArgumentParser(description="merging prediction code")
parser.add_argument("--test_file", type=str, help="Filename to merge(jsonl)", default="nikluge-ea-2023-test.jsonl")
parser.add_argument("--predict-file", type=str, help="name of prediction file(json)", default="test_data_clean_predict.json")
parser.add_argument("--output-file", type=str, help="Filename to save(jsonl)", default="nikluge-ea-2023-test_predict.jsonl")
args = parser.parse_args()


name = "test"

data = []


with jsonlines.open(args.test_file) as f:
    for line in f.iter():
        data.append(line)

with open(args.predict_file, encoding='utf-8') as f:
    predict = json.load(f)



point = {
    "joy":33.1,     #4498
    "anticipation": 18.4, #4532
    "trust":35.7, #4593
    "surprise":21.0, #4661
    "disgust":26.5,  #4672
    "fear":56.4, #4704
    "anger":35.2,  #4670
    "sadness":50.8  #4650
}
#위 백분율은 자체 제작 모델에 최적화되어 있습니다. percent_simulation.py의 출력을 바탕으로 point를 수정하세요.


print(point)

for i in range(len(data)):
    data[i]["output"] = {"joy": "False", "anticipation": "False", "trust": "False", "surprise": "False", "disgust": "False", "fear": "False", "anger": "False", "sadness": "False"}
    for emo in list(point.keys()):
        if (predict[i][emo] >= point[emo]):
            data[i]["output"][emo] = "True"


with open(args.output_file, encoding= "utf-8",mode="w") as file: 
	for i in data: file.write(json.dumps(i,ensure_ascii=False) + "\n")