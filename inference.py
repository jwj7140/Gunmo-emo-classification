import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import numpy as np
import json
import argparse

parser = argparse.ArgumentParser(description="inference code")
parser.add_argument("--type", type=str, help="Type to inference(test or dev)", default="test")
args = parser.parse_args()

#모델을 불러옵니다.
model_id = "EleutherAI/polyglot-ko-12.8b"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map="auto")

lora_id = "squarelike/modu_emo_classify_12.8B-prompt-4epoch_v6"
model = PeftModel.from_pretrained(
    model,
    lora_id)

model.eval()
model.config.use_cache = True


def gen(text="", target=""):
    inputs = tokenizer(
                f"아래는 문장에서 대상에 대한 감정을 매우 정확하게 분류한다.\n### 문장: {text}\n### 대상: {target}\n### 감정:",
                return_tensors='pt',
                return_token_type_ids=False
            )

    outputs = model.generate(
            **inputs,
            max_new_tokens=1,
            temperature=0.001,
            return_dict_in_generate=True,
            output_scores=True
        )

    emo = [6716,1321,3065,6603,11333,4993,4921,8300]
    #여기서 emo의 토큰번호는 각각 "기쁨", "기대", "신뢰", "당황", "혐오", "공포", "분노", "슬픔"에 해당합니다.
    emo_name = ["joy","anticipation","trust","surprise","disgust","fear","anger","sadness"]
    dic = []
    for tokenNum in emo:
        outputs.sequences[0][-1] = tokenNum

        transition_scores = model.compute_transition_scores(
            outputs.sequences, (outputs.scores[0].float(),), normalize_logits=True
        )
        dic.append(round(np.exp(transition_scores[0][0].numpy())*100,2))
    return dict(zip(emo_name,dic))
    #각 감정에 대한 확률값을 반환합니다.

def generate(type):
    with open(f"{type}_data_clean.json", "r", encoding='utf-8') as json_file:
        json_data = json.load(json_file)
    predict = []
    for a, line in enumerate(json_data):
        assume = gen(text=line["text"], target=line["target"])

        predict.append(assume)
        print(a+1, assume)

    with open(f"{type}_data_clean_predict.json", "w", encoding='utf-8') as json_file:
        json.dump(predict, json_file, indent=2, ensure_ascii=False)
    

generate(args.type)