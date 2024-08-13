from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
from datasets import Dataset, Audio, ClassLabel
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
import numpy as np
import evaluate
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


f1_metric = evaluate.load("evaluate-0.1.2/metrics/f1")
acc_metric=evaluate.load('evaluate-0.1.2/metrics/accuracy')

data = {"audio": [], "label": []}
infer_dataset='data_files/xxx.txt' # The path of the dataset to be inferred, the format of the dataset is as follows:
# AD_detection/data/English/AD/xxx.wav
# AD_detection/data/English/HC/xxx.wav
with open(infer_dataset, 'r') as fns:
    for fn in fns:
        path = fn.strip()
        cl = path.split('/')[-2]
        data["audio"].append(path)
        data["label"].append(cl)
data = Dataset.from_dict(data).cast_column("audio", Audio()).cast_column("label", ClassLabel(names=['AD', 'HC']))
id2label = {0: "AD", 1: "HC"}
model_path="model_output_path/checkpoint-3030"
fw = open(f'infer_output/Chinese/{Path(model_path).parts[-1]}_{Path(infer_dataset).parts[-1]}', 'w')
feature_extractor = AutoFeatureExtractor.from_pretrained(model_path)
model = AutoModelForAudioClassification.from_pretrained(model_path).to(device)
model.eval()
pred_label=[]
for i in tqdm(range(len(data))):
    inputs = feature_extractor(data[i]["audio"]["array"], sampling_rate=16000, max_length=560000,truncation=True,padding=True, return_tensors="pt")
    inputs = inputs.to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
        logits = logits.to(device)
    scores = list(F.softmax(logits, dim=1).detach().cpu().numpy()[0])
    fw.write(' '.join([data[i]["audio"]["path"], data[i]["audio"]["path"].split('/')[-2], str(scores),id2label[scores.index(max(scores))]]) + '\n')
    pred_label.append(int(scores.index(max(scores))))
    print(f'{Path(data[i]["audio"]["path"]).parts[-1]} truth label: {Path(data[i]["audio"]["path"]).parts[-2]},predicted label: {id2label[scores.index(max(scores))]}')
acc=sum(np.array(pred_label)==np.array(data['label']))/len(data)
print(f'acc:{round(acc,4)}')
fw.write(f'acc:{round(acc,4)}')
fw.close()