import random
import numpy as np
import pandas as pd
import soundfile
from datasets import Dataset, DatasetDict, Audio, ClassLabel
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification, TrainingArguments, Trainer
import evaluate
from pathlib import Path
import json
from matplotlib import pyplot as plt
import wave
import librosa
from tqdm import tqdm
import shutil
import random

#Take out a single speech of a single speaker
def split_unique_English_sample(folder_path,num_of_duration):
    audio_durations_list = []
    audio_path_list=[p for p in Path(folder_path).rglob("*.wav")]
    filter_audio=[]
    AD_path_list=[]
    HC_path_list=[]
    for audio_path in audio_path_list:
        with wave.open(str(audio_path), 'rb') as f:
            wav_time = f.getparams().nframes/f.getparams().framerate
            if wav_time>=num_of_duration:
                if audio_path.parts[-2]=='AD':
                    AD_path_list.append(audio_path)
                else:
                    HC_path_list.append(audio_path)
                audio_durations_list.append(wav_time)
    AD_par_unique=set([p.stem.split('-')[0] for p in AD_path_list])
    HC_par_unique=set([p.stem.split('-')[0] for p in HC_path_list])
    AD_par_unique_path_list=[]
    HC_par_unique_path_list=[]
    for unique in AD_par_unique:
        AD_par_unique_path_list.append(random.choice([str(p) for p in AD_path_list if unique in str(p)]))
    for unique in HC_par_unique:
        HC_par_unique_path_list.append(random.choice([str(p) for p in HC_path_list if unique in str(p)]))
    print(f'AD now:{len(AD_par_unique_path_list)}')
    print(f'HC now:{len(HC_par_unique_path_list)}')
    random.shuffle(AD_par_unique_path_list)
    random.shuffle(HC_par_unique_path_list)
    return AD_par_unique_path_list,HC_par_unique_path_list

wav_folder_path="AD_detection/English/"
num_of_duration=35
AD_par_unique_path_list,HC_par_unique_path_list=split_unique_English_sample(wav_folder_path,num_of_duration)

#Take out the training and validation set of the k-fold
def get_k_fold_data(k,i,wav_path_list):
    assert k>1
    fold_size=len(wav_path_list)//k
    wav_train=None
    for j in range(k):
        idx=slice(j*fold_size,(j+1)*fold_size)
        wav_part=wav_path_list[idx]
        if j==i:
            wav_valid=wav_part
        elif wav_train is None:
            wav_train=wav_part
        else:
            wav_train=wav_part+wav_train
    return wav_train,wav_valid

f1_metric = evaluate.load("evaluate-0.1.2/metrics/f1")
acc_metric=evaluate.load('evaluate-0.1.2/metrics/accuracy')

def compute_metrics(eval_pred):
    predictions = np.argmax(eval_pred.predictions, axis=1)
    result_acc=acc_metric.compute(predictions=predictions, references=eval_pred.label_ids)
    result_f1 = f1_metric.compute(predictions=predictions, references=eval_pred.label_ids, average="weighted")
    return result_f1

labels=['AD', 'HC']
label2id, id2label = dict(), dict()

for i, label in enumerate(labels):
    label2id[label] = str(i)
    id2label[str(i)] = label

num_labels = len(id2label)

def preprocess_function(examples):
    audio_arrays = [x["array"] for x in examples["audio"]]
    inputs = feature_extractor(audio_arrays, sampling_rate=16000,max_length=560000,truncation=True,padding=True,return_tensors="pt")
    return inputs

model_path="model/wav2vec2-large-xlsr-53-english/"
eval_steps=1
save_steps=1
per_device_train_batch_size=1
gradient_accumulation_steps=4
per_device_eval_batch_size=4
save_total_limit=2
num_train_epochs=15
logging_steps=1
seed=42
data_seed=42

output_dir='model_output/English/'+f'{Path(model_path).parts[-1]}'+'_'+f'{Path(wav_folder_path).parts[-1]}_5_folds_cross_valid_{num_of_duration}s'
Path(output_dir).mkdir(parents=True,exist_ok=True)

k=5
for i in range(k):

    data = DatasetDict()
    AD_wav_train,AD_wav_valid=get_k_fold_data(k,i,AD_par_unique_path_list)
    HC_wav_train,HC_wav_valid=get_k_fold_data(k,i,HC_par_unique_path_list)
    wav_train=AD_wav_train+HC_wav_train
    wav_valid=AD_wav_valid+HC_wav_valid
    random.shuffle(wav_train)
    random.shuffle(wav_valid)
    print(f'wav_train_num:{len(wav_train)},wav_valid_num:{len(wav_valid)}')
    dict_wav={'wav_train':wav_train,'wav_valid':wav_valid}
    for wav_tv in list(dict_wav.keys()):
        d = {"audio": [], "label": []}
        for wav_fn in dict_wav[wav_tv]:
            cl = wav_fn.split('/')[-2]
            d["audio"].append(wav_fn)
            d["label"].append(cl)
        d = Dataset.from_dict(d).cast_column("audio", Audio()).cast_column("label", ClassLabel(names=['AD', 'HC']))
        data[wav_tv] = d
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_path)

    encoded_data = data.map(preprocess_function, remove_columns="audio", batched=True)

    model = AutoModelForAudioClassification.from_pretrained(model_path, num_labels=num_labels, label2id=label2id, id2label=id2label)
    
    output_dir_of_one_fold=f'{output_dir}/{i+1}_fold'
    Path(output_dir_of_one_fold).mkdir(parents=True,exist_ok=True)
    
    training_args = TrainingArguments(
    output_dir=output_dir_of_one_fold,
    # evaluation_strategy="epoch",
    evaluation_strategy='steps',
    eval_steps=eval_steps,
    #save_strategy="epoch",
    save_strategy='steps',
    save_steps = save_steps, 
    save_total_limit = save_total_limit,
    learning_rate=3e-5,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    per_device_eval_batch_size=per_device_eval_batch_size,   
    num_train_epochs=num_train_epochs,
    warmup_ratio=0.1,
    logging_strategy='steps',
    logging_steps=logging_steps,
    load_best_model_at_end=True,
    metric_for_best_model='f1',
    dataloader_drop_last=False,
    seed=seed,
    data_seed=data_seed)
    
    trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_data["wav_train"],
    eval_dataset=encoded_data["wav_valid"],
    tokenizer=feature_extractor,
    compute_metrics=compute_metrics,)
    trainer.train()
    
    

