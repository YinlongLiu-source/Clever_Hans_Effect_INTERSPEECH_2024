import torch
import torch.nn as nn
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
import soundfile as sf
import numpy as np
from transformers import Wav2Vec2Model
from pathlib import Path
import wave
import random
from tqdm import tqdm
import pandas as pd
from sklearn.decomposition import PCA
def split_unique_English_sample(wav_folder_path,num_of_duration):
    audio_durations_list = []
    audio_path_list=[p for p in Path(wav_folder_path).rglob("*.wav")]
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
audio_path_list=AD_par_unique_path_list+HC_par_unique_path_list
random.shuffle(audio_path_list)

model_path="model_output_dir/checkpoint-xxx/"
feature_extractor=AutoFeatureExtractor.from_pretrained(model_path)
feature_extractor_set={}
for audio_path in tqdm(audio_path_list):
    audio_arrays,sr=sf.read(audio_path)
    input_values=feature_extractor(audio_arrays, sampling_rate=16000,max_length=560000,truncation=True,padding=True,return_tensors="pt").input_values
    feature_extractor_set[Path(audio_path).stem+'='+Path(audio_path).parts[-2]]=input_values
w2v2_feature_set=[]
w2v2finetune=Wav2Vec2Model.from_pretrained(model_path)
w2v2finetune.cuda().eval()
with torch.no_grad():
    for key in tqdm(feature_extractor_set.keys()):
        input_values=feature_extractor_set[key].cuda()
        last_hidden_feature=torch.mean(w2v2finetune(input_values)[0],dim=1).tolist()
        if key.split('=')[1]=='AD':
            audio_name_last_hidden_feature_label=[key.split('=')[0]]+last_hidden_feature[0]+[0]
        elif key.split('=')[1]=='HC':
            audio_name_last_hidden_feature_label=[key.split('=')[0]]+last_hidden_feature[0]+[1]
        w2v2_feature_set.append(audio_name_last_hidden_feature_label)
pca = PCA(n_components=5)# or 10
new_feature_set = pca.fit_transform(np.array(w2v2_feature_set)[:,1:-1])
label_arr=np.expand_dims(np.array(w2v2_feature_set)[:,-1],axis=1)
new_feature_set=np.hstack((np.array(new_feature_set),label_arr))
print(pca.explained_variance_ratio_)
pd.DataFrame(w2v2_feature_set).to_csv('35s_w2v21024_'+f'_{Path(model_path).parts[-1]}'+f'_{Path(wav_folder_path).parts[-1]}'+'.csv')
pd.DataFrame(new_feature_set).to_csv('35s_pca_5_'+f'_{Path(model_path).parts[-1]}'+f'_{Path(wav_folder_path).parts[-1]}'+'.csv')


