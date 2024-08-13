import opensmile
from pathlib import Path
import pandas as pd
from tqdm import tqdm
smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.ComParE_2016,
    feature_level=opensmile.FeatureLevel.Functionals,)

feature_set = pd.DataFrame()
audio_dir="AD_detection/English/"
audio_path_list=[p for p in Path(audio_dir).rglob("*.wav")]
print(f'num of audio: {len(audio_path_list)}')
for audio_path in tqdm(audio_path_list):
    feature_of_audio = smile.process_file(str(audio_path))
    feature_of_audio['label']=audio_path.parts[-2]
    feature_set = pd.concat([feature_set,feature_of_audio],ignore_index=True)

feature_set.to_csv('English_ComParE_2016_6373_feature_set.csv')
