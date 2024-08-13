import soundfile as sf
import noisereduce as nr
from pydub import AudioSegment, effects 
import numpy as np
from pathlib import Path
from tqdm import tqdm 
original_audio_path='AD_detection/English/'
original_audio_path_list=[p for p in Path(original_audio_path).rglob('*.wav') if p.is_file()]
for p in tqdm(original_audio_path_list):                          
    # load data
    data, sr = sf.read(str(p))
    # perform noise reduction
    reduced_noise = nr.reduce_noise(y=data, sr=sr,stationary=True)
    snr_path='AD_detection/English_snr/'+p.parts[-2]
    Path(snr_path).parent.mkdir(parents=True, exist_ok=True)
    snr_path='AD_detection/English_snr/'+p.parts[-2]+'/'+p.stem+'_snr.wav'
    sf.write(snr_path, reduced_noise,sr)
    rawsound = AudioSegment.from_file(snr_path, "wav")
    normalizedsound = effects.normalize(rawsound)  
    Path('AD_detection/English_snrn/'+p.parts[-2]).mkdir(parents=True, exist_ok=True)
    normalizedsound.export('AD_detection/English_snrn/'+p.parts[-2]+'/'+p.stem+'_snrn.wav', format="wav")