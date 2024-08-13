# Clever_Hans_Effect_INTERSPEECH_2024
The resource of paper "Clever Hans Effect Found in Automatic Detection of Alzheimer's Disease through Speech"
# Please set up the environment first：
pip install -r requirements.txt 
# The introduction of the .py files:
"fine_tune_wav2vec2.py": fine tune wav2vec 2.0 model.
"infer.py": Using fine-tuned wav2vec 2.0 for inference.
"extract_ComParE_2016.py": Extracting the ComParE 2016 feature set.
"machine_learning_classifier.py": Building machine learning classifier.
"extract_w2v2finetune_feature_and_pca.py": Extracting features from the last hidden layer of the fne-tuned wav2vec 2.0 model and using PCA for dimensionality reduction.
"preprocess_audio.py": Using noisereduce and pydub package for preprocessing the original audio recordings.
