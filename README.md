# Clever_Hans_Effect_INTERSPEECH_2024

[**paper link**](https://www.isca-archive.org/interspeech_2024/liu24f_interspeech.html).

The resource of paper "Clever Hans Effect Found in Automatic Detection of Alzheimer's Disease through Speech"

This paper was awarded the [**2024 ISCA Best Student Paper Award**](https://interspeech2024.org/isca-best-student-paper-award-2024/).

# Please set up the environment first：

```
pip install -r requirements.txt
```

# The introduction of the .py files:

1. **fine_tune_wav2vec2.py**: fine tuning wav2vec 2.0 model.
2. **infer.py**: Using fine-tuned wav2vec 2.0 for inference.
3. **extract_ComParE_2016.py**: Extracting ComParE 2016 feature set.
4. **machine_learning_classifier.py**: Building machine learning classifier.
5. **extract_w2v2finetune_feature_and_pca.py:** Extracting features from the last hidden layer of the fne-tuned wav2vec 2.0 model and using PCA for dimensionality reduction.
6. **preprocess_audio.py**: Using noisereduce and pydub package for preprocessing the original audio recordings.
   
# English dataset 

While the data of [Pitt corpus](https://dementia.talkbank.org/access/English/Pitt.html), [ADReSS](https://dementia.talkbank.org/ADReSS-2020/), and [ADReSSo](https://dementia.talkbank.org/ADReSSo-2021/) are publicly available, we are not able to redistribute any of these data per Data Use agreement with Dementia Bank. Individual investigators need to contact the [Dementia Bank](https://dementia.talkbank.org/) to request access to the data.

# Citations

If you use this repository in your research or project, please cite our paper:

```
@article{liu2024clever,
  title={Clever Hans Effect Found in Automatic Detection of Alzheimer's Disease through Speech},
  author={Liu, Yin-Long and Feng, Rui and Yuan, Jia-Hong and Ling, Zhen-Hua},
  journal={arXiv preprint arXiv:2406.07410},
  year={2024}
}
```

Thank you for your support!
