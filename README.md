# Pitch Accent Detection improves Pretrained Automatic Speech Recognition

This repository contains the code accompanying our INTERSPEECH 2025 paper:

- **Title:** Enhancing Robot Instruction Understanding and Disambiguation via Speech Prosody  
- **Authors:** David Sasu and Natalie Schluter  
- **Affiliation:** IT University of Copenhagen

## Abstract

We show the performance of Automatic Speech Recognition (ASR) systems that use semi-supervised speech representations can be boosted by a complimentary pitch accent detection module, by introducing a joint ASR and pitch accent detection model. The pitch accent detection component of our model achieves a significant improvement on the state-of-the-art for the task, closing the gap in F1-score by 41%.  Additionally, the ASR performance in joint training decreases WER by 28.3% on LibriSpeech, under limited resource fine-tuning.  With these results, we show the importance of extending pretrained speech models to retain or re-learn important prosodic cues such as pitch accent. 

## Contributions

- Significant enhancement in pitch accent detection performance, surpassing previous state-of-the-art.
- A joint ASR and pitch accent detection model that improves ASR performance in limited resource scenarios.
- A semi-supervised self-training approach to augment pitch accent annotations, further enhancing ASR performance.


## Datasets

The experiments leverage the following datasets:
- Boston University Radio News Corpus (BURNC)
- LibriSpeech
- Libri-light


## Model Architecture

Our joint Prosody-ASR model:
- Employs pretrained wav2vec 2.0 as the base speech representation model.
- Adds an extra linear layer and layer normalization for improved pitch accent classification.


## Repository Structure

```text
. 
├── model_script.py/        # Code for joint model
├── train_script.py/        # Code for training and evaluating joint model
├── LICENSE
└── README.md
```

## Citation

If you use this code, please cite our paper:

```bibtex
@inproceedings{sasu2025pitch,
  title={Pitch Accent Detection Improves Pretrained Automatic Speech Recognition},
  author={Sasu, David and Schluter, Natalie},
  booktitle={Proceedings of Interspeech 2025},
  year={2025}
}
```
License

This project is licensed under the MIT License - see the LICENSE file for details.
