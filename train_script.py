import os
import sys
import parselmouth
import json
import numpy as np
import evaluate
import datasets
import torch
import heapq
import torchaudio
import torch.nn.functional as F
import argparse
import warnings
import kenlm
import string
import jiwer


from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from transformers import TrainingArguments
from datasets import load_metric, load_dataset, Audio
from transformers import (
    TrainingArguments,
    Trainer,
    TrainerCallback,
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
    Wav2Vec2CTCTokenizer,
    AutoFeatureExtractor,
    AutoProcessor,
    get_scheduler,
    pipeline,
)
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_scheduler
from tqdm.auto import tqdm
from fairseq.models.transformer_lm import TransformerLanguageModel
from model_script import *
from pyctcdecode import build_ctcdecoder, BeamSearchDecoderCTC


def load_dataset_without_prosody_labels(file_path, data_type):
    '''
    Pre-processes the librispeech dataset.

    Params:
        file_path (str): Path to where the data is stored.
        data_type (str): Indicates whether the data is dev or eval set.
    
    Returns:
        dataset (obj): The pre-processed dataset object.
    '''
    data_files = {}
    if file_path is not None:
        data_files[data_type] = file_path

    phrasing_features = datasets.Features({
        'path': datasets.features.Value('string'),
        'text': datasets.features.Value('string')
    })

    dataset = datasets.load_dataset("json", data_files=data_files, features=phrasing_features)
    return dataset



def load_dataset_with_prosody_labels(file_train,file_eval,file_valid):
    '''
    Loads the train, eval and validation data from the paths where they are stored.

    Params:
        file_train (str): Path to the training data
        file_eval (str): Path to the evaluation data
        file_valid (str): Path to the validation data
    
    Returns:
        dataset (obj): Object containing the extracted datasets.
    '''
    data_files = {}
    if file_train is not None:
        data_files["train"] = file_train
    if file_eval is not None:
        data_files["eval"] = file_eval
    if file_valid is not None:
        data_files["valid"] = file_valid

    
    phrasing_features = datasets.Features({
        'path': datasets.features.Value('string'), 
        'label': datasets.features.Sequence(datasets.features.Value(dtype='float64')),
        'text': datasets.features.Value('string'),
    })


   
    dataset = datasets.load_dataset("json", data_files=data_files, features=phrasing_features)
    return dataset


def compute_metrics(pred):
    '''
    Computes the word error rate.

    Params:
        pred (obj): Contains the logits predicted by the model.
    
    Returns:
        wer_result (float): Represents the computed word error rate.
    '''
    wer = evaluate.load("wer")
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits[0], axis=-1)
    pred.label_ids[0][pred.label_ids[0] == -100] = processor.tokenizer.pad_token_id
    pred_str = processor.batch_decode(pred_ids)
    label_str = processor.batch_decode(pred.label_ids[0], group_tokens=False)
    for i in range(min(10, len(pred_str))):
        print(f"Prediction: {pred_str[i]}")
        print(f"Reference: {label_str[i]}")
    non_empty_ref = []
    non_empty_preds = []
    for ref_index in range(len(label_str)):
        if label_str[ref_index] != '':
            non_empty_ref.append(label_str[ref_index])
            non_empty_preds.append(pred_str[ref_index])
    wer_result = wer.compute(predictions=non_empty_preds, references=non_empty_ref)
    return {"wer": wer_result}



def calculate_wer_and_cer(predicted_texts, correct_texts):
    '''
    Calculates the word error rate and the character error rate.

    Params:
        predicted_texts (str): Texts predicted by the model.
        correct_texts (str): Gold-label texts.
    
    Returns:
        wer (float): Computed word error rate
        cer (float): Computed character error rate
    '''
    wer = jiwer.wer(correct_texts, predicted_texts)
    cer = jiwer.cer(correct_texts, predicted_texts)
    print(f"Word Error Rate: {wer}, Character Error Rate: {cer}")
    return wer, cer



def inference(dataset, model_dir):
    '''
    Uses the trained model to predict text.

    Params:
        dataset (obj): This represents the either the dev or eval dataset.
        model_dir (str): This is the path to where the trained model is stored.
    
    Returns:
        predicted_texts (str): This represents the predicted texts from the model.
        correct_texts (str): This represents the gold-labelled texts of the audios.
    '''
    predicted_texts = ""
    correct_texts = ""
    transcriber = pipeline("automatic-speech-recognition", model=model_dir, tokenizer=processor)
    for audio_item in dataset:
        audio_path = audio_item['audio']['path']
        audio_transcription = audio_item["text"]
        prediction = transcriber(audio_path)
        print('correct text: ', audio_transcription)
        print('predicted text: ', prediction)
        predicted_texts = predicted_texts + " " + prediction['text']
        correct_texts = correct_texts + " " + audio_transcription
    return predicted_texts, correct_texts


def inference_with_ngram(dataset, model_dir):
    '''
    Performs inference with the trained language model.

    Params:
        dataset (obj): Dataset that is being used for the inference process.
        model_dir (str): Path to the model used for the inference process.
        
    Returns:
        overall_wer (float): The computed word error rate generated during the inference process.
    '''
    sentences_info = []
    wav2vec_processor = Wav2Vec2Processor.from_pretrained(model_dir)
    model = Wav2Vec2ForCTC.from_pretrained(model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    vocab_list = wav2vec_processor.tokenizer.convert_ids_to_tokens(list(range(wav2vec_processor.tokenizer.vocab_size)))
    decoder = build_ctcdecoder(labels=vocab_list,kenlm_model_path="/home/dasa/cross_lingual_prosodic_annotation/Research Code/pretrained_prosody_annotation_models/useful_files/wav2tobi_code/Wav2ToBI/librispeech_lm_corpus_dir/kenlm/build/bin/4gram.bin")
    predicted_texts = ""
    correct_texts = ""
    for audio_item in dataset:
        audio_path = audio_item['audio']['path']
        audio_transcription = audio_item["text"]
        waveform, sample_rate = torchaudio.load(audio_path)
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)
        input_values = wav2vec_processor(waveform.squeeze().numpy(), return_tensors="pt", sampling_rate=16000).input_values.to(device)
        with torch.no_grad():
            logits = model(input_values).logits[0].cpu().numpy()
        prediction_text = decoder.decode(logits)
        print('predicted text: ', prediction_text)
        print('correct text: ', audio_transcription)
        sentences_info.append([prediction_text, audio_transcription])
        predicted_texts = predicted_texts + " " + prediction_text
        correct_texts = correct_texts + " " + audio_transcription
    return predicted_texts, correct_texts, sentences_info



def beam_search_decode_acoustic(log_probs, processor, beam_width=10):
    '''
    Performs beam search decoding using the acoustic model only.

    Params:
        log_probs (torch.Tensor or numpy.ndarray): Log probabilities from the acoustic model (time_steps, vocab_size).
        processor (Wav2Vec2Processor): Processor with tokenizer.
        beam_width (int): Beam width.

    Returns:
        hypotheses (list): List of top N hypotheses with their scores.
    '''
    if isinstance(log_probs, torch.Tensor):
        log_probs = log_probs.numpy()
    labels = processor.tokenizer.convert_ids_to_tokens(range(processor.tokenizer.vocab_size))
    labels[processor.tokenizer.pad_token_id] = ''  
    labels[processor.tokenizer.word_delimiter_token_id] = ' ' 
    decoder = build_ctcdecoder(
        labels=labels,
        kenlm_model_path=None  # Set to None since we're not using a KenLM model here
    )
    probs = np.exp(log_probs)
    beam_results = decoder.decode_beams(probs, beam_width=beam_width)
    top_hypotheses = []
    for beam in beam_results[:beam_width]:
        text = beam[0]
        score = beam[3]
        top_hypotheses.append((text, score))
    return top_hypotheses



def compute_lm_scores_batch(transformer_lm, texts):
    '''
    Computes log probabilities of multiple texts using the Transformer LM.

    Params:
        transformer_lm (TransformerLanguageModel): The Transformer LM.
        texts (list): List of texts to score.

    Returns:
        lm_log_probs (list): List of log probabilities.
    '''
    lm_log_probs = []
    for text in texts:
        # Encode the text
        tokens = transformer_lm.encode(text)
        # Compute log probability
        lm_output = transformer_lm.score(text)
        lm_log_prob = lm_output['positional_scores'].sum().item()
        lm_log_probs.append(lm_log_prob)
    return lm_log_probs


def rescore_hypotheses(hypotheses, transformer_lm, lm_weight=0.5):
    '''
    Rescores the hypotheses using the Transformer LM.

    Params:
        hypotheses (list): List of tuples (text, acoustic_score).
        transformer_lm (TransformerLanguageModel): Loaded Transformer LM.
        lm_weight (float): Language model weight.

    Returns:
        rescored_hypotheses (list): List of tuples (text, total_score).
    '''
    texts = [hyp[0] for hyp in hypotheses]
    lm_scores = compute_lm_scores_batch(transformer_lm, texts)
    rescored_hypotheses = []
    for i, (text, acoustic_score) in enumerate(hypotheses):
        lm_score = lm_scores[i]
        total_score = acoustic_score + lm_weight * lm_score
        rescored_hypotheses.append((text, total_score))
    return rescored_hypotheses



def inference_with_transformer_LM(dataset, asr_model_dir, lm_model_dir, transformer_checkpoint_file_path, transformer_data_bin_path, transformer_sentence_model_path, beam_width=10, lm_weight=0.5):
    '''
    Performs inference with the trained Transformer language model.

    Params:
        dataset (Dataset): Dataset for inference.
        asr_model_dir (str): Path to the ASR model.
        lm_model_dir (str): Path to the Transformer language model directory.
        beam_width (int): Beam width for beam search.
        lm_weight (float): Weighting factor for the language model score.

    Returns:
        predicted_texts (str): Concatenated predicted texts.
        correct_texts (str): Concatenated ground truth texts.
        sentences_info (list): List of [predicted_text, actual_text] pairs.
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = Wav2Vec2Processor.from_pretrained(asr_model_dir)
    asr_model = Wav2Vec2ForCTC.from_pretrained(asr_model_dir).to(device)
    asr_model.eval()
    transformer_lm = TransformerLanguageModel.from_pretrained(
        lm_model_dir,
        checkpoint_file=transformer_checkpoint_file_path, 
        data_name_or_path=transformer_data_bin_path, #data_bin path
        bpe='sentencepiece',  # Adjust based on your preprocessing
        sentencepiece_model=transformer_sentence_model_path  # Path to your SentencePiece model
    )
    transformer_lm.eval()
    predicted_texts = ""
    correct_texts = ""
    sentences_info = []
    for audio_item in dataset:
        audio_path = audio_item['audio']['path']
        audio_transcription = audio_item["text"]
        waveform, sample_rate = torchaudio.load(audio_path)
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
        input_values = processor(waveform.squeeze().numpy(), return_tensors="pt", sampling_rate=16000).input_values.to(device)
        with torch.no_grad():
            logits = asr_model(input_values).logits 
            log_probs = F.log_softmax(logits, dim=-1)[0].cpu()
        hypotheses = beam_search_decode_acoustic(log_probs, processor, beam_width=beam_width)
        rescored_hypotheses = rescore_hypotheses(hypotheses, transformer_lm, lm_weight=lm_weight)
        best_hypothesis = max(rescored_hypotheses, key=lambda x: x[1])[0]
        sentences_info.append([best_hypothesis, audio_transcription])
        predicted_texts += " " + best_hypothesis
        correct_texts += " " + audio_transcription
    return predicted_texts, correct_texts, sentences_info


   
def librispeech_preprocess_function(samples):
    '''
    Pre-processes the librispeech dataset samples.

    Params:
        samples (dict): Dataset samples to be pre-processed.

    Returns:
        inputs (dict): Pre-processed data samples.
    '''
    if samples is None:
        return None

    pitches = []
    audio_arrays = []
    sampling_rate = 16000
    for x in samples["audio"]:
        audio_arrays.append(x["array"])
        snd = parselmouth.Sound(values = x["array"], sampling_frequency = 16_000)
        pitch = snd.to_pitch()
        pitches.append(list(pitch.selected_array['frequency'])[::2])
        
    
    inputs = processor(audio_arrays, sampling_rate=sampling_rate, text=samples["text"])

    inputs['pitch'] = pitches
    
    return inputs


def preprocess_data_with_prosody_labels(samples):
    '''
    Pre-processes the dataset samples that contain prosody labels

    Params:
        samples (dict): Dataset samples to be pre-processed.

    Returns:
        inputs (dict): Pre-processed data samples.
    '''
    if samples is None:
        return None
    pitches = []
    audio_arrays = []
    for x in samples["audio"]:
        audio_arrays.append(x["array"])
        snd = parselmouth.Sound(values = x["array"], sampling_frequency = 16_000)
        pitch = snd.to_pitch()
        pitches.append(list(pitch.selected_array['frequency'])[::2])
    labels = samples["label"]
    labels_rate = 50 # labels per second
    num_padded_labels = round(args.max_duration * labels_rate)

    for label in labels:
        for _ in range(len(label), num_padded_labels):
            label.append(0)
        if remove_extra_label:
            label.pop()
    inputs = processor(audio_arrays, sampling_rate=16000, text=samples["text"])
    inputs['pitch'] = pitches
    inputs['prosodic_labels'] = labels
    return inputs


def asr_model_evaluation(dataset, trained_model_path, lm_model_dir, transformer_checkpoint_file_path, transformer_data_bin_path, transformer_sentence_model_path):
    '''
    Evaluates the trained model.

    Params:
        dataset (dict): Dataset containing val and test data samples.
        trained_model_path (str): Path to the trained model.
        lm_model_dir (str): Path to the language model to be used in the evaluation process.
        transformer_checkpoint_file_path (str): Path to the transformer model to be used in the evaluation process.
        transformer_data_bin_path (str): The binarized data directory with dictionary and .bin/.idx files that Fairseq uses internally.
        transformer_sentence_model_path (str): The SentencePiece model file used to turn raw strings into the subword token IDs that both the Transformer LM and ASR beam‐search rescoring expect.
    
    Returns:
        None
    '''
    if args.use_burnc == True:
        pred_without_lm, actual_without_lm = inference(dataset["eval"], trained_model_path)
        pred_with_ngram, actual_with_ngram, sentences = inference_with_ngram(dataset["eval"], trained_model_path)
        pred_with_transformer, actual_with_transformer, sentences = inference_with_transformer_LM(dataset["eval"], trained_model_path, lm_model_dir, transformer_checkpoint_file_path, transformer_data_bin_path, transformer_sentence_model_path, beam_width=10, lm_weight=0.5)
    else:
        pred_without_lm, actual_without_lm = inference(processed_dataset_test, trained_model_path)
        pred_with_ngram, actual_with_ngram, sentences = inference_with_transformer_LM(processed_dataset_test, trained_model_path, lm_model_dir, transformer_checkpoint_file_path, transformer_data_bin_path, transformer_sentence_model_path, beam_width=10, lm_weight=0.5)
        pred_with_transformer, actual_with_transformer, sentences = inference_with_ngram(processed_dataset_valid, trained_model_path)
    calculate_wer_and_cer(pred_with_ngram, actual_with_ngram)
    return


def prosody_model_inference(model, eval_dataloader, output_file_path):
    '''
    Performs inference with the prosody head of the joint model.

    Params:
        model (obj): The trained joint model.
        eval_dataloader (obj): The dataloader object containing the test dataset.
        output_file_path (str): The path where the predicted labels should be stored.
    
    Returns:
        None
    '''
    metric = load_metric("mse")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = os.path.dirname(output_file_path)
    if out_dir != "":
        os.makedirs(out_dir, exist_ok=True)
    model.eval()
    all_predictions = []
    for _, batch in enumerate(eval_dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        try:
            with torch.no_grad():
                outputs = model(**batch)
                logits = outputs["prosody_logits"]
                predictions = logits.reshape(-1)
                labels = batch["prosodic_labels"].reshape(-1)
                all_predictions.append(predictions.cpu().detach().numpy())
                metric.add_batch(predictions=predictions, references=labels)
        except:
            continue
    return


class UnfreezeModelCallback(TrainerCallback):
    def __init__(self, unfreeze_step, model):
        self.unfreeze_step = unfreeze_step
        self.model = model

    def on_step_begin(self, args, state, control, **kwargs):
        if state.global_step == self.unfreeze_step:
          print("Unfreezing!!")
          model.unfreeze_base_model()


@dataclass
class DataCollatorCTCWithPadding:
    processor: AutoProcessor
    padding: Union[bool, str] = "longest"

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor, List[float]]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        batch = self.processor.pad(input_features, padding=self.padding, return_tensors="pt")
        max_length = max(len(feature["pitch"]) for feature in features)
        pitch_batch = torch.full((len(features), max_length), fill_value=0.0)  # Replace 0.0 with an appropriate padding value for pitch
        for i, feature in enumerate(features):
            pitch_batch[i, :len(feature["pitch"])] = feature["pitch"].float().clone().detach()
        max_label_length = max(len(feature["asr_labels"]) for feature in features)
        asr_labels_batch = torch.full((len(features), max_label_length), fill_value=-100, dtype=torch.long)  # -100 is often used for ignoring in loss computation
        for i, feature in enumerate(features):
            label_len = len(feature["asr_labels"])
            asr_labels_batch[i, :label_len] = feature["asr_labels"].float().clone().detach()
        batch["asr_labels"] = asr_labels_batch
        batch["pitch"] = pitch_batch
        if "prosodic_labels" in features[0]:
            max_prosodic_label_length = max(len(feature["prosodic_labels"]) for feature in features)
            prosodic_labels_batch = torch.full((len(features), max_prosodic_label_length), fill_value=0, dtype=torch.float) 
            for i, feature in enumerate(features):
                label_len = len(feature["prosodic_labels"])
                prosodic_labels_batch[i, :label_len] = feature["prosodic_labels"].float().clone().detach()
            batch["prosodic_labels"] = prosodic_labels_batch
        return batch






if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a model for audio frame classification')
    parser.add_argument('--model_checkpoint', type=str, default="facebook/wav2vec2-base-960h", help='Path, url or short name of the model')
    parser.add_argument('--file_train', type=str, help='Path to the training dataset (a JSON file)')
    parser.add_argument('--burnc_file_valid', type=str, help='Path to the BURNC validation dataset (a JSON file)')
    parser.add_argument('--burnc_file_eval', type=str, help='Path to the BURNC evaluation (test) dataset (a JSON file)')
    parser.add_argument('--librispeech_file_valid', type=str, help='Path to the Librispeech validation dataset (a JSON file)')
    parser.add_argument('--librispeech_file_eval', type=str, help='Path to the Librispeech evaluation (test) dataset (a JSON file)')
    parser.add_argument('--num_epochs', type=int, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--file_output', type=str, help='Path for the output file (output.txt)')
    parser.add_argument('--model_save_dir', type=str, help='Directory for saving the training log and the finetuned model')
    parser.add_argument('--max_duration', type=float, default=21.0, help='Maximum duration of audio files, default = 21s (must be >= duration of the longest file)')
    parser.add_argument('--mode', type=str, default="both", help='Mode: "train", "eval" or "both" (default is "both")')
    parser.add_argument('--epochs_between_checkpoints', type=int, default=1, help='Number of epochs between saved checkpoints during training. Default is 1 - saving every epoch.')
    parser.add_argument('--lr_init', type=float, default=5e-5, help='Initial learning rate')
    parser.add_argument('--lr_num_warmup_steps', type=int, default=0, help='Number of warmup steps for the learning rate scheduler')
    parser.add_argument('--remove_last_label', type=int, default=1, help='Remove the last value from ref. labels to match the number of predictions? 0 = no, 1 = yes (Default: yes)')
    parser.add_argument('--training_metric', type=str, default='asr', help='provide the metric that would be used to compute the loss')
    parser.add_argument('--use_burnc', type=bool, default=False, help='State True if the BURNC dataset is being used otherwise state False')



    args = parser.parse_args()

    if args.remove_last_label > 0:
        remove_extra_label = True # in a 20.0 s audio, there will be 1000 labels but only 999 logits -> remove the last label so the numbers match
    else: # if the labels are already fixed elsewhere
        remove_extra_label = False

    do_train = args.mode in ['train','both']
    do_eval = args.mode in ['eval','both']
    if args.epochs_between_checkpoints < 0:
        raise ValueError("''--epochs_between_checkpoints'' must be >= 0")
    if do_train:
        if args.file_train is None:
            raise ValueError("Training requires path to the training dataset (argument '--file_train <path>'). "
                             "To disable training and only run evaluation using the existing model, use '--mode 'eval''")
        if args.num_epochs is None:
            raise ValueError("For training the model, the number of epochs must be specified (argument '--num_epochs <number>'). "
                             "To disable training and only run evaluation using the existing model, use '--mode 'eval''")
        if args.model_save_dir is None:
            warnings.warn("argument ''--model_save_dir'' is not set -> the finetuned model will NOT be saved.")
            if args.epochs_between_checkpoints > 0:
                print("Checkpoints during training will also NOT be saved.")
        if args.file_valid is None:
            print("There is no validation set. Loss will be calculated only on the training set.")
            do_validation = False
        else:
            do_validation = True
    else:
        do_validation = False

    if do_eval:
        do_validation = True
        if args.file_eval is None:
            raise ValueError("Evaluation requires path to the evaluation dataset (argument '--file_eval <path>'). "
                             "To disable evaluation and only perform training, use '--mode 'train''")
    if args.model_save_dir is None or args.epochs_between_checkpoints == 0:
        save_checkpoints = False
    else:
        save_checkpoints = True



 
    
    
    
    # MODEL INITIALISATION
    model = Wav2Vec2CombinedASR.from_pretrained('facebook/wav2vec2-base-960h')
    model.initialize_lstm_model_params('optimal_lstm_weights.pth')
    model.freeze_base_model_except_head()
    model.randomly_initialize_base_model_head()
    model_tokenizer = Wav2Vec2CTCTokenizer.from_pretrained('facebook/wav2vec2-base-960h')
    feature_extractor = AutoFeatureExtractor.from_pretrained(
        "facebook/wav2vec2-base-960h",
    )
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=model_tokenizer)


    # DATA PRE-PROCESSING
    dataset = load_dataset_with_prosody_labels(args.file_train,args.burnc_file_eval,args.burnc_file_valid)
    dataset = dataset.rename_column("path","audio")
    dataset = dataset.cast_column("audio", datasets.features.Audio(sampling_rate=16_000))

    if do_train:
        print("Processing training data...")
        processed_dataset_train = dataset["train"].map(preprocess_data_with_prosody_labels, remove_columns=["audio","label", "text"], batched=True)
        processed_dataset_train = processed_dataset_train.rename_column("labels", "asr_labels")
        processed_dataset_train.set_format("torch", columns=["input_values", "pitch", "prosodic_labels", "asr_labels"])
        train_dataloader = DataLoader(processed_dataset_train, shuffle=True, batch_size=args.batch_size)
        
    else:
        processed_dataset_train = None
    
    if do_validation:
        print("Processing validation data...")
        if args.use_burnc:
            processed_dataset_valid = dataset["valid"].map(preprocess_data_with_prosody_labels, remove_columns=["audio","label", "text"], batched=True)
            processed_dataset_valid = processed_dataset_valid.rename_column("labels", "asr_labels")
            processed_dataset_valid.set_format("torch", columns=["input_values", "pitch", "prosodic_labels", "asr_labels"])
            valid_dataloader = DataLoader(processed_dataset_valid, shuffle=False, batch_size=args.batch_size)
        else:
            validation_dataset = args.librispeech_file_valid
            validation_dataset = load_dataset_without_prosody_labels(validation_dataset, "val")
            validation_dataset = validation_dataset["val"]
            validation_dataset = validation_dataset.rename_column("path","audio")
            processed_dataset_valid = validation_dataset.cast_column("audio", Audio(sampling_rate=16_000))
    else:
        processed_dataset_valid = None
    

    if do_eval:
        print("Processing test data...")
        if args.use_burnc:
            processed_dataset_test = dataset["eval"].map(preprocess_data_with_prosody_labels, remove_columns=["audio","label", "text"], batched=True)
            processed_dataset_test = processed_dataset_test.rename_column("labels", "asr_labels")
            processed_dataset_test.set_format("torch", columns=["input_values", "pitch", "prosodic_labels", "asr_labels"])
            eval_dataloader = DataLoader(processed_dataset_test, batch_size=1)
        else:
            test_dataset = args.librispeech_file_eval
            test_dataset = load_dataset_without_prosody_labels(test_dataset, "test")
            test_dataset = test_dataset["test"]
            test_dataset = test_dataset.rename_column("path","audio")
            processed_dataset_test = test_dataset.cast_column("audio", Audio(sampling_rate=16_000))
    else:
        processed_dataset_test = None



    data_collator = DataCollatorCTCWithPadding(processor=processor, padding="longest")


    
    # MODEL TRAINING
    training_args = TrainingArguments(
          output_dir=args.model_save_dir,
          per_device_train_batch_size=8,
          gradient_accumulation_steps=2,
          learning_rate=2e-5,
          warmup_steps=1500,
          max_steps=30000,
          gradient_checkpointing=True,
          fp16=False,
          group_by_length=True,
          evaluation_strategy="steps",
          per_device_eval_batch_size=8,
          save_steps=100,
          eval_steps=100,
          logging_steps=25,
          load_best_model_at_end=True,
          metric_for_best_model="wer",
          greater_is_better=False,
          push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=processed_dataset_train,
        eval_dataset=processed_dataset_valid,
        tokenizer=processor,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
        callbacks=[UnfreezeModelCallback(unfreeze_step=15000, model=model)]
    )


    if do_train:
        print("Starting training...")
        trainer.train()
        model.config.save_pretrained(args.model_save_dir)

 

        
