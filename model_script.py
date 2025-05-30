import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Model, Wav2Vec2Config, Wav2Vec2ForCTC
from torch.nn import CTCLoss, MSELoss
import datasets
import os

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class Wav2Vec2CombinedASR(Wav2Vec2ForCTC):
    def __init__(self, config):
        super().__init__(config)
        self.lstm_hidden_size = 256  # Example size, adjust as needed
        self.lstm = nn.LSTM(config.hidden_size + 1, self.lstm_hidden_size, batch_first=True, bidirectional=True, num_layers=1)
        self.extra_prosody_layer = nn.Linear(self.lstm_hidden_size * 2, self.lstm_hidden_size * 2)
        self.prosody_classifier = nn.Linear(self.lstm_hidden_size * 2, 1)
        self.config.ctc_zero_infinity = True
        self.config.dropout = 0.3
    
    def freeze_base_model_except_head(self):
        for param in self.parameters():
          param.requires_grad = False
        for param in self.lm_head.parameters():
          param.requires_grad = True
        return
    
    def unfreeze_base_model(self):
        for param in self.parameters():
          param.requires_grad = True
        super().freeze_feature_encoder()
        return

    def normalize(self, tensor):
        mean = tensor.mean()
        std = tensor.std()
        return (tensor - mean)/std
    
    def randomly_initialize_base_model_head(self):
        lm_head = self.lm_head
        for module in lm_head.modules():
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=0.02)
                if module.bias is not None:
                    module.bias.data.zero_()
        return

    def initialize_lstm_model_params(self, lstm_model_parameters_path):
        lstm_params = torch.load(lstm_model_parameters_path)
        self.lstm.load_state_dict(lstm_params)
        return 


    def forward(self, input_values, pitch=None, attention_mask=None, asr_labels=None, prosodic_labels=None):
        if -1 in prosodic_labels:
            include_prosody = False
        else:    
            include_prosody = True
        include_asr = True
        asr_model_outputs = super().forward(input_values=input_values, attention_mask=attention_mask, labels=asr_labels, output_hidden_states=True, return_dict=True)
        hidden_states = asr_model_outputs.hidden_states[-1]
        prosody_logits = None
        if pitch is not None:
            if (pitch.shape[1] < hidden_states.shape[1]):
                pitch = F.pad(pitch, pad=(0, hidden_states.shape[1] - pitch.shape[1]), mode='constant', value=0)
            elif (pitch.shape[1] > hidden_states.shape[1]):
                pitch = pitch[:hidden_states.shape[1]]
            pitch = pitch.reshape(pitch.shape[0], pitch.shape[1],1)
        tup = (pitch.to(device, dtype=torch.float),hidden_states)
        x_ = torch.cat(tup,2)
        x_lstm, (_, _) = self.lstm(x_)
        prosody_vectors = self.extra_prosody_layer(x_lstm)
        normalized_prosody_vectors = self.normalize(prosody_vectors)
        prosody_logits = self.prosody_classifier(normalized_prosody_vectors)
        loss = None
        if asr_labels is not None or prosodic_labels is not None:
            loss = 0
            if asr_labels is not None and include_asr:
                loss += asr_model_outputs.loss
            if prosodic_labels is not None and prosody_logits is not None and include_prosody:
                mse_loss = MSELoss()
                prosody_logits = prosody_logits.squeeze(2)
                prosody_target_size = prosodic_labels.shape[1]
                padding_right = prosody_target_size - prosody_logits.shape[1]
                prosody_logits = F.pad(prosody_logits, (0, padding_right))
                loss += mse_loss(prosody_logits, prosodic_labels)
        return {
            "asr_output": asr_model_outputs.logits,
            "prosody_logits": prosody_logits,
            "loss": loss
        }


