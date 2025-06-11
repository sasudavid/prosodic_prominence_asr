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
        '''
        Freezes the base wav2vec2 model except the Prosody and ASR Head.
        '''
        for param in self.parameters():
          param.requires_grad = False
        for param in self.lm_head.parameters():
          param.requires_grad = True
        return
    
    def unfreeze_base_model(self):
        '''
        Unfreezes the wav2vec2 model but keeps the feature encoder frozen.
        '''
        for param in self.parameters():
          param.requires_grad = True
        super().freeze_feature_encoder()
        return

    def normalize(self, tensor):
        '''
        Normalizes a given input tensor

        Params:
            tensor (obj): The input tensor object.

        Returns:
            Normalized tensor object.
        '''
        mean = tensor.mean()
        std = tensor.std()
        return (tensor - mean)/std
    
    def randomly_initialize_base_model_head(self):
        '''
        Randomly initializes the weights of the ASR and Prosody Labelling heads.
        '''
        lm_head = self.lm_head
        for module in lm_head.modules():
            #for linear models, ensure that the weights are drawn from a normal distribution and the bias terms have a zero value.
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=0.02)
                if module.bias is not None:
                    module.bias.data.zero_()
        return

    def initialize_lstm_model_params(self, lstm_model_parameters_path):
        '''
        Initialize the bilstm model with the optimal hyper-parameter values found.

        Params:
            lstm_model_parameters_path (str): This is the path to where the hyper-parameter values are stored.
        '''
        lstm_params = torch.load(lstm_model_parameters_path)
        self.lstm.load_state_dict(lstm_params)
        return 


    def forward(self, input_values, pitch=None, attention_mask=None, asr_labels=None, prosodic_labels=None):
        #check to see if prosody/ASR labels are provided to help determine whether the prosody/ASR head should be included in output generation.
        if -1 in prosodic_labels:
            include_prosody = False
        else:    
            include_prosody = True
        include_asr = True

        #feed the input values extracted from the audio into the wav2vec2 model
        asr_model_outputs = super().forward(input_values=input_values, attention_mask=attention_mask, labels=asr_labels, output_hidden_states=True, return_dict=True)

        #obtain the last hidden state of the wav2vec2 model
        #hidden_states = asr_model_outputs.hidden_states[0] (mistake made)
        hidden_states = asr_model_outputs.hidden_states[-1]

        


        #check to see if you would like to include pitch information.
        if pitch is not None:
            #If pitch information is to be included, check to see if the pitch length provided is smaller than the time-step length of the hidden states
            if (pitch.shape[1] < hidden_states.shape[1]):
                #if the pitch is found to be smaller, pad the pitch with zeros to ensure that its length is equal to the time-step length of the hidden states
                pitch = F.pad(pitch, pad=(0, hidden_states.shape[1] - pitch.shape[1]), mode='constant', value=0)
            #However, if the pitch length is found to be greater, then you would have to truncate it's length
            elif (pitch.shape[1] > hidden_states.shape[1]):
                pitch = pitch[:hidden_states.shape[1]]

            #reshape the dimensions of the pitch tensor
            pitch = pitch.reshape(pitch.shape[0], pitch.shape[1],1)

        #append the pitch information as additional information to the last dimension of the hidden state
        tup = (pitch.to(device, dtype=torch.float),hidden_states)
        x_ = torch.cat(tup,2)

        #push the hidden state information together with the pitch into the bilstm to generate logits for the Prosody labels
        #NB: if you choose to remove pitch information you must edit the input dimensions of the LSTM model as well.
        x_lstm, (_, _) = self.lstm(x_)
        prosody_vectors = self.extra_prosody_layer(x_lstm)
        normalized_prosody_vectors = self.normalize(prosody_vectors)
        prosody_logits = self.prosody_classifier(normalized_prosody_vectors)

        
        
        if asr_labels is not None or prosodic_labels is not None:

            #initialize the loss
            loss = 0
            
            if asr_labels is not None and include_asr:
                #compute the loss for the ASR Head
                loss += asr_model_outputs.loss
                
            if prosodic_labels is not None and prosody_logits is not None and include_prosody:
                #compute the loss for the Prosody Head
                mse_loss = MSELoss()
                prosody_logits = prosody_logits.squeeze(2)
                prosody_target_size = prosodic_labels.shape[1]
                padding_right = prosody_target_size - prosody_logits.shape[1]
                prosody_logits = F.pad(prosody_logits, (0, padding_right))
                loss += mse_loss(prosody_logits, prosodic_labels)

        #return the ASR logits, Prosody Logits and the total loss
        return {
            "asr_output": asr_model_outputs.logits,
            "prosody_logits": prosody_logits,
            "loss": loss
        }


