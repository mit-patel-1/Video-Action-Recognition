"""
Module: models.py

This module defines the LRCN (Long-term Recurrent Convolutional Network) model for video
classification. It uses an image or video CNN to extract features and an LSTM to
process the feature sequence.
An additional fully-connected layer is used to output the final class predictions.

Classes:
    Identity: A helper module that returns the input unchanged. It is used to replace the fully-connected
              layer of the ResNet backbone.
    LRCN: The main model that integrates the CNN backbone, an LSTM, dropout regularization, and a final
          fully-connected layer to produce class logits.
"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchvision import models
from torchvision.models.video import R2Plus1D_18_Weights, r2plus1d_18

class Identity(nn.Module):
    """
    A placeholder identity operator that is argument-insensitive.
    
    This module is used to replace the fully-connected (fc) layer in the ResNet backbone,
    effectively making the backbone output the raw features before classification.
    
    Example:
        >>> identity = Identity()
        >>> output = identity(input_tensor)
    """
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        """
        Forward pass that returns the input as is.
        
        Args:
            x (Tensor): Input tensor.
        
        Returns:
            Tensor: The same tensor x.
        """
        return x

class LRCN(nn.Module):
    """
    LRCN (Long-term Recurrent Convolutional Network) for video classification.
    
    This model uses a ResNet image backbone or an R(2+1)D video backbone to extract features.
    An LSTM network is then used to model the temporal dynamics across the sequence of frame features.
    Dropout is applied before the final fully-connected layer that produces class logits.

    Args:
        hidden_size (int): Number of features in the hidden state of the LSTM.
        n_layers (int): Number of recurrent layers in the LSTM.
        dropout_rate (float): Dropout rate applied before the final classification layer.
        n_classes (int): Number of output classes.
        pretrained (bool, optional): If True, uses a ResNet model pretrained on ImageNet. Default is True.
        feature_fusion (bool): Combine mean frame features with the final LSTM state.
        temporal_pool (bool): Predict at each real LSTM step and average predictions.
        cnn_model (str, optional): Specifies the ResNet variant to use as the backbone.
                                   Options: 'resnet18', 'resnet34', 'resnet50', 'resnet101',
                                   'resnet152', or 'r2plus1d_18'.
                                   Default is 'resnet34'.
    
    Raises:
        ValueError: If the specified cnn_model is not supported.
    """
    def __init__(self, hidden_size, n_layers, dropout_rate, n_classes, pretrained=True,
                 cnn_model='resnet34', feature_fusion=False, temporal_pool=False):
        super(LRCN, self).__init__()
        if feature_fusion and temporal_pool:
            raise ValueError('Temporal prediction does not use CNN feature fusion.')
        self.video_backbone = cnn_model == 'r2plus1d_18'
        if self.video_backbone and (feature_fusion or temporal_pool):
            raise ValueError('The video backbone uses the final LSTM state.')

        # Set up the ResNet backbone as a 2D CNN feature extractor.
        if self.video_backbone:
            weights = R2Plus1D_18_Weights.KINETICS400_V1 if pretrained else None
            base_cnn = r2plus1d_18(weights=weights)
        elif cnn_model == 'resnet18':
            base_cnn = models.resnet18(pretrained=pretrained)
        elif cnn_model == 'resnet34':
            base_cnn = models.resnet34(pretrained=pretrained)
        elif cnn_model == 'resnet50':
            weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            base_cnn = models.resnet50(weights=weights)
        elif cnn_model == 'resnet101':
            base_cnn = models.resnet101(pretrained=pretrained)
        elif cnn_model == 'resnet152':
            base_cnn = models.resnet152(pretrained=pretrained)
        else:
            raise ValueError('The input CNN backbone is not supported, please choose a valid ResNet variant.')

        # Retrieve the number of features output by the CNN's original fully-connected layer.
        num_features = base_cnn.fc.in_features
        
        # Replace the original fc layer with an identity mapping so that raw features are returned.
        base_cnn.fc = Identity()
        self.base_model = base_cnn

        # Define the LSTM to process the sequence of frame features.
        self.rnn = nn.LSTM(num_features, hidden_size, n_layers, batch_first=True)
        
        # Define dropout for regularization.
        self.dropout = nn.Dropout(dropout_rate)
        
        # Final fully-connected layer to produce logits for each class.
        self.feature_fusion = feature_fusion
        self.temporal_pool = temporal_pool
        classifier_size = hidden_size + num_features if feature_fusion else hidden_size
        self.fc = nn.Linear(classifier_size, n_classes)

    def forward(self, x, lengths=None):
        """
        Forward pass for the LRCN model.
        
        The input tensor x is expected to have the shape:
            (batch_size, time_steps, channels, height, width)
        
        The image CNN extracts one feature vector per frame. The video CNN uses
        spatial and temporal convolutions and reduces time by a factor of eight.
        The LSTM processes the real feature sequence before classification.

        Args:
            x (Tensor): Input tensor of shape (batch_size, time_steps, channels, height, width).
            lengths (Tensor, optional): Number of real frames in each video.

        Returns:
            Tensor: Video logits, or step logits (batch_size, time_steps, n_classes)
                when temporal_pool is enabled. Padding is masked when computing loss and scores.
        """
        batch_size, time_steps = x.shape[:2]
        if batch_size == 0 or time_steps == 0:
            raise ValueError('Each batch must contain videos with at least one frame.')
        if lengths is None:
            lengths = torch.full((batch_size,), time_steps, dtype=torch.long)
        lengths = torch.as_tensor(lengths, dtype=torch.long, device='cpu')
        if lengths.shape != (batch_size,) or (lengths < 1).any() or (lengths > time_steps).any():
            raise ValueError('Video lengths must match the batch and available frames.')

        if self.video_backbone:
            # Process equal lengths together so padding cannot enter the video CNN.
            feature_lengths = (lengths + 7) // 8
            features = None
            for length in lengths.unique().tolist():
                active = (lengths == length).nonzero(as_tuple=True)[0].to(
                    x.device, non_blocking=True)
                clips = x.index_select(0, active)[:, :length].transpose(1, 2)
                encoded = self.base_model.stem(clips)
                for block in (self.base_model.layer1, self.base_model.layer2,
                              self.base_model.layer3, self.base_model.layer4):
                    encoded = block(encoded)
                # Keep time for the LSTM and average only the spatial dimensions.
                encoded = encoded.mean(dim=(-1, -2)).transpose(1, 2)
                encoded = nn.functional.pad(
                    encoded, (0, 0, 0, int(feature_lengths.max()) - encoded.shape[1]))
                if features is None:
                    features = encoded.new_zeros(batch_size, *encoded.shape[1:])
                features = features.index_copy(0, active, encoded)
            lengths = feature_lengths
        else:
            # Only real frames go through the CNN.
            frame_features = []
            same_length = bool((lengths == lengths[0]).all())
            for idx in range(int(lengths.max())):
                if same_length:
                    # Keep the original video batch in every CNN call.
                    step_features = self.base_model(x[:, idx])
                else:
                    # Find active rows on the CPU to avoid a CUDA mask synchronization.
                    active = (lengths > idx).nonzero(as_tuple=True)[0].to(x.device, non_blocking=True)
                    frame = self.base_model(x[:, idx].index_select(0, active))
                    step_features = frame.new_zeros(batch_size, frame.shape[1])
                    step_features = step_features.index_copy(0, active, frame)
                frame_features.append(step_features)
            features = torch.stack(frame_features, dim=1)

        # Keep each video separate and leave padding out of the LSTM.
        packed = pack_padded_sequence(features, lengths, batch_first=True, enforce_sorted=False)
        recurrent, (hidden, _) = self.rnn(packed)
        if self.temporal_pool:
            steps, _ = pad_packed_sequence(recurrent, batch_first=True)
            return self.fc(self.dropout(steps))
        video_features = hidden[-1]
        if self.feature_fusion:
            # Padded feature rows are zero, so only real frames enter the mean.
            device_lengths = lengths.to(x.device, non_blocking=True)
            mean_features = features.sum(dim=1) / device_lengths.unsqueeze(1)
            video_features = torch.cat((video_features, mean_features), dim=1)
        out = self.dropout(video_features)
        
        # Pass the final output through the fully-connected layer to get class logits.
        out = self.fc(out)
        return out
