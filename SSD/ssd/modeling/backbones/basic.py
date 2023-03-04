import torch
from torch import nn
from typing import Tuple, List


class BasicModel(torch.nn.Module):
    """
    This is a basic backbone for SSD.
    The feature extractor outputs a list of 6 feature maps, with the sizes:
    [shape(-1, output_channels[0], 38, 38),
     shape(-1, output_channels[1], 19, 19),
     shape(-1, output_channels[2], 10, 10),
     shape(-1, output_channels[3], 5, 5),
     shape(-1, output_channels[3], 3, 3),
     shape(-1, output_channels[4], 1, 1)]
    """
    def __init__(self,
            output_channels: List[int],
            image_channels: int,
            output_feature_sizes: List[Tuple[int]]):
        super().__init__()
        self.out_channels = output_channels
        self.output_feature_shape = output_feature_sizes

        self.output_layers = nn.ModuleList([ 
            # 38x38 
            nn.Sequential(
                nn.Conv2d(image_channels, 32, kernel_size=3, padding=1, stride =1),
                nn.ReLU(),
                nn.MaxPool2d(2,2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1),
                nn.ReLU(),
                nn.MaxPool2d(2,2),
                nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1),
                nn.ReLU(),
                nn.Conv2d(64, output_channels[0], kernel_size=3, padding=1, stride=2),
                nn.ReLU(),            
            ),
            # 19x19
            nn.Sequential( 
                nn.ReLU(),
                nn.Conv2d(output_channels[0], 128, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(128, output_channels[1], kernel_size=3, padding=1, stride=2),
                nn.ReLU(),
            ),
            #10x10 
            nn.Sequential( 
                nn.ReLU(),
                nn.Conv2d(output_channels[1], 256, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(256, output_channels[2], kernel_size=3, padding=1, stride=2),
                nn.ReLU(),
            ),

            #5x5
            nn.Sequential( 
                nn.ReLU(),
                nn.Conv2d(output_channels[2], 128, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(128, output_channels[3], kernel_size=3, padding=1, stride=2),
                nn.ReLU(),
            ),
            #3x3
            nn.Sequential( 
                nn.ReLU(),
                nn.Conv2d(output_channels[3], 128, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(128, output_channels[4], kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
            ),
            #1x1
            nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(output_channels[4], 128, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(128, output_channels[5], kernel_size=3, padding = 0, stride = 1),
                nn.ReLU(),
            ),

            
        ])

    def forward(self, x):
        """
        The forward functiom should output features with shape:
            [shape(-1, output_channels[0], 38, 38),
            shape(-1, output_channels[1], 19, 19),
            shape(-1, output_channels[2], 10, 10),
            shape(-1, output_channels[3], 5, 5),
            shape(-1, output_channels[3], 3, 3),
            shape(-1, output_channels[4], 1, 1)]
        We have added assertion tests to check this, iteration through out_features,
        where out_features[0] should have the shape:
            shape(-1, output_channels[0], 38, 38),
        """
        out_features = []

        for layer in self.output_layers.children(): 
            x = layer(x)
            out_features.append(x)

        for idx, feature in enumerate(out_features):
            out_channel = self.out_channels[idx]
            h, w = self.output_feature_shape[idx]
            expected_shape = (out_channel, h, w)
            assert feature.shape[1:] == expected_shape, \
                f"Expected shape: {expected_shape}, got: {feature.shape[1:]} at output IDX: {idx}"
        assert len(out_features) == len(self.output_feature_shape),\
            f"Expected that the length of the outputted features to be: {len(self.output_feature_shape)}, but it was: {len(out_features)}"
        return tuple(out_features)

