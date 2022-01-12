import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv,self).__init__()
        self.conv = nn.Sequential(
            # Here, we use "SAME" CONV. In the original paper, we use "VALID"
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False), # no bias cuz we're using batchnorm
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False), # no bias cuz we're using batchnorm
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        
    def forward(self, x):
        return self.conv(x)

    
class UNET(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]): # out_channels = 1 cuz we're on Kirvada datasets which has only 2 classes -> sigmoid binary classification
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # IMPORTANT: Becareful of the input size -> choose dim that's divisible by 16 since there's 4 steps
        
        # Down Part of U-Net
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature)) # add layer to this modulelist
            in_channels = feature
            
        # Up Part of U-Net
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2 
                )
            )
            self.ups.append(DoubleConv(feature*2, feature))
        
        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        
        # Final Convolution -> # classes
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1) # out_channels = 1 in this task
        
    def forward(self, x):
        skip_connections = []
        
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2] # because our step = 2
            
            # in case concat dimensions don't match!
            if x.shape != skip_connection.shape: 
                x = TF.resize(x, size=skip_connection.shape[2:]) # [batch_size, # channels, h, w] -> h, w
                
            concat_skip = torch.cat((skip_connection, x), dim=1) # Since, we used "SAME" CONV, dimensions match for concat
            x = self.ups[idx+1](concat_skip) # Then, put it to doubleConv
            
        return self.final_conv(x)