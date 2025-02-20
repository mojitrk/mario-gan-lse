import torch
import torch.nn as nn
import torch.nn.functional as F

class ProgressiveBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ProgressiveBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )
        self.attention = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.Sigmoid()
        )
        self.residual = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        x = self.conv(x)
        attention = self.attention(x)
        x = x * attention
        x = x + self.residual(x)
        return x

class PlayabilityModule(nn.Module):
    def __init__(self):
        super(PlayabilityModule, self).__init__()
        self.path_conv = nn.Conv2d(1, 1, kernel_size=3, padding=1)
        self.jump_conv = nn.Conv2d(1, 1, kernel_size=(3, 5), padding=(1, 2))
        self.enemy_conv = nn.Conv2d(1, 1, kernel_size=3, padding=1)

    def forward(self, x):
        path_score = torch.sigmoid(self.path_conv(x)).mean()
        jump_score = torch.sigmoid(self.jump_conv(x)).mean()
        enemy_score = torch.sigmoid(self.enemy_conv(x)).mean()
        return path_score + jump_score + enemy_score

class Generator(nn.Module):
    def __init__(self, latent_dim, num_conditions):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.num_conditions = num_conditions
        
        # Embedding layer for conditions
        self.label_embedding = nn.Embedding(num_conditions, num_conditions)
        
        # Initial size calculations
        self.initial_size = (13 // 16 + 1, 368 // 16 + 1)
        
        # Enhanced dense processing
        self.dense = nn.Sequential(
            nn.Linear(latent_dim + num_conditions, 512 * self.initial_size[0] * self.initial_size[1]),
            nn.BatchNorm1d(512 * self.initial_size[0] * self.initial_size[1]),
            nn.ReLU(True),
            nn.Dropout(0.2)
        )
        
        # Progressive generation blocks
        self.progression = nn.ModuleList([
            ProgressiveBlock(512, 256),
            ProgressiveBlock(256, 128),
            ProgressiveBlock(128, 64)
        ])
        
        # Game-specific feature extraction
        self.feature_extraction = nn.ModuleDict({
            'platform': nn.Conv2d(64, 64, kernel_size=3, padding=1),
            'gap': nn.Conv2d(64, 64, kernel_size=(1, 5), padding=(0, 2)),
            'enemy': nn.Conv2d(64, 64, kernel_size=3, padding=1)
        })
        
        # Final generation
        self.final = nn.Sequential(
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )
        
        # Playability checker
        self.playability = PlayabilityModule()

    def forward(self, z, labels):
        # Process conditions
        embedded_labels = self.label_embedding(labels)
        z = torch.cat([z, embedded_labels], dim=1)
        
        # Initial processing
        x = self.dense(z)
        x = x.view(x.size(0), 512, self.initial_size[0], self.initial_size[1])
        
        # Progressive generation
        for block in self.progression:
            x = block(x)
        
        # Apply game-specific features
        platform_features = self.feature_extraction['platform'](x)
        gap_features = self.feature_extraction['gap'](x)
        enemy_features = self.feature_extraction['enemy'](x)
        
        x = x + platform_features + gap_features + enemy_features
        
        # Final generation
        x = self.final(x)
        
        # Calculate playability score
        playability_score = self.playability(x)
        
        return x, playability_score

class MultiScaleDiscriminator(nn.Module):
    def __init__(self, num_conditions):
        super(MultiScaleDiscriminator, self).__init__()
        self.num_conditions = num_conditions
        
        # Condition embedding
        self.label_embedding = nn.Embedding(num_conditions, num_conditions)
        
        # Multi-scale discrimination
        self.scales = nn.ModuleList([
            self._make_discriminator_block(scale_factor=1),
            self._make_discriminator_block(scale_factor=2),
            self._make_discriminator_block(scale_factor=4)
        ])
        
        self.downsample = nn.AvgPool2d(2)
        self.final_merge = nn.Sequential(
            nn.Conv2d(3, 1, kernel_size=1),
            nn.AdaptiveAvgPool2d(1)
        )
        
    def _make_discriminator_block(self, scale_factor):
        return nn.Sequential(
            nn.Conv2d(1 + self.num_conditions, 64 * scale_factor, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64 * scale_factor, 128 * scale_factor, 3, stride=2, padding=1),
            nn.BatchNorm2d(128 * scale_factor),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128 * scale_factor, 256 * scale_factor, 3, stride=2, padding=1),
            nn.BatchNorm2d(256 * scale_factor),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256 * scale_factor, 1, 3, stride=1, padding=1)
        )

    def forward(self, x, labels):
        # Process conditions
        embedded_labels = self.label_embedding(labels)
        embedded_labels = embedded_labels.view(embedded_labels.size(0), self.num_conditions, 1, 1)
        embedded_labels = embedded_labels.expand(-1, -1, x.size(2), x.size(3))
        
        # Multi-scale discrimination
        results = []
        current_x = x
        
        for discriminator in self.scales:
            # Concatenate input and conditions
            current_input = torch.cat([current_x, embedded_labels], dim=1)
            results.append(discriminator(current_input))
            current_x = self.downsample(current_x)
            embedded_labels = self.downsample(embedded_labels)
        
        # Merge results
        merged = torch.cat([F.interpolate(r, size=results[0].shape[2:]) for r in results], dim=1)
        output = self.final_merge(merged)
        
        return torch.sigmoid(output.view(-1))

class Discriminator(MultiScaleDiscriminator):
    def __init__(self, num_conditions):
        super(Discriminator, self).__init__(num_conditions)