import copy
import math
import torch
from torch import nn
from torch.nn import functional as F

from einops import repeat, rearrange
from einops.layers.torch import Rearrange

import modules

class ViT(nn.Module):
    def __init__(self,
        image_size,
        patch_size,
        num_classes,
        dim,
        depth,
        heads,
        mlp_dim,
        pool = 'cls',
        channels = 3,
        dim_head = 64,
        dropout = 0.,
        emb_dropout = 0.
    ):
        super().__init__()
        
        image_height, image_width = (image_size, image_size)
        patch_height, patch_width = (patch_size, patch_size)
        
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        
        self.transformer = modules.Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Linear(dim, num_classes)
    
    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape
        
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        # pos_embeddings = repeat(self.pos_embedding, '1 n d -> b n d', b = b)
        x = x + self.pos_embedding
        x = self.dropout(x)
        
        x = self.transformer(x)
        
        x = x[:, 1:, :].mean(dim = 1) if self.pool == 'mean' else x[:, 0]
        
        x = self.to_latent(x)
        return self.mlp_head(x)

    
class MAE(nn.Module):
    def __init__(
        self,
        *,
        encoder,
        decoder_dim,
        masking_ratio = 0.75,
        decoder_depth = 1,
        decoder_heads = 8,
        decoder_dim_head = 64
    ):
        super().__init__()
        assert masking_ratio > 0 and masking_ratio < 1, 'masking ratio must be kept between 0 and 1'
        self.masking_ratio = masking_ratio
        
        self.encoder = encoder
        num_patches, encoder_dim = encoder.pos_embedding.shape[-2:] # The num_patches includes cls tokens
        
        self.to_patch = encoder.to_patch_embedding[0]
        self.patch_to_emb = nn.Sequential(*encoder.to_patch_embedding[1:])
        
        self.pixel_values_per_patch = encoder.to_patch_embedding[2].weight.shape[-1]
        
        # decoder parameters
        self.decoder_dim = decoder_dim
        self.enc_to_dec = nn.Linear(encoder_dim, decoder_dim) if encoder_dim != decoder_dim else nn.Identity()
        self.mask_token = nn.Parameter(torch.randn(decoder_dim))
        self.decoder = modules.Transformer(dim = decoder_dim, depth = decoder_depth, heads = decoder_heads, dim_head = decoder_dim_head, mlp_dim = decoder_dim * 4)
        self.decoder_pos_emb = nn.Parameter(torch.randn(1, num_patches, decoder_dim))
        self.to_pixels = nn.Linear(decoder_dim, self.pixel_values_per_patch)
        
    def forward(self, img):
        device = img.device
        
        _, num_channels, image_height, image_width = img.shape
        
        # get patches
        patches = self.to_patch(img) # (B, h'*w', p1 * p2 * C)
        batch, num_patches, *_ = patches.shape
        
        patch_height = patch_width = int((self.pixel_values_per_patch//num_channels)**.5)
        assert patch_height * patch_width * num_channels == self.pixel_values_per_patch
        
        # patch to encoder tokens and add positions
        tokens = self.patch_to_emb(patches)
        tokens += self.encoder.pos_embedding[:, 1:, :]
        
        # class tokens
        cls_token = self.encoder.cls_token + self.encoder.pos_embedding[:, :1, :]
        cls_tokens = repeat(cls_token, '1 1 d -> b 1 d', b = batch)
            
        # calculate of patches needed to be masked, and get random indices, dividing it up for mask vs unmasked
        num_masked = int(self.masking_ratio * num_patches)
        rand_indices = torch.rand(batch, num_patches, device = device).argsort(dim = -1)
        masked_indices, unmasked_indices = rand_indices[:, :num_masked], rand_indices[:, num_masked:]
        
        # get the unmasked tokens to be encoded
        batch_range = torch.arange(batch, device = device)[:, None]
        tokens = tokens[batch_range, unmasked_indices]
        
        # append class tokens
        tokens = torch.cat((cls_tokens, tokens), dim=1)
        
        # get the patches to be masked for the final reconstruction loss
        masked_patches = patches[batch_range, masked_indices]
        
        # attend with vision transformer
        encoded_tokens = self.encoder.transformer(tokens)
        
        # project encoder to decoder dimensions, if they are not equal - the paper says you can get away with a smaller dimension for decoder
        decoder_tokens = self.enc_to_dec(encoded_tokens)
        
        _decoder_tokens = decoder_tokens[:, 1:, :] # no cls token
        
        # repeat mask tokens for number of masked, and add the positions using the masked indices derived above
        mask_tokens = repeat(self.mask_token, 'd -> b n d', b = batch, n = num_masked)
        
        # concat the masked tokens to the decoder tokens and attend with decoder
        decoder_input = torch.zeros(batch, num_patches, self.decoder_dim, device=device)
        decoder_input[batch_range, unmasked_indices] = _decoder_tokens
        decoder_input[batch_range, masked_indices] = mask_tokens
        decoder_input = torch.cat([decoder_tokens[:, :1, :], decoder_input], dim=1) # append cls token
        decoder_input = decoder_input + self.decoder_pos_emb[:, :, :]
        
        decoded_tokens = self.decoder(decoder_input)
        _decoded_tokens = decoded_tokens[:, 1:, :] # no cls token
        
         # splice out the mask tokens and project to pixel values
        mask_tokens = _decoded_tokens[batch_range, masked_indices]
        pred_pixel_values = self.to_pixels(mask_tokens)
        
        # calculate reconstruction loss
        recon_loss = F.mse_loss(pred_pixel_values, masked_patches)
        
        # reconstruction image
        recon_img = torch.zeros(batch, num_patches, self.pixel_values_per_patch, device=device)
        recon_img[batch_range, unmasked_indices] = patches[batch_range, unmasked_indices]
        recon_img[batch_range, masked_indices] = pred_pixel_values
        
        recon_img = rearrange(recon_img, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', p1 = patch_height, p2 = patch_width, h = image_height // patch_height, w = image_width // patch_width)
        
        return recon_loss, recon_img
