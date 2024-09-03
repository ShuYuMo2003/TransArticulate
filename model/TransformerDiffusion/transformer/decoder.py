import torch
from torch import nn
from rich import print

from .layers.decoder_layer import DecoderLayer
from .layers.post_encoder import PostEncoder
from .layers.token import MLPTokenizer, MLPUnTokenizer

class TransformerDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.device = config['device']

        self.config = config
        self.part_structure = self.config['part_structure']
        self.m_config = self.config['transformer_model_paramerter']
        self.d_model = self.m_config['d_model']

        d_token_latencode = sum(
            [v for k, v in self.part_structure.items() if k != 'condition']
        )

        d_token_condition = sum(
            [v for k, v in self.part_structure.items() if k != 'latentcode']
        )

        self.tokenizer      = MLPTokenizer(d_token=d_token_latencode,
                                           d_hidden=self.m_config['tokenizer_hidden_dim'],
                                           d_model=self.d_model,
                                           drop_out=self.m_config['tokenizer_dropout'])

        self.untokenizer    = MLPUnTokenizer(d_token=d_token_condition,
                                             d_hidden=self.m_config['tokenizer_hidden_dim'],
                                             d_model=self.d_model,
                                             drop_out=self.m_config['tokenizer_dropout'])

        self.postencoder    = PostEncoder(dim=self.m_config['encoder_kv_dim'], d_model=self.d_model,
                                          deepth=self.m_config['post_encoder_deepth'])

        self.layers         = nn.ModuleList([
            DecoderLayer(config)
                for _ in range(self.m_config['n_layer'])
        ])


    def generate_mask(self, n_part):
        mask = torch.ones(n_part, n_part, device=self.device, dtype=torch.int16)
        # mask = torch.tril(mask) # no need mask
        return mask

    def forward(self, input, padding_mask, enc_data):
        # ('token'/'dfn'/'dfn_fa') * batch * part_idx * attribute_dim
        enc_data = self.postencoder(enc_data)
        tokens = input['token']

        batch, n_part, d_model = tokens.size()

        tokens = self.tokenizer(tokens)
        attn_mask = self.generate_mask(n_part)

        for idx, layer in enumerate(self.layers):
            # tokens = layer(tokens, padding_mask, attn_mask, enc_data, None)
            tokens = layer(tokens, padding_mask, attn_mask, enc_data)

        tokens = self.untokenizer(tokens)
        return tokens