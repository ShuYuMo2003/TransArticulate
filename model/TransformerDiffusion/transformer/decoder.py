import torch
from torch import nn
from rich import print
from functools import reduce

from .layers.decoder_layer import DecoderLayer
from .layers.post_encoder import PostEncoder, ResnetBlockFC
from .layers.token import MLPTokenizer, MLPUnTokenizer
from .layers.position import PositionGRUEmbedding
from .layers.vq_embedding import VQEmbedding

class TransformerDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.device = config['device']

        self.config = config
        self.part_structure = self.config['part_structure']
        self.m_config = self.config['transformer_model_paramerter']
        self.d_model = self.m_config['d_model']
        self.vq_dim = self.m_config['vq_expand_dim']

        d_token_latencode = sum(
            [v for k, v in self.part_structure.items() if k != 'condition']
        )

        d_token_condition = sum(
            [v for k, v in self.part_structure.items() if k != 'latentcode']
        )

        self.dim_latent = self.part_structure['latentcode']
        self.dim_condition = self.part_structure['condition']

        self.position_embedding = PositionGRUEmbedding(d_model=self.d_model,
                                                       dim_single_emb=self.m_config['position_embedding_dim_single_emb'],
                                                       dropout=self.m_config['position_embedding_dropout'])


        self.expand_latent_dim = reduce(lambda x, y: x * y, self.m_config['vq_expand_dim'])

        self.latentcode_encoder = nn.Sequential(*[
            ResnetBlockFC(self.expand_latent_dim, 0.1)
            for _ in range(self.m_config['before_vq_net_deepth'])
        ])

        self.latentcode_expand_fc = nn.Linear(self.dim_latent,  self.expand_latent_dim)
        self.vq_embedding   = VQEmbedding(self.m_config['n_embed'], self.m_config['vq_expand_dim'][0], beta=self.m_config['vq_beta'])
        self.latentcode_to_condition = nn.Linear(self.expand_latent_dim, self.dim_condition)

        self.tokenizer      = MLPTokenizer(d_token=d_token_condition,
                                           d_hidden=self.m_config['tokenizer_hidden_dim'],
                                           d_model=self.d_model,
                                           drop_out=self.m_config['tokenizer_dropout'])

        self.untokenizer    = MLPUnTokenizer(d_token=d_token_condition,
                                             d_hidden=self.m_config['tokenizer_hidden_dim'],
                                             d_model=self.d_model,
                                             drop_out=self.m_config['tokenizer_dropout'])

        self.postencoder    = PostEncoder(dim=self.m_config['encoder_kv_dim'], d_model=self.d_model,
                                          dropout=self.m_config['post_encoder_dropout'],
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

        batch, n_part, _ = input['token'].size()

        # import pdb; pdb.set_trace()

        # Convert latent code to condition
        latents = input['token'][:, :, -self.dim_latent:]
        expand_latents = self.latentcode_expand_fc(latents)
        expand_latents = self.latentcode_encoder(expand_latents)
        expand_latents = expand_latents.view(batch * n_part, *self.vq_dim)
        vq_loss, expand_latents, perplexity, min_encodings, min_encoding_indices  \
                = self.vq_embedding(expand_latents)
        expand_latents = expand_latents.view(batch, n_part, -1)
        condition = self.latentcode_to_condition(expand_latents)
        input['token'] = torch.cat((input['token'][..., :-self.dim_latent], condition), dim=-1)

        # Tokenize the input
        input['token'] = self.tokenizer(input['token'])

        tokens = self.position_embedding(input)

        attn_mask = self.generate_mask(n_part)

        for idx, layer in enumerate(self.layers):
            # tokens = layer(tokens, padding_mask, attn_mask, enc_data, None)
            tokens = layer(tokens, padding_mask, attn_mask, enc_data)

        tokens = self.untokenizer(tokens)
        return tokens, vq_loss