from typing import Sequence

import numpy as onp

import jax
from jax import vmap
from jax import numpy as jnp
from jax.random import split, PRNGKey

from flax import struct
from flax import linen as nn

@struct.dataclass
class TransformerConfig:
    """Global hyperparameters used to minimize obnoxious kwarg plumbing."""

    num_heads: int = 4
    num_enc_layers: int = 1
    num_dec_layers: int = 1
    dropout_rate: float = 0.1
    deterministic: bool = False
    d_model: int = 40
    add_positional_encoding: bool = False
    max_len: int = 2000  # positional encoding
    obs_emb_hidden_sizes: Sequence[int] = (100,)
    num_latents: int = 1


class ObsEmbed(nn.Module):
    config: TransformerConfig

    @nn.compact
    def __call__(self, x):
        for feat in self.config.obs_emb_hidden_sizes:
            x = nn.Dense(feat)(x)
            x = nn.relu(x)

        x = nn.Dense(self.config.d_model)(x)

        return x


class PositionalEncoder(nn.Module):
    config: TransformerConfig
    # todo: shall we add dropout? there's documentation to read though.

    @staticmethod
    def init_pe(d_model: int, max_length: int):
        positions = jnp.arange(max_length)[:, None]
        div_term = jnp.exp(jnp.arange(0, d_model) * (-jnp.log(10000.0) / d_model))

        temp = positions * div_term
        even_mask = positions % 2 == 0

        pe = jnp.where(even_mask, jnp.sin(temp), jnp.cos(temp))

        return pe[None, :, :]

    @nn.compact
    def __call__(self, x):
        cfg = self.config
        pe = self.variable(
            "consts", "pe", PositionalEncoder.init_pe, cfg.d_model, cfg.max_len
        )
        # batch_apply_pe = nn.vmap(lambda x, pe: x + pe[:x.shape[0]], in_axes=(0,None))
        return x + pe.value[:, : x.shape[1]]


class MlpBlock(nn.Module):
    """Transformer MLP / feed-forward block.
    Args:
      config: TransformerConfig dataclass containing hyperparameters.
    """

    config: TransformerConfig

    @nn.compact
    def __call__(self, inputs):
        """Applies Transformer MlpBlock module."""
        cfg = self.config
        x = nn.Dense(cfg.d_model * 2)(inputs)
        x = nn.relu(x)
        x = nn.Dropout(rate=cfg.dropout_rate)(x, deterministic=cfg.deterministic)
        output = nn.Dense(cfg.d_model)(x)
        output = nn.Dropout(rate=cfg.dropout_rate)(
            output, deterministic=cfg.deterministic
        )
        return output


class EncoderLayer(nn.Module):
    """Transformer encoder layer.
    Args:
      config: TransformerConfig dataclass containing hyperparameters.
    """

    config: TransformerConfig

    @nn.compact
    def __call__(self, inputs):
        """Applies EncoderBlock module.
        Args:
          inputs: input data for decoder
        Returns:
          output after transformer encoder block.
        """
        cfg = self.config

        # Encoder block.
        assert inputs.ndim == 3
        x = nn.LayerNorm()(inputs)
        x = nn.SelfAttention(
            num_heads=cfg.num_heads,
            use_bias=False,  # should we use bias? I guess it doesn't matter
            broadcast_dropout=False,
            dropout_rate=cfg.dropout_rate,
            deterministic=cfg.deterministic,
            decode=False,
        )(x)
        x = nn.Dropout(rate=cfg.dropout_rate)(x, deterministic=cfg.deterministic)
        x = x + inputs

        # MLP block.
        z = nn.LayerNorm()(x)
        z = MlpBlock(config=cfg)(z)

        return x + z


class DecoderLayer(nn.Module):
    """Transformer encoder-decoder layer.
    Args:
      config: TransformerConfig dataclass containing hyperparameters.
    """

    config: TransformerConfig

    @nn.compact
    def __call__(self, output_emb, encoded_input):
        """Applies EncoderBlock module.
        Args:
          inputs: input data for decoder
        Returns:
          output after transformer encoder block.
        """

        cfg = self.config

        # Decoder block.
        assert encoded_input.ndim == 3 and output_emb.ndim == 3
        x = nn.LayerNorm()(output_emb)
        x = nn.SelfAttention(
            num_heads=cfg.num_heads,
            use_bias=False,
            broadcast_dropout=False,
            dropout_rate=cfg.dropout_rate,
            deterministic=cfg.deterministic,
            decode=False,
        )(x)
        x = nn.Dropout(rate=cfg.dropout_rate)(x, deterministic=cfg.deterministic)
        x = x + output_emb

        z = nn.LayerNorm()(x)

        x = nn.MultiHeadDotProductAttention(
            num_heads=cfg.num_heads,
            use_bias=False,
            broadcast_dropout=False,
            dropout_rate=cfg.dropout_rate,
            deterministic=cfg.deterministic,
            decode=False,
        )(z, encoded_input)

        x = x + z

        # MLP block.
        z = nn.LayerNorm()(x)

        z = MlpBlock(config=cfg)(z)

        return x + z


class TransformerStack(nn.Module):
    config: TransformerConfig

    @nn.compact
    def __call__(self, q):
        assert len(q.shape) == 3
        cfg = self.config
        x = ObsEmbed(cfg)(q)
        enc_input = PositionalEncoder(cfg)(x) if cfg.add_positional_encoding else x

        # for _ in range(cfg.num_enc_layers):
        enc_input = nn.Sequential(
            [EncoderLayer(cfg) for _ in range(cfg.num_enc_layers)]
        )(enc_input)

        start = self.param("start", nn.initializers.uniform(), (1, 1, cfg.d_model))
        so_far_dec = start.repeat(q.shape[0], axis=0)

        dec = so_far_dec
        for _ in range(cfg.num_dec_layers):
            dec = DecoderLayer(cfg)(dec, enc_input)
        
        num_params = cfg.num_latents
        dist_params = nn.Sequential(
            [nn.Dense(cfg.d_model * 2), nn.relu, nn.Dense(num_params)]
        )(dec)
        
        assert dist_params.shape[1] == 1
        return dist_params[:,0]