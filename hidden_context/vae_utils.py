import math
from functools import partial

import ipdb
import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR
import torch.nn.functional as F
from torch.utils.data import DataLoader, SequentialSampler, Sampler, RandomSampler
from typing import Sized, Optional, Iterator
import torch.nn as nn
from transformers import Trainer, EvalPrediction
import wandb
from hidden_context.transformer_utils import *
from transformers.optimization import get_cosine_schedule_with_warmup


class PairEncoder(nn.Module):
    """
    Model to encode pairs of accepted and rejected responses
    """

    def __init__(self, embed_dim, hidden_dim, output_dim):
        super(PairEncoder, self).__init__()

        self._model = nn.Sequential(
            nn.Linear(2 * embed_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, e_c, e_r):
        x = torch.cat([e_c, e_r], dim=1)
        return self._model(x)


class SequenceEncoder(nn.Module):
    """
    Model to encode sequence of responses
    """

    def __init__(self, input_dim, latent_dim):
        super(SequenceEncoder, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.linear = nn.Identity()  # TODO: Do we need linear layer?
        self.w_q = nn.Linear(input_dim, input_dim)
        self.w_k = nn.Linear(input_dim, input_dim)
        self.w_v = nn.Linear(input_dim, input_dim)
        # self.w_q = nn.Identity()
        # self.w_k = nn.Identity()
        self.mean_layer = nn.Linear(input_dim, latent_dim)
        self.log_var_layer = nn.Linear(input_dim, latent_dim)
        self.layer_norm = nn.Identity()     # nn.LayerNorm(latent_dim)      # todo: add LayerNorm?

    def forward(
        self, sequences, seq_start_end
    ):  # (C_1+C_2+...+C_n, D), [(0, C_1), (C_1, C_1+C_2), ..., (C_1+...+C_n-1, C_1+...+C_n)]
        outputs = []
        for _, (start, end) in enumerate(seq_start_end):
            context = sequences[start:end]  # C_i x D
            q = self.w_q(context)
            k = self.w_k(context)
            attention_scores = torch.matmul(
                q, k.transpose(0, 1)
            )
            # transformed_seq = self.linear(context)  # C_i x D'
            # attention_scores = torch.matmul(
            #     transformed_seq, transformed_seq.transpose(0, 1)
            # )  # C_i x C_i
            attention_scores = attention_scores / (context.shape[-1] ** 0.5)
            attention_weights = F.softmax(attention_scores, dim=-1)  # C_i x C_i
            weighted_values = torch.matmul(attention_weights, self.w_v(context))  # C_i x D
            output = torch.mean(weighted_values, dim=0)  # D
            outputs.append(output)
        outputs = torch.stack(outputs, dim=0)  # n x D

        mean = self.layer_norm(self.mean_layer(outputs))
        log_var = self.layer_norm(self.log_var_layer(outputs))
        return mean, log_var


class MLPSequenceEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLPSequenceEncoder, self).__init__()
        self.hidden_layer = nn.Linear(input_dim, hidden_dim)
        self.mean_layer = nn.Linear(hidden_dim, output_dim)
        self.log_var_layer = nn.Linear(hidden_dim, output_dim)
        self.layer_norm = nn.Identity()
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, sequences, seq_start_end):
        inputs = sequences.view(-1, 8, sequences.shape[-1])
        inputs = inputs.view(inputs.shape[0], -1)
        hidden = self.hidden_layer(inputs)
        hidden = self.activation(hidden)
        mean = self.mean_layer(hidden)
        log_var = self.log_var_layer(hidden)
        return mean, log_var



class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Decoder, self).__init__()
        self._model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, xc, xr, z):
        xc = torch.cat([xc, z], dim=1)
        xr = torch.cat([xr, z], dim=1)
        rc = self._model(xc)
        rr = self._model(xr)
        return rc, rr


class Attention(nn.Module):
    def __init__(self, input_dim, latent_dim, output_dim):
        super(Attention, self).__init__()
        self.w_q = nn.Linear(latent_dim, output_dim)     # mapping z to q
        # self.w_k = nn.Linear(input_dim, output_dim)     # mapping tokens to k
        # self.w_v = nn.Linear(input_dim, output_dim)     # mapping tokens to v
        self.w_k = nn.Identity()
        self.w_v = nn.Identity()


    def forward(self, x, z, mask=None):
        q = self.w_q(z.unsqueeze(-2))     # 1 x d
        k = self.w_k(x)     # l x d
        v = self.w_v(x)     # l x d'
        attention_scores = torch.matmul(q, k.transpose(-2, -1))     # 1 x l
        attention_scores = attention_scores / (x.shape[-1] ** 0.5)  # 1 x l
        if mask is not None:
            attention_scores = attention_scores * mask
        attention_weights = F.softmax(attention_scores, dim=-1)     # 1 x l
        weighted_values = torch.matmul(attention_weights, v)        # 1 x d'

        return weighted_values


def make_transformer_encoder(d_model, N=2, h=8, d_ff=4096, dropout=0.1):
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    return clones(TransformerLayer(d_model, c(attn), c(ff), dropout), N)


class TransformerSequenceEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim, concat=False):
        super(TransformerSequenceEncoder, self).__init__()
        if not concat:
            input_dim = input_dim * 2
        self.transformer_encoder = make_transformer_encoder(input_dim)
        self.mean_layer = nn.Linear(input_dim, latent_dim)
        self.log_var_layer = nn.Linear(input_dim, latent_dim)
        self.layer_norm = nn.LayerNorm(latent_dim)
        self.concat = concat

    def forward(self, context_chosen, context_rejected, seq_start_end):
        outputs = []
        for _, (start, end) in enumerate(seq_start_end):
            chosen = context_chosen[start:end]      # C_i x D
            rejected = context_rejected[start:end]  # C_i x D
            if not self.concat:
                context = torch.cat([chosen, rejected], dim=1).unsqueeze(0)      # 1 x C_i x 2D
            else:
                context = chosen
            for layer in self.transformer_encoder:
                context = layer(context)
            context_embedding = torch.mean(context[0], dim=0)     # 2D
            outputs.append(context_embedding)
        outputs = torch.stack(outputs, dim=0)  # n x 2D

        mean = self.layer_norm(self.mean_layer(outputs))
        log_var = self.layer_norm(self.log_var_layer(outputs))
        return mean, log_var


class VAEModel(nn.Module):
    def __init__(self, encoder_embed_dim, decoder_embed_dim, hidden_dim, latent_dim, llm_encoder, llm_contexts_encoder,
                 fixed_contexts=False, fixed_llm_embeddings=False, use_causal_lm=False, use_attention_layer=False,
                 use_transformer=False, concat_chosen_rejected=False):
        super(VAEModel, self).__init__()
        self.llm_encoder = llm_encoder
        self.llm_contexts_encoder = llm_contexts_encoder
        if concat_chosen_rejected:
            if not use_transformer:
                # TODO: fixed length=8 now!!!
                self.sequence_encoder = MLPSequenceEncoder(encoder_embed_dim * 8, 4096, latent_dim)
            else:
                self.sequence_encoder = TransformerSequenceEncoder(encoder_embed_dim, latent_dim, concat=True)
        else:
            if not use_transformer:
                self.pair_encoder = PairEncoder(encoder_embed_dim, hidden_dim, latent_dim)
                self.sequence_encoder = SequenceEncoder(latent_dim, latent_dim)
            else:
                self.sequence_encoder = TransformerSequenceEncoder(encoder_embed_dim, latent_dim, concat=False)
        self.decoder = Decoder(decoder_embed_dim + latent_dim, hidden_dim)
        # self.decoder = Decoder(embed_dim, hidden_dim)

        self.latent_dim = latent_dim
        self.fixed_contexts = fixed_contexts
        self.fixed_llm_embeddings = fixed_llm_embeddings
        self.use_causal_lm = use_causal_lm
        self.use_attention_layer = use_attention_layer
        self.use_transformer = use_transformer
        self.concat_chosen_rejected = concat_chosen_rejected
        if self.use_attention_layer:
            self.attention_layer = Attention(decoder_embed_dim, latent_dim, decoder_embed_dim)

        self.use_vqvae = False
        self.saved_embeddings = torch.Tensor(4, latent_dim)
        self.saved_embeddings.uniform_(-1, 1)
        if self.use_vqvae:
            self.n_embeddings = 2
            init_bound = 1 / self.n_embeddings
            embedding = torch.Tensor(self.n_embeddings, latent_dim)
            embedding.uniform_(-init_bound, init_bound)
            self.embedding = torch.nn.Parameter(embedding, requires_grad=True)
            # init_bound = 1 / n_embeddings
            # embedding = torch.Tensor(n_embeddings, latent_dim)
            # embedding.uniform_(-init_bound, init_bound)
            # self.register_buffer("embedding", embedding)
            # self.register_buffer("ema_count", torch.zeros(n_embeddings))
            # self.register_buffer("ema_weight", self.embedding.clone())
            self.commitment_cost = 0.0
            self.decay = 0.999
            self.epsilon = 1e-5

    def reparameterization(self, mean, std):
        epsilon = torch.randn_like(std).to(mean.device)  # sampling epsilon
        # epsilon *= 0     # TODO: set scale of variance here
        z = mean + std * epsilon                         # reparameterization trick
        z = F.normalize(z, p=2, dim=-1) * math.sqrt(z.shape[-1])
        return z

    def encode_pair(self, e_c, e_r):
        return self.pair_encoder(e_c, e_r)

    def encode_sequence(self, sequences, seq_start_end):
        return self.sequence_encoder(sequences, seq_start_end)

    def decode(self, e_c, e_r, z):
        return self.decoder(e_c, e_r, z)

    def forward(
        self,
        target_chosen,
        target_rejected,
        context_chosen,
        context_rejected,
        seq_start_end,
        user_type,
        ground_truth_user_vector=False,
        mask_chosen=None,
        mask_rejected=None,
    ):
        if self.concat_chosen_rejected:
            if not self.use_transformer:
                # TODO: fixed length=8 now!!!
                mean, log_var = self.sequence_encoder(context_chosen, seq_start_end)
            else:
                mean, log_var = self.sequence_encoder(context_chosen, context_rejected, seq_start_end)
        else:
            if not self.use_transformer:
                pair_embed = self.encode_pair(context_chosen, context_rejected)

                mean, log_var = self.encode_sequence(pair_embed, seq_start_end)
            else:
                mean, log_var = self.sequence_encoder(context_chosen, context_rejected, seq_start_end)

        if self.use_vqvae:
            # ipdb.set_trace()
            M, D = self.embedding.size()
            x_flat = mean.detach().reshape(-1, D)
            distances = (-torch.cdist(x_flat, self.embedding, p=2)) ** 2
            indices = torch.argmin(distances.float(), dim=-1)
            encodings = F.one_hot(indices, M).float()
            quantized = F.embedding(indices, self.embedding)
            quantized = quantized.view_as(mean)

            # if self.training:
            #     self.ema_count = self.decay * self.ema_count + (1 - self.decay) * torch.sum(encodings, dim=0)
            #     n = torch.sum(self.ema_count)
            #     self.ema_count = (self.ema_count + self.epsilon) / (n + M * self.epsilon) * n
            #
            #     dw = torch.matmul(encodings.t(), x_flat)
            #     self.ema_weight = self.decay * self.ema_weight + (1 - self.decay) * dw
            #     self.embedding = self.ema_weight / self.ema_count.unsqueeze(-1)

            codebook_loss = F.mse_loss(mean.detach(), quantized) * 1e-2
            e_latent_loss = F.mse_loss(mean, quantized.detach())
            commitment_loss = 1e-2 * e_latent_loss
            # self.commitment_cost
            quantized = mean + (quantized - mean).detach()
            # import pdb; pdb.set_trace()
            avg_probs = torch.mean(encodings, dim=0)
            perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
            rc, rr = self.decode(target_chosen, target_rejected, quantized)
            return rc, rr, mean, quantized, commitment_loss, codebook_loss, perplexity, avg_probs, indices

        mean = torch.clamp(mean, -1, 1)

        # Version 1
        _log_var = torch.clamp(log_var, -1, 1)
        if ground_truth_user_vector:
            # z = (user_type * 2 - 1).unsqueeze(1).repeat(1, mean.shape[1])       # todo: change ground-truth implementation
            # z = torch.zeros_like(mean).reshape(mean.shape[0], 4, -1)
            # for idx in range(user_type.shape[0]):
            #     z[idx][int(user_type[idx])] += 1
            # z = z.reshape(mean.shape[0], -1)
            z = torch.zeros_like(mean)
            self.saved_embeddings = self.saved_embeddings.to(mean.device)
            for idx in range(user_type.shape[0]):
                z[idx] = self.saved_embeddings[int(user_type[idx])]
        else:
            z = self.reparameterization(mean, torch.exp(0.5 * _log_var))
            # target_z = torch.zeros([4, 4, mean.shape[-1] // 4], dtype=z.dtype).to(z.device)
            # target_z[0, 0] = 1
            # target_z[1, 1] = 1
            # target_z[2, 2] = 1
            # target_z[3, 3] = 1
            # target_z = target_z.reshape(4, -1)
            #
            # weights = F.softmax(torch.tensordot(z, target_z, dims=([1], [1])), dim=-1)
            # z = weights @ target_z

        # Version 2
        # act = torch.nn.Softplus()
        # std = act(log_var) * 1e-3
        # # std = torch.sigmoid(log_var) * 1e-3    # use log_var as std prediction
        # _log_var = torch.log(std ** 2)
        # z = self.reparameterization(mean, std)

        if not self.training and not ground_truth_user_vector:
            z = mean
        if self.use_attention_layer:
            target_chosen = self.attention_layer(target_chosen, z, mask_chosen)
            target_rejected = self.attention_layer(target_rejected, z, mask_rejected)
        rc, rr = self.decode(target_chosen, target_rejected, z)

        return rc, rr, mean, _log_var, z

    def save_model(self, path):
        # state_dict = {
        #     "pair_encoder": self.pair_encoder.state_dict(),
        #     "sequence_encoder": self.sequence_encoder.state_dict(),
        #     "decoder": self.decoder.state_dict(),
        #     "latent_dim": self.latent_dim,
        # }
        torch.save(self, path)

    # def load_model(self, path, llm_encoder):
    #     state_dict = torch.load(path)
    #     self.pair_encoder.load_state_dict(state_dict["pair_encoder"])
    #     self.sequence_encoder.load_state_dict(state_dict["sequence_encoder"])
    #     self.decoder.load_state_dict(state_dict["decoder"])
    #     self.latent_dim = state_dict["latent_dim"]
    #     self.llm_encoder = llm_encoder


def get_contrastive_loss(z, user_type, enable=True):
    #epsilon = 1.0
    #batch_size = z.shape[0]
    #dim = z.shape[1]
    #loss = torch.tensor(0., device=z.device)
    #same_counter = 0
    #different_counter = 0
    #for i in range(batch_size):
    #    for j in range(batch_size):
    #        if i < j:
    #            if user_type[i] == user_type[j]:
    #                same_counter += 1
    #                loss += torch.sum(torch.abs(z[i, :] - z[j, :]) ** 2) / dim
    #            else:
    #                loss += max(epsilon - torch.sum(torch.abs(z[i, :] - z[j, :]) ** 2) / dim,
    #                            torch.tensor(0., device=z.device))
    #loss /= (batch_size * (batch_size - 1) / 2)
    #return loss

    if not enable:
        return torch.tensor(0., device=z.device)
    
    temperature = 0.05
    embeddings = F.normalize(z, p=2, dim=1)
    labels = user_type.unsqueeze(0)
    similarity_matrix = torch.matmul(embeddings, embeddings.T)
    mask = torch.eye(similarity_matrix.size(0), dtype=torch.bool).to(embeddings.device)
    denominator = torch.sum(torch.exp(similarity_matrix / temperature) * (~mask), dim=1, keepdim=True)
    positive_mask = labels == labels.T
    positive_mask = positive_mask & (~mask)
    numerator = torch.sum(similarity_matrix * positive_mask, dim=1) / (torch.sum(positive_mask, dim=1) + 1e-9)
    loss = torch.log(denominator.squeeze(1)) - numerator
    return loss.mean()


class GroupedRandomSampler(Sampler[int]):
    data_source: Sized

    def __init__(self, data_source: Sized, group_size: int) -> None:
        self.data_source = data_source
        self.group_size = group_size

    def __iter__(self) -> Iterator[int]:
        n = len(self.data_source)
        seed = int(torch.empty((), dtype=torch.int64).random_().item())
        generator = torch.Generator()
        generator.manual_seed(seed)
        grouped_indices = torch.randperm(n // self.group_size, generator=generator).tolist()
        indices = list()
        for idx in grouped_indices:
            for i in range(self.group_size):
                indices.append(self.group_size * idx + i)
        yield from indices

    def __len__(self) -> int:
        return len(self.data_source)


class VAETrainer(Trainer):
    def __init__(
        self, *args, lr_lambda=None, kl_loss_weight=None, use_annealing=False, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.lr_lambda = lr_lambda
        self.kl_loss_weight = kl_loss_weight
        self.use_annealing = use_annealing
        self.annealer = Annealer(
            total_steps=1e4, shape="cosine", baseline=0.1, cyclical=True    # todo: change total_step here
        )

    # def get_train_dataloader(self) -> DataLoader:
    #     if self.train_dataset is None:
    #         raise ValueError("Trainer: training requires a train_dataset.")
    #     # train_sampler = GroupedRandomSampler(self.train_dataset, 2)
    #     train_sampler = RandomSampler(self.train_dataset)
    #     return DataLoader(
    #         self.train_dataset,
    #         batch_size=self.args.train_batch_size,
    #         sampler=train_sampler,
    #         collate_fn=self.data_collator,
    #         drop_last=self.args.dataloader_drop_last,
    #     )

    @classmethod
    def per_sample_loss(cls, rewards_chosen, rewards_rejected):
        return -nn.functional.logsigmoid(rewards_chosen - rewards_rejected)

    def loss(self, rewards_chosen, rewards_rejected):
        return torch.mean(self.per_sample_loss(rewards_chosen, rewards_rejected))

    def compute_loss(self, wrapped_model, inputs, return_outputs=False):
        if isinstance(wrapped_model, VAEModel):
            model = wrapped_model  # .module
        else:
            model = wrapped_model.module
        device = model.llm_encoder.device
        batch_size = inputs["seq_start_end"].shape[0]
        if model.fixed_llm_embeddings:
            embeddings_chosen = torch.tensor(inputs["embeddings_chosen"]).to(device).bfloat16()
            embeddings_rejected = torch.tensor(inputs["embeddings_rejected"]).to(device).bfloat16()
        else:
            if model.use_causal_lm:
                input_ids = torch.concatenate(
                        [
                            inputs["input_ids_chosen"],
                            inputs["input_ids_rejected"],
                        ],
                        dim=0,
                    )
                attention_mask = torch.concatenate(
                        [
                            inputs["attention_mask_chosen"],
                            inputs["attention_mask_rejected"],
                        ],
                        dim=0,
                    )
                last_hidden_state = model.llm_encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True
                ).hidden_states[-1]
                masked_last_hidden_state = last_hidden_state * attention_mask.unsqueeze(-1)
                token_length = torch.sum(attention_mask, dim=1)
                embeddings = torch.sum(masked_last_hidden_state, dim=1) / token_length.unsqueeze(-1)
                embeddings_chosen = embeddings[:batch_size]
                embeddings_rejected = embeddings[batch_size:]
            elif model.use_attention_layer:
                embeddings_chosen = model.llm_encoder(
                    input_ids=inputs["input_ids_chosen"],
                    attention_mask=inputs["attention_mask_chosen"],
                    output_hidden_states=True
                ).hidden_states[-1]
                embeddings_rejected = model.llm_encoder(
                    input_ids=inputs["input_ids_rejected"],
                    attention_mask=inputs["attention_mask_rejected"],
                    output_hidden_states=True
                ).hidden_states[-1]
            else:
                embeddings = model.llm_encoder(
                    input_ids=torch.concatenate(
                        [
                            inputs["input_ids_chosen"],
                            inputs["input_ids_rejected"],
                        ],
                        dim=0,
                    ),
                    attention_mask=torch.concatenate(
                        [
                            inputs["attention_mask_chosen"],
                            inputs["attention_mask_rejected"],
                        ],
                        dim=0,
                    ),
                )[0]
                embeddings_chosen = embeddings[:batch_size]
                embeddings_rejected = embeddings[batch_size:]

        if model.fixed_contexts:
            contexts_embeddings_chosen = torch.tensor(inputs["contexts_embeddings_chosen"]).to(device).bfloat16()
            contexts_embeddings_rejected = torch.tensor(inputs["contexts_embeddings_rejected"]).to(device).bfloat16()
        elif model.concat_chosen_rejected:
            input_ids_fused = inputs["contexts_input_ids_fused"]
            attention_mask_fused = inputs["contexts_attention_mask_fused"]
            token_length_chosen = torch.eq(input_ids_fused,
                                           model.llm_contexts_encoder.config.pad_token_id).int().argmax(-1) - 1
            with torch.no_grad():
                last_hidden_state_fused = model.llm_contexts_encoder(
                    input_ids=input_ids_fused,
                    attention_mask=attention_mask_fused,
                    output_hidden_states=True
                ).hidden_states[-1]

                weights_for_non_padding_fused = attention_mask_fused * torch.arange(
                    start=1, end=last_hidden_state_fused.shape[1] + 1
                ).unsqueeze(0).to(attention_mask_fused.device).float()
                sum_embeddings = torch.sum(last_hidden_state_fused * weights_for_non_padding_fused.unsqueeze(-1),
                                           dim=1)
                num_of_none_padding_tokens_fused = torch.sum(weights_for_non_padding_fused, dim=-1).unsqueeze(-1)
                contexts_embeddings_fused = sum_embeddings / num_of_none_padding_tokens_fused
                # token_length_fused = torch.where(token_length_fused >= 0, token_length_fused,
                #                                   last_hidden_state_fused.shape[1] - 1)
                # contexts_embeddings_fused = torch.diagonal(torch.index_select(last_hidden_state_fused, 1, token_length_fused - 1)).T
                contexts_embeddings_chosen = contexts_embeddings_fused
                contexts_embeddings_rejected = contexts_embeddings_fused
        else:
            if model.use_causal_lm:
                last_hidden_state_chosen = model.llm_contexts_encoder(
                    input_ids=inputs["contexts_input_ids_chosen"],
                    attention_mask=inputs["contexts_attention_mask_chosen"],
                    output_hidden_states=True
                ).hidden_states[-1]
                masked_last_hidden_state_chosen = last_hidden_state_chosen * inputs[
                    "contexts_attention_mask_chosen"].unsqueeze(-1)
                token_length_chosen = torch.sum(inputs["contexts_attention_mask_chosen"], dim=1)
                contexts_embeddings_chosen = torch.sum(masked_last_hidden_state_chosen,
                                                       dim=1) / token_length_chosen.unsqueeze(-1)

                last_hidden_state_rejected = model.llm_contexts_encoder(
                    input_ids=inputs["contexts_input_ids_rejected"],
                    attention_mask=inputs["contexts_attention_mask_rejected"],
                    output_hidden_states=True
                ).hidden_states[-1]
                masked_last_hidden_state_rejected = last_hidden_state_rejected * inputs[
                    "contexts_attention_mask_rejected"].unsqueeze(-1)
                token_length_rejected = torch.sum(inputs["contexts_attention_mask_rejected"], dim=1)
                contexts_embeddings_rejected = torch.sum(masked_last_hidden_state_rejected,
                                                         dim=1) / token_length_rejected.unsqueeze(-1)
            else:
                input_ids_chosen = inputs["contexts_input_ids_chosen"]
                attention_mask_chosen = inputs["contexts_attention_mask_chosen"]
                token_length_chosen = torch.eq(input_ids_chosen,
                                               model.llm_contexts_encoder.config.pad_token_id).int().argmax(-1) - 1
                input_ids_rejected = inputs["contexts_input_ids_rejected"]
                attention_mask_rejected = inputs["contexts_attention_mask_rejected"]
                token_length_rejected = torch.eq(input_ids_rejected,
                                                 model.llm_contexts_encoder.config.pad_token_id).int().argmax(-1) - 1

                with torch.no_grad():
                    last_hidden_state_chosen = model.llm_contexts_encoder(
                        input_ids=input_ids_chosen,
                        attention_mask=attention_mask_chosen,
                        output_hidden_states=True
                    ).hidden_states[-1]

                    weights_for_non_padding_chosen = attention_mask_chosen * torch.arange(
                        start=1, end=last_hidden_state_chosen.shape[1] + 1
                    ).unsqueeze(0).to(attention_mask_chosen.device).float()
                    sum_embeddings = torch.sum(last_hidden_state_chosen * weights_for_non_padding_chosen.unsqueeze(-1), dim=1)
                    num_of_none_padding_tokens_chosen = torch.sum(weights_for_non_padding_chosen, dim=-1).unsqueeze(-1)
                    contexts_embeddings_chosen = sum_embeddings / num_of_none_padding_tokens_chosen
                    # token_length_chosen = torch.where(token_length_chosen >= 0, token_length_chosen,
                    #                                   last_hidden_state_chosen.shape[1] - 1)
                    # contexts_embeddings_chosen = torch.diagonal(torch.index_select(last_hidden_state_chosen, 1, token_length_chosen - 1)).T
                    last_hidden_state_rejected = model.llm_contexts_encoder(
                        input_ids=input_ids_rejected,
                        attention_mask=attention_mask_rejected,
                        output_hidden_states=True
                    ).hidden_states[-1]

                    weights_for_non_padding_rejected = attention_mask_rejected * torch.arange(
                        start=1, end=last_hidden_state_rejected.shape[1] + 1
                    ).unsqueeze(0).to(attention_mask_rejected.device).float()
                    sum_embeddings = torch.sum(last_hidden_state_rejected * weights_for_non_padding_rejected.unsqueeze(-1),
                                               dim=1)
                    num_of_none_padding_tokens_rejected = torch.sum(weights_for_non_padding_rejected, dim=-1).unsqueeze(-1)
                    contexts_embeddings_rejected = sum_embeddings / num_of_none_padding_tokens_rejected
                    # token_length_rejected = torch.where(token_length_rejected >= 0, token_length_rejected,
                    #                                     last_hidden_state_rejected.shape[1] - 1)
                    # contexts_embeddings_rejected = torch.diagonal(torch.index_select(last_hidden_state_rejected, 1, token_length_rejected - 1)).T
                    #
                    # contexts_embeddings_chosen = model.llm_contexts_encoder(
                    #     input_ids=inputs["contexts_input_ids_chosen"],
                    #     attention_mask=inputs["contexts_attention_mask_chosen"],
                    # )[0]
                    # contexts_embeddings_rejected = model.llm_contexts_encoder(
                    #     input_ids=inputs["contexts_input_ids_rejected"],
                    #     attention_mask=inputs["contexts_attention_mask_rejected"]
                    # )[0]
        seq_start_end = inputs["seq_start_end"]
        user_type = torch.tensor(inputs["user_type"]).to(device).bfloat16()

        if model.use_vqvae:
            rewards_chosen, rewards_rejected, mean, quantized, commitment_loss, codebook_loss, perplexity, avg_probs, indices = model(
                embeddings_chosen,
                embeddings_rejected,
                contexts_embeddings_chosen,
                contexts_embeddings_rejected,
                seq_start_end,
                user_type,
                ground_truth_user_vector=False,       # todo: set to True for debug usage
                mask_chosen=inputs["attention_mask_chosen"],
                mask_rejected=inputs["attention_mask_rejected"],
            )
            reproduction_loss = self.loss(rewards_chosen, rewards_rejected)
            loss = reproduction_loss + commitment_loss + codebook_loss
            # print("Loss: {}".format(loss))
            # print("Reproduction loss: {}".format(reproduction_loss))
            # print("Commitment loss: {}".format(commitment_loss))
            # print("Perplexity: {}".format(perplexity))
            # print("Codebook loss: {}".format(codebook_loss))
            # print("\n\n")
            accuracy = torch.mean((rewards_chosen > rewards_rejected).float())
            if not return_outputs:
                self.log(
                    {
                        "loss": loss.item(),
                        "accuracy": accuracy.item(),
                        "reproduction_loss": reproduction_loss.item(),
                        "commitment_loss": commitment_loss.item(),
                        "codebook_loss": codebook_loss.item(),
                        "perplexity": perplexity.item(),
                        "avg_probs": avg_probs.detach().cpu().numpy(),
                    }
                )
                return loss
            else:
                if return_outputs:
                    return loss, {
                        "rewards_chosen": rewards_chosen,
                        "rewards_rejected": rewards_rejected,
                        "mean": mean,
                        "log_var": mean,
                        "z": quantized,
                        "user_type": indices,
                    }

        else:
            rewards_chosen, rewards_rejected, mean, log_var, z = model(
                embeddings_chosen,
                embeddings_rejected,
                contexts_embeddings_chosen,
                contexts_embeddings_rejected,
                seq_start_end,
                user_type,
                ground_truth_user_vector=False,       # todo: set to True for debug usage
                mask_chosen=inputs["attention_mask_chosen"],
                mask_rejected=inputs["attention_mask_rejected"],
            )

            reproduction_loss = self.loss(rewards_chosen, rewards_rejected)
            contrastive_loss = get_contrastive_loss(z, user_type)
            if self.kl_loss_weight == 0:
                loss = reproduction_loss + contrastive_loss
                accuracy = torch.mean((rewards_chosen > rewards_rejected).float())
                if not return_outputs:
                    self.log(
                        {
                            "train_recon_loss": reproduction_loss.mean().item(),
                            "train_contrastive_loss": contrastive_loss.mean().item(),
                            "train_accuracy": accuracy.mean().item(),
                            "rewards_chosen": rewards_chosen.mean().item(),
                            "rewards_rejected": rewards_rejected.mean().item(),
                            "embeddings_chosen": embeddings_chosen.mean().item(),
                            "embeddings_rejected": embeddings_rejected.mean().item(),
                            "mean": mean.mean().item(),
                            "log_var": log_var.mean().item()
                        }
                    )
            else:
                kld = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp(), dim=1).mean()
                if self.use_annealing:
                    kld = self.annealer(kld)
                    self.annealer.step()
                kld = self.kl_loss_weight * kld
                loss = reproduction_loss + kld  + 1e-0 * contrastive_loss
                accuracy = torch.mean((rewards_chosen > rewards_rejected).float())
                if not return_outputs:
                    self.log(
                        {
                            "train_recon_loss": reproduction_loss.mean().item(),
                            "train_contrastive_loss": contrastive_loss.mean().item(),
                            "train_kld": kld.mean().item(),
                            "train_accuracy": accuracy.mean().item(),
                            "rewards_chosen": rewards_chosen.mean().item(),
                            "rewards_rejected": rewards_rejected.mean().item(),
                            "embeddings_chosen": embeddings_chosen.mean().item(),
                            "embeddings_rejected": embeddings_rejected.mean().item(),
                            "mean": mean.mean().item(),
                            "log_var": log_var.mean().item()
                        }
                    )
            if return_outputs:
                return loss, {
                    "rewards_chosen": rewards_chosen,
                    "rewards_rejected": rewards_rejected,
                    "mean": mean,
                    "log_var": log_var,
                    "z": z,
                    "user_type": user_type,
                }
            return loss

    def create_scheduler(self, num_training_steps: int, optimizer=None):
        # if self.lr_lambda is not None:
        #     lr_lambda = partial(
        #         self.lr_lambda,
        #         num_training_steps=num_training_steps,
        #     )
        #     self.lr_scheduler = LambdaLR(optimizer, lr_lambda)
        #     return self.lr_scheduler
        # else:
        #     return super().create_scheduler(num_training_steps, optimizer)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.03 * num_training_steps),
            num_training_steps=num_training_steps
        )
        self.lr_scheduler = scheduler
        return scheduler

    @classmethod
    def compute_metrics(cls, eval_prediction: EvalPrediction):
        rewards_chosen, rewards_rejected, mean, log_var, z, user_type = (
            eval_prediction.predictions
        )
        rewards_chosen = torch.from_numpy(rewards_chosen)
        rewards_rejected = torch.from_numpy(rewards_rejected)
        mean = torch.from_numpy(mean)
        log_var = torch.from_numpy(log_var)
        z = torch.from_numpy(z)
        loss = cls.per_sample_loss(rewards_chosen, rewards_rejected)
        kld = -torch.sum(1 + log_var - mean.pow(2) - log_var.exp(), dim=-1)
        contrastive_loss = get_contrastive_loss(z, user_type, enable=False)
        accuracy = torch.mean((loss < np.log(2)).float())

        def plot_latent(latent):
            from sklearn.manifold import TSNE
            z_embedding = TSNE(n_components=2, init='random', perplexity=20, learning_rate="auto").fit_transform(latent.numpy())
            import matplotlib.pyplot as plt
            colors = [f"C{int(i)}" for i in user_type]
            plt.scatter(z_embedding[:, 0], z_embedding[:, 1], c=colors)
            im = wandb.Image(plt)
            plt.close()
            return im
        im1 = plot_latent(mean)
        im2 = plot_latent(z)

        return {
            "loss": loss.mean().item(),
            "accuracy": accuracy.item(),
            "kld": kld.mean().item(),
            # "contrastive_loss": contrastive_loss.mean().item(),
            # # "total_loss": loss.mean().item() + kld.item(),
            "mean_embeddings": im1,
            "z_embeddings": im2,
        }


class Annealer:
    """
    This class is used to anneal the KL divergence loss over the course of training VAEs.
    After each call, the step() function should be called to update the current epoch.
    """

    def __init__(self, total_steps, shape, baseline=0.0, cyclical=False, disable=False):
        """
        Parameters:
            total_steps (int): Number of epochs to reach full KL divergence weight.
            shape (str): Shape of the annealing function. Can be 'linear', 'cosine', or 'logistic'.
            baseline (float): Starting value for the annealing function [0-1]. Default is 0.0.
            cyclical (bool): Whether to repeat the annealing cycle after total_steps is reached.
            disable (bool): If true, the __call__ method returns unchanged input (no annealing).
        """
        self.total_steps = total_steps
        self.current_step = 0
        self.cyclical = cyclical
        self.shape = shape
        self.baseline = baseline
        if disable:
            self.shape = "none"
            self.baseline = 0.0

    def __call__(self, kld):
        """
        Args:
            kld (torch.tensor): KL divergence loss
        Returns:
            out (torch.tensor): KL divergence loss multiplied by the slope of the annealing function.
        """
        out = kld * self.slope()
        return out

    def slope(self):
        if self.shape == "linear":
            y = self.current_step / self.total_steps
        elif self.shape == "cosine":
            y = (math.cos(math.pi * (self.current_step / self.total_steps - 1)) + 1) / 2
        elif self.shape == "logistic":
            exponent = (self.total_steps / 2) - self.current_step
            y = 1 / (1 + math.exp(exponent))
        elif self.shape == "none":
            y = 1.0
        else:
            raise ValueError(
                "Invalid shape for annealing function. Must be linear, cosine, or logistic."
            )
        y = self.add_baseline(y)
        return y

    def step(self):
        if self.current_step < self.total_steps:
            self.current_step += 1
        if self.cyclical and self.current_step >= self.total_steps:
            self.current_step = 0
        return

    def add_baseline(self, y):
        y_out = y * (1 - self.baseline) + self.baseline
        return y_out

    def cyclical_setter(self, value):
        if value is not bool:
            raise ValueError(
                "Cyclical_setter method requires boolean argument (True/False)"
            )
        else:
            self.cyclical = value
        return
