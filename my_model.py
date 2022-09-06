import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.activation import MultiheadAttention
import numpy as np

class my_model1(nn.Module):
    def __init__(self, config):
        super().__init__()
        semantic_embed = np.load("E:\data\iemocap\glove300d_full.npy")
        #print("semantic embeded shape",semantic_embed.shape)
        semantic_embed = np.concatenate([np.zeros([1, semantic_embed.shape[1]]), semantic_embed], axis=0)
        #print("then",semantic_embed.shape)
        self.semantic_embed = nn.Embedding.from_pretrained(torch.FloatTensor(semantic_embed), freeze=False)

        self.semantic_linear = nn.Linear(300, 256)

        # This the embedding for the audio features

        self.acoustic_cnn1 = nn.Conv1d(config['acoustic']['embedding_dim'], 64, 5, 1)
        self.acoustic_cnn2 = nn.Conv1d(64, 128, 2, 1)
        self.acoustic_cnn3 = nn.Conv1d(128, int(config['acoustic']['hidden_dim'] / 2), 2, 1)
        self.acoustic_mean1 = nn.AvgPool2d((1, 5), (1, 1))
        self.acoustic_mean2 = nn.AvgPool2d((1, 2), (1, 1))
        self.acoustic_mean3 = nn.AvgPool2d((1, 2), (1, 1))

        # This the embedding for the semantic features
        self.fuse_lstm = nn.LSTM(512,256, 1, bidirectional=True,
            batch_first=True, dropout=0.5)

        # Add the cross-modal excitement layer

        self.acoustic_excit = nn.Embedding(config['semantic']['embedding_size'] + 1,256)
        self.semantic_excit = nn.Linear(256, 256)

        self.loss_name = "BCE"
        self.classifier = nn.Linear( 2 * 256,4)

    def forward(
            self,
            acoustic_input,
            acoustic_length,
            semantic_input,
            semantic_length,
            align_input, ):
        # first perform the encode for the first-step semantic partterns
        semantic_embed = self.semantic_embed(semantic_input)  # [B,T,C]
        semantic_embed = self.semantic_linear(semantic_embed)
        # first perform the encode for the first-step acoustic partterns
        acoustic_embed = self.acoustic_cnn1(acoustic_input.permute(0, 2, 1))

        acoustic_align = self.acoustic_mean1(align_input[:, None, :, :])

        acoustic_embed = self.acoustic_cnn2(acoustic_embed)
        acoustic_align = self.acoustic_mean2(acoustic_align)

        acoustic_embed = self.acoustic_cnn3(acoustic_embed)  # [B,C,A]
        acoustic_align = self.acoustic_mean3(acoustic_align)  # [B,1,T,A]

        acoustic_embed = acoustic_embed.permute(0, 2, 1)[:, None, :, :].repeat(1, semantic_embed.size(1), 1,
                                                                               1)  # [B,T,A,C]
        acoustic_align = (acoustic_align.squeeze(1)[:, :, :, None] > 0).float()  # [B,T,A,1]

        # Think about the new way of calculate the mean

        acoustic_mean = torch.sum(acoustic_embed * acoustic_align, dim=2) / (
                    torch.sum(acoustic_align, dim=2) + 1e-6)  # [B,T,C]
        # Think about the new way of calculate the max
        acoustic_max, _ = torch.max(acoustic_embed * acoustic_align - 1e6 * (1.0 - acoustic_align), dim=2)
        # concat it with both embed
        acoustic_embed = torch.cat([acoustic_mean, acoustic_max], dim=-1)

        # then we use the cross modal excitement information

        semantic_excit = F.sigmoid(self.semantic_excit(acoustic_embed))
        semantic_embed = semantic_embed * semantic_excit + semantic_embed  # These two lines are different, we add the residual connection

        acoustic_excit = F.sigmoid(self.acoustic_excit(semantic_input))
        acoustic_embed = acoustic_embed * acoustic_excit + acoustic_embed  # These two lines are different, we add the residual connection

        fuse_embed = torch.cat([semantic_embed, acoustic_embed], dim=2)
        # Then we use the fuse lstm to encode the multimodal information
        fuse_pack = nn.utils.rnn.pack_padded_sequence(
            fuse_embed, semantic_length.cpu(), batch_first=True, enforce_sorted=False
        )
        fuse_embed, _ = self.fuse_lstm(fuse_pack)
        fuse_embed, _ = nn.utils.rnn.pad_packed_sequence(
            fuse_embed, batch_first=True
        )
        # Here we get the final results, we use the max pooling to generate the results
        fuse_mask = torch.arange(
            semantic_input.size(1))[None, :].repeat(semantic_input.size(0), 1
                                                    ).to(semantic_input.device)
        fuse_mask = (fuse_mask < semantic_length[:, None].repeat(1, semantic_input.size(1))).float()

        if self.loss_name == 'BCE':
            fuse_embed = fuse_embed - (1 - fuse_mask[:, :, None]) * 1e6
            fuse_embed = torch.max(fuse_embed, dim=1)[0]
            logits = self.classifier(fuse_embed)

        elif self.loss_name == 'CTC':
            logits = self.classifier(fuse_embed)  # [B,T,Dim]
            logits = F.log_softmax(logits)
            logits = logits * fuse_mask[:, :, None]

        return logits
