import bmtrain as bmt
import torch.nn as nn
from .linear import Linear


class LowRankLinear(bmt.DistributedModule):
    #  ------------------------------------------------------------------------------------------
    #  Copyright (c) Microsoft Corporation. All rights reserved.
    #  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
    #  ------------------------------------------------------------------------------------------
    #  copy from loralib and do some refactor
    def __init__(self,
        in_features,
        out_features,
        r=8,
        lora_alpha=16,
        lora_dropout=0.0,
    ):
        super().__init__()
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        if r > 0:
            self.lora_A = Linear(dim_out = r, dim_in = in_features, cps = 2)
            self.lora_B = Linear(dim_out = out_features, dim_in = r, cps = 1)
            self.scaling = self.lora_alpha / self.r

    def forward(self, x):
        return (self.lora_B(self.lora_A(self.lora_dropout(x)))) * self.scaling

