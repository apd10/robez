from typing import Optional

import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
from torch.nn.parameter import Parameter
import math

import hashed_embedding_bag
import pdb
#from torch_sparse import coalesce

class HashedEmbeddingBagFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hashed_weights, indices, offsets, mode, embedding_dim, random_numbers, val_offset, norm, uma_chunk_size, no_bag, sparse):
        if indices.dim() == 2:
            if offsets is not None:
                raise ValueError("if indices is 2D, then offsets has to be None"
                                ", as indices is treated is a mini-batch of"
                                " fixed length sequences. However, found "
                                "offsets of type {}".format(type(offsets)))
            offsets = torch.arange(0, indices.numel(), indices.size(1),
                                dtype=torch.long, device=indices.device)
            indices = indices.reshape(-1)
        elif indices.dim() == 1:
            if offsets is None:
                raise ValueError("offsets has to be a 1D Tensor but got None")
            if offsets.dim() != 1:
                raise ValueError("offsets has to be a 1D Tensor")
        else:
            raise ValueError("indices has to be 1D or 2D Tensor,"
                            " but got Tensor of dimension {}".format(indices.dim()))

        if mode == 'sum':
            mode_enum = 0
        elif mode == 'mean':
            mode_enum = 1
            raise ValueError("mean mode not supported")
        elif mode == 'max':
            mode_enum = 2
            raise ValueError("max mode not supported")

        if val_offset is not None:
            indices = indices + val_offset


        hashed_weights_size = hashed_weights.size(0)
        output, offset2bag, bag_size, max_indices, hashed_idx = \
            hashed_embedding_bag.forward(hashed_weights, indices, offsets, mode_enum, embedding_dim, random_numbers, uma_chunk_size)
        if norm is not None:
            #assert(keymode_enum == 1)
            output = output/norm
        ctx.save_for_backward(indices, offsets, offset2bag, bag_size, max_indices, hashed_idx)
        ctx.mode_enum = mode_enum
        ctx.hashed_weights_size = hashed_weights_size
        ctx.sparse = sparse
        ctx.no_bag = no_bag
        return output

    @staticmethod
    def backward(ctx, grad):
        indices, offsets, offset2bag, bag_size, max_indices, hashed_idx = ctx.saved_variables
        hashed_weights_size = ctx.hashed_weights_size
        mode_enum = ctx.mode_enum
        embedding_dim = grad.size(1)
        if not ctx.no_bag:
            weight_grad = hashed_embedding_bag.backward(
                grad, indices, offsets, offset2bag, bag_size, max_indices, hashed_idx, hashed_weights_size, False, mode_enum, embedding_dim)
        else:
            hashed_idx1 = hashed_idx.reshape(-1)
            grad1 = grad.reshape(-1)
            if ctx.sparse:
                unique, inv_idx = torch.unique(hashed_idx1, return_inverse=True)
                values = torch.zeros(unique.shape, device=indices.device, dtype=torch.float32)
                values.scatter_add_(0,inv_idx, grad1)
                weight_grad = torch.sparse_coo_tensor(unique.view(1, -1), values, (ctx.hashed_weights_size,), device=indices.device)
                #unique, values = coalesce(torch.stack([torch.zeros(hashed_idx1.shape[0], device=indices.device, dtype=hashed_idx1.dtype), hashed_idx1]), grad1, m=1, n=ctx.hashed_weights_size, op='add')
                #weight_grad = torch.sparse_coo_tensor(unique[1].view(1,-1), values, (ctx.hashed_weights_size,), device=indices.device)
            else:
                weight_grad = torch.zeros((hashed_weights_size,),dtype=torch.float32, device=indices.device)
                weight_grad.scatter_add_(0, hashed_idx1, grad1)
        return weight_grad, None, None, None, None, None, None, None,None,None,None


class HashedEmbeddingBag(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        compression:float = 1. / 64.,
        mode:str = "sum",
        _weight: Optional[torch.Tensor] = None,
        val_offset = None,
        seed = 1024,
        uma_chunk_size = 1,
        padding_idx = None,
        no_bag = False,
        sparse = False)->None:
        super(HashedEmbeddingBag, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        memory = int(num_embeddings * embedding_dim * compression + 1)
        #memory = int(np.exp2(int(np.log2(memory)))) #  make sure it is power of 2
        self.weight_size = memory
        self.no_bag = no_bag
        self.sparse = sparse
        self.val_offset = val_offset
        self.mode = mode
        self.norm = None
        self.uma_chunk_size = uma_chunk_size
        self.padding_idx = padding_idx
        r = np.random.RandomState(seed)
        random_numbers = np.concatenate([np.array([2038074743]), r.randint(0, 2038074743, (50,))]) # set of 50 random numbers to use
        self.random_numbers = Parameter(torch.from_numpy(random_numbers.astype(np.int64)), requires_grad=False)
        print("RandomNumbers: ", self.random_numbers[:5])

        if _weight is None :
            low = -math.sqrt(1 / self.num_embeddings)
            high = math.sqrt(1 / self.num_embeddings)
            self.hashed_weight = Parameter(torch.rand(self.weight_size) * (high - low) + low)
            self.central = False
            #self.reset_parameters()
            print("Inside HashedEmbeddingBag (after reset): ", num_embeddings, embedding_dim, compression, self.weight_size, self.hashed_weight.shape)
        else:
            #assert len(_weight.shape) == 1 and _weight.shape[0] == weight_size, \
            #    'Shape of weight does not match num_embeddings and embedding_dim'
            print("Central weight", "val_offset", self.val_offset)
            self.hashed_weight = _weight
            self.weight_size = self.hashed_weight.numel()
            self.central = True
            assert(self.val_offset is not None)

        self.weight = self.hashed_weight
        print("HashedEmbeddingBag: ", num_embeddings, embedding_dim, "mode", mode,
              "central", self.central, 
              "weight_size", self.weight_size,
              "seed", seed,
              "uma_chunk_size", self.uma_chunk_size,
              "no_bag[backward]", self.no_bag,
              "sparse[backward]", self.sparse)

    def forward(self, indices: torch.Tensor, offsets: Optional[torch.Tensor] = None, per_sample_weights=None) -> torch.Tensor:
        i_shape = indices.shape
        indices = indices.view(-1)
        if self.padding_idx is not None:
            original_count = indices.shape[0]
            indx_mask = (indices != self.padding_idx)
            indx_padd_mask = (indices == self.padding_idx)
            indices = indices[indx_mask]

        if offsets is None:
            offsets  = torch.arange(len(indices), device = indices.device)
        assert(per_sample_weights is None)
        embeddings =  HashedEmbeddingBagFunction.apply(
            self.hashed_weight,
            indices,
            offsets,
            self.mode,
            self.embedding_dim,
            self.random_numbers,
            self.val_offset,
            self.norm,
            self.uma_chunk_size,
            self.no_bag,
            self.sparse
        )
        if self.padding_idx is not None:
            Aembeddings = torch.zeros(original_count, self.embedding_dim, device=indices.device)
            Aembeddings[indx_mask,:] = embeddings[:,:]
            embeddings = Aembeddings
        if len(i_shape) > 1:
            embeddings = embeddings.view(*i_shape, embeddings.shape[-1])
        return embeddings
