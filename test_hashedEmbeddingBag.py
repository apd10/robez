import torch
import hashedEmbeddingBag
import torch.nn as nn
import numpy as np
import time
import pdb

def test_fwd():
    n = 10000
    m = 128
    robez_size = 100
    idx_weight = torch.arange(100).cuda(0)
    weight = torch.rand(100).cuda(0)
    
    E = hashedEmbeddingBag.HashedEmbeddingBag(n, m, _weight=weight, val_offset=0, uma_chunk_size=32, no_bag=True, sparse=False).cuda(0)
    # getting complete idx matrix
    E.hashed_weight.data[:] = idx_weight[:]
    indices = torch.arange(n).cuda(0)

    idx = E(indices).long()
    E.hashed_weight.data[:] = weight[:]

    # embedding table
    embeddings = E(indices)

    
def test_bwd():
    n = 10000
    m = 128
    robez_size = 100
    weight = nn.Parameter(torch.rand(100).cuda(0))
    
    E = hashedEmbeddingBag.HashedEmbeddingBag(n, m, _weight=weight, val_offset=0, uma_chunk_size=32, no_bag=True, sparse=False).cuda(0)
    
    indices = torch.arange(10).cuda(0)
    embeddings = E(indices)
    loss = torch.norm(embeddings)
    loss.backward()
    print(loss)
    print(E.weight.grad)

def test1():
    n = 10000
    m = 128
    robez_size = 100
    weight = nn.Parameter(torch.rand(100).cuda(0))
    
    E = hashedEmbeddingBag.HashedEmbeddingBag(n, m, _weight=weight, val_offset=0, uma_chunk_size=32, no_bag=False, sparse=False).cuda(0)
    
    indices = torch.arange(10).cuda(0)
    embeddings = E(indices)
    loss = torch.norm(embeddings)
    loss.backward()
    print(loss)
    print(E.weight.grad)

def test2():
    n = 10000
    m = 128
    robez_size = 100
    weight = nn.Parameter(torch.rand(100).cuda(0))
    E = hashedEmbeddingBag.HashedEmbeddingBag(n, m, _weight=weight, val_offset=0, uma_chunk_size=32, no_bag=True, sparse=True).cuda(0)
    indices = torch.arange(10).cuda(0)
    embeddings = E(indices)
    loss = torch.norm(embeddings)
    loss.backward()
    print(loss)
    print(E.weight.grad)

def test3():

    n = 10000
    m = 128
    robez_size = 100
    E = hashedEmbeddingBag.HashedEmbeddingBag(n, m, _weight=None, compression=1.0 * robez_size / m / n , val_offset=0, uma_chunk_size=32, no_bag=True, sparse=True).cuda(0)
    indices = torch.arange(10).cuda(0)
    embeddings = E(indices)
    loss = torch.norm(embeddings)
    loss.backward()
    print(torch.max(E.weight), torch.min(E.weight))
    print(loss)
    print(E.weight.grad)
    pdb.set_trace()

def test_fwd_bag():
    n = 10000
    m = 128
    robez_size = 100
    idx_weight = torch.arange(100).cuda(0)
    
    weight = torch.rand(100).cuda(0)
    E = hashedEmbeddingBag.HashedEmbeddingBag(n, m, _weight=weight, val_offset=0, uma_chunk_size=32, no_bag=False, sparse=False).cuda(0)

    # getting complete idx matrix

    E.hashed_weight.data[:] = idx_weight[:]
    indices = torch.arange(n).cuda(0)
    idx = E(indices).long()
    weight = torch.rand(100).cuda(0)
    E.hashed_weight.data[:] = weight[:]

    # embedding table
    embeddings_all = E(indices)
    indices = torch.arange(10).cuda(0)
    offsets = torch.arange(2).cuda(0) * 5
    embeddings = E(indices, offsets)

    torch.testing.assert_close(embeddings[0], torch.sum(embeddings_all[0:5], dim=0))
