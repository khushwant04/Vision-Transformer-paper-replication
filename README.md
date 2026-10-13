# Vision Transformer (ViT) Paper Replication

A step-by-step PyTorch build of the ViT-Base architecture from *An Image is Worth 16x16 Words*, written in a single notebook (`ViT.ipynb`).

## What the notebook covers

1. **Hyperparameters** in ViT-Base/16 style: patch size 16, latent size 768, 12 heads, 12 encoder blocks, dropout 0.1, 224x224 input.
2. **`InputEmbedding`**: splits the image into 16x16 patches with `einops.rearrange`, applies a linear projection to 768 dims, prepends a class token and adds a positional embedding.
3. **`Encoder`**: a pre-norm transformer block with `LayerNorm`, `nn.MultiheadAttention`, a residual connection, and an MLP (768 -> 3072 -> 768, GELU, dropout) with a second residual connection.
4. **`ViT`**: the embedding, a stack of 12 encoders, and an MLP head on the class token (`num_classes = 10`).
5. **Shape checks**: each component is run on a random `(8, 3, 224, 224)` batch to confirm the output shapes.

The notebook stops at the architecture. It does not load a dataset, train or evaluate. The training hyperparameters it defines (`epochs`, `base_lr`, `weight_decay`) are not used yet.

## Known differences from the paper

These show up when you read the code closely. They are useful to know before adding a training loop:

- `nn.MultiheadAttention` is created without `batch_first=True`, so the `(batch, tokens, dim)` input is read as `(seq, batch, dim)` and attention runs across the batch axis.
- The class token and positional embedding are shaped `(batch_size, 1, dim)`. The paper uses `(1, 1, dim)` and `(1, N+1, dim)`. As written here, every patch position gets the same positional vector and the model is tied to batch size 8.
- `nn.Parameter(...).to(device)` returns a plain tensor on GPU, so these two embeddings are not registered as trainable parameters when CUDA is used.
- One `LayerNorm` is shared by both sub-layers of each encoder block. The paper uses two.

## Getting started

```bash
git clone https://github.com/khushwant04/Vision-Transformer-paper-replication.git
cd Vision-Transformer-paper-replication
pip install -r requirements.txt jupyter
jupyter notebook ViT.ipynb
```

The notebook runs on CPU or CUDA. It runs on random input, so no dataset download is needed.

## Reference

Dosovitskiy et al., *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale*, ICLR 2021. [arXiv:2010.11929](https://arxiv.org/abs/2010.11929)

## License

Apache-2.0. See [LICENSE](LICENSE).
