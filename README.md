# Qwen2 Inference with PyTorch Only

## 1. About

For better understanding how LLM works, I separated and simplified the Qwen2 model from the original `transformers` package. Therefore, PyTorch and safetensors are the only packages required. Also, detailed code comments were made (although many in Chinese).

Some other details:

- Generative dialogue and batch input are supported.
- Attention Implementation: Eager, rather than sdpa or FlashAttention
- For simplicity, the code is run on a single gpu. Therefore, the GPU should have more than 30GB memory with `deepseek-r1-distill-qwen-14B` model. However, if you use smaller models, it can be solved.
- The tokenizer differs from the one from `transformers` package in terms of api and part of implementation.

## 2. Preparation

- Download model weights from [HuggingFace/DeepSeek](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B/tree/main). Place four safetensors file under `deepseek-r1-distill-qwen-14B` folder.

- Download `merge.txt` and `vocab.json` from [HuggingFace/Qwen](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-1M/tree/main). Place them under `Qwen-tokenizer` folder.

## 3. How to Run

```python
python main.py
```
![](./example.gif)