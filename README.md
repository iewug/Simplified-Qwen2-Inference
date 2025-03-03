# Qwen2 Inference with PyTorch Only

## 1. About

For better understanding how LLM works, I separated and simplified the Qwen2 model from the original `transformers` package. Therefore, Pytorch and safetensors are the only packages required. Also, detailed code comments were made (although many in Chinese). Approximately 30GB of GPU memory is required.

## 2. Preparation

- Download model weights from [HuggingFace/DeepSeek](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B/tree/main). Place four safetensors file under `deepseek-r1-distill-qwen-14B` folder.

- Download `merge.txt` and `vocab.json` from [HuggingFace/Qwen](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-1M/tree/main). Place them under `Qwen-tokenizer` folder.

## 3. How to Run

```python
python main.py
```
![](./example.gif)