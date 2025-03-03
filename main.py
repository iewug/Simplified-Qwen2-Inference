import json
from configuration_qwen2 import Qwen2Config
from modeling_qwen2 import Qwen2ForCausalLM
from safetensors.torch import load_model
import torch
from tokenizer import Qwen2Tokenizer
import readline # support arrow keys and backspace keys for input()

# loading modelConfig
modelConfig_file = 'config.json'
with open(modelConfig_file, encoding="utf-8") as f:
    modelConfig = json.load(f)

configuration = Qwen2Config(**modelConfig)

with torch.device("cuda:0"):
    model = Qwen2ForCausalLM(configuration)
    model.eval()


# loading weight
print('Loading weight...')
weight_files = [
    "deepseek-r1-distill-qwen-14B/model-00001-of-000004.safetensors",
    "deepseek-r1-distill-qwen-14B/model-00002-of-000004.safetensors",
    "deepseek-r1-distill-qwen-14B/model-00003-of-000004.safetensors",
    "deepseek-r1-distill-qwen-14B/model-00004-of-000004.safetensors"
]
for weight_file in weight_files:
    load_model(model,weight_file,strict=False)
print('Done')


tokenizer = Qwen2Tokenizer(vocab_file='Qwen-tokenizer/vocab.json',merges_file='Qwen-tokenizer/merges.txt')
maxIterNum = 4096
###################
# 1. conversation #
###################
messages = []
while True:
    # encode
    userInput = input(">> ")
    messages.append({"role": "user", "content": userInput})
    input_ids = tokenizer.encode_template(messages)
    # generate
    token_list = []
    with torch.no_grad():
        x = torch.tensor([input_ids]).to("cuda:0")
        past_key_values = None
        for i in range(maxIterNum):
            logits,past_key_values = model(x,past_key_values=past_key_values)
            token_id = logits.argmax(dim=-1).item()
            tokenizer.decode_stream([token_id])
            if token_id == 151643: # <｜end▁of▁sentence｜>
                print()
                break
            token_list.append(token_id)
            x = torch.tensor([[token_id]]).to("cuda:0")
    output = tokenizer.decode(token_list)
    messages.append({"role": "assistant", "content":output})


##################
# 2. Batch Input #
##################
# text_list = ['太阳升起的方向是','The direction the sun rises is']
# bpe_tokens_list, attention_mask = tokenizer.encode_batch(text_list)
# with torch.no_grad():
#     x = torch.tensor(bpe_tokens_list).to("cuda:0")
#     attention_mask = torch.tensor(attention_mask).to("cuda:0")
#     logits,past_key_values = model(x,attention_mask=attention_mask,logits_to_keep=x.size(1))
#     token_id_list = logits.argmax(dim=-1).cpu().tolist()
#     print(tokenizer.decode_batch(token_id_list))