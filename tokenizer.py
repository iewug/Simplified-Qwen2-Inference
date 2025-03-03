'''
tokenizer class
diff from transformers api
'''
import json
import regex as re
from typing import Union,List,Tuple

def bytes_to_unicode():
    """
    把字节(0~255)映射到一个unicode上, 人为规定, 避免了空格和控制符等一些不可见字符
    e.g. 介的utf8是0xE4BB8B, 对应的unicode是ä»ĭ
         空格的utf8是0x20, 对应的unicode是Ġ
    """
    bs = (
        list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))

class Qwen2Tokenizer:
    def __init__(self, vocab_file:str, merges_file:str):
        # load vocabulary file
        with open(vocab_file, encoding="utf-8") as vocab_handle:
            self.encoder = json.load(vocab_handle) # {"!":0,...}
        # add special tokens
        self.encoder["<｜end▁of▁sentence｜>"]=151643
        self.encoder["<｜User｜>"]=151644
        self.encoder["<｜Assistant｜>"]=151645
        self.encoder["<｜begin▁of▁sentence｜>"]=151646
        self.encoder["<think>"]=151648
        self.encoder["</think>"]=151649
        self.decoder = {v: k for k, v in self.encoder.items()} # {0:"!",...}
        
        # load merge file
        # 加载merges策略 {('Ġ','Ġ'):0, ...} 对应非负整数小的优先级高 这里两个空格的合并是最高优先级
        bpe_merges = []
        with open(merges_file, encoding="utf-8") as merges_handle:
            for i, line in enumerate(merges_handle):
                line = line.strip()
                if (i == 0 and line.startswith("#version:")) or not line:
                    continue
                bpe_merges.append(tuple(line.split()))
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))

        # 单字节与unicode对应规则
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}

        # 正则表达式来做pretokenization 差不多就是按照空格切分单词
        # [token for token in re.findall(pat, "介绍一下你自己。Introduce yourself")] = ['介绍一下你自己', '。', 'Introduce', ' yourself']
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

        # for stream output
        self.stream_buffer = []

    @staticmethod
    def get_pairs(word):
        """
        get_pairs('abcde') -> {('a', 'b'), ('d', 'e'), ('c', 'd'), ('b', 'c')}
        """
        pairs = set()
        prev_char = word[0]
        for char in word[1:]:
            pairs.add((prev_char, char))
            prev_char = char
        return pairs
    
    # 分词 譬如 'ä»ĭç»įä¸Ģä¸ĭä½łèĩªå·±' -> 'ä»ĭç»įä¸Ģä¸ĭ ä½łèĩªå·±'
    def _bpe(self, token):
        """
        假设是'abcde', 首先分成{('a', 'b'), ('d', 'e'), ('c', 'd'), ('b', 'c')},
        然后在bperank里面找优先级, 优先级高的先合并, 再分成{('ab', 'c'), ('d', 'e'), ('c', 'd')},
        如此往复, 一直到无法合并为止, 譬如{('abc'), ('de')}
        """
        word = tuple(token)
        pairs = self.get_pairs(word) # 相邻两项组成pair

        if not pairs:
            return token

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                except ValueError:
                    new_word.extend(word[i:])
                    break
                else:
                    new_word.extend(word[i:j])
                    i = j

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = self.get_pairs(word)
        word = " ".join(word)
        return word

    def encode(self, text:str, retID=True)->List[Union[str,int]]:
        '''
        Args:
            text: String to encode.
            retID: If True return ids else return bpe_tokens.
        
        Returns:
            list: If `retID` is True, returns a list of token IDs (List[int]).
                  If `retID` is False, returns a list of BPE tokens (List[str]).
        '''
        bpe_tokens = []
        for token in re.findall(self.pat, text):
            # token in ['介绍一下你自己', '。', 'Introduce', ' yourself']
            token = "".join(
                self.byte_encoder[b] for b in token.encode("utf-8")
            ) # 依次为 ä»ĭç»įä¸Ģä¸ĭä½łèĩªå·±   ãĢĤ   Introduce   Ġyourself
            bpe_tokens.extend(bpe_token for bpe_token in self._bpe(token).split(" "))
        if not retID:
            return bpe_tokens # ['ä»ĭç»įä¸Ģä¸ĭ', 'ä½łèĩªå·±', 'ãĢĤ', 'Int', 'roduce', 'Ġyourself']
        else:
            return [self.encoder.get(bpe_token) for bpe_token in bpe_tokens] # [109432, 107828, 1773, 1072, 47845, 6133]

    def encode_batch(self, text_list:List[str], retID=True)->Tuple[List[List[Union[str,int]]],List[List[int]]]:
        '''
        Args:
            text_list (List[str]): List of string to encode.
            retID (bool): If True return token IDs else return BPE tokens.
        
        Returns:
            tuple: A tuple containing two elements:
                - If `retID` is True, returns a list of token IDs list (List[List[int]]).
                  If `retID` is False, returns a list of BPE tokens list (List[List[str]]).
                - A list of attention masks (List[List[int]]), where 0 indicates padding and 1 indicates non-padding.
        '''
        bpe_tokens_list = []
        for text in text_list:
            bpe_tokens = []
            for token in re.findall(self.pat,text):
                token = "".join(
                    self.byte_encoder[b] for b in token.encode("utf-8")
                )
                bpe_tokens.extend(bpe_token for bpe_token in self._bpe(token).split(" "))
            bpe_tokens_list.append(bpe_tokens)
        # add left padding
        token_len_list = [len(bpe_tokens) for bpe_tokens in bpe_tokens_list]
        max_token_len = max(token_len_list)
        attention_mask = [[0]*(max_token_len-token_len)+[1]*token_len for token_len in token_len_list]
        for i in range(len(token_len_list)):
            bpe_tokens_list[i] = ["<｜end▁of▁sentence｜>"] * (max_token_len-token_len_list[i]) + bpe_tokens_list[i]

        if not retID:
            return bpe_tokens_list, attention_mask
        else:
            return [[self.encoder.get(bpe_token) for bpe_token in bpe_tokens] for bpe_tokens in bpe_tokens_list], attention_mask

    def encode_template(self, messages:List[dict], tokenize=True, add_generation_prompt=True)->Union[str,List[int]]:
        r'''
        Args:
            messages (List[dict]): Each dict's keys = ["role", "content"], "role" can only be one of "system", "user" and "assistant".
            tokenize (bool): If True return token IDs else return plain text.
            add_generation_prompt (bool): Add `<｜Assistant｜><think>\n` at the end.
        
        Returns:
            Union[str,List[int]]: If `tokenize` is True, returns token IDs (List[int]).
                                  If `tokenize` is False, returns plain text (str).
        '''
        if not tokenize:
            text = "<｜begin▁of▁sentence｜>"
            for message in messages:
                if message['role'] == 'system':
                    text += message['content']
                elif message['role'] == 'user':
                    text += '<｜User｜>' + message['content']
                elif message['role'] == 'assistant':
                    text += '<｜Assistant｜>' + message['content'] + '<｜end▁of▁sentence｜>'
            if add_generation_prompt:
                text += '<｜Assistant｜><think>\n'
            return text
        else:
            ids = [151646]
            for message in messages:
                if message['role'] == 'system':
                    ids += self.encode(message['content'])
                elif message['role'] == 'user':
                    ids += [151644] + self.encode(message['content'])
                elif message['role'] == 'assistant':
                    ids += [151645] + self.encode(message['content']) + [151643]
            if add_generation_prompt:
                ids += [151645,151648,198]
            return ids
        
    # def decode(self, bpe_token_ids:List[int])->str: # 不支持special token
    #     bpe_tokens = [self.decoder.get(bpe_token_id) for bpe_token_id in bpe_token_ids]
    #     text = "".join(bpe_tokens)
    #     text = bytearray([self.byte_decoder[c] for c in text]).decode("utf-8")
    #     return text
    
    def decode(self, bpe_token_ids: List[int]) -> str: # 会打印special token
        text_parts = []  # 存储最终文本的各个部分
        byte_buffer = []  # 临时存储字节数据

        for bpe_token_id in bpe_token_ids:
            token_str = self.decoder.get(bpe_token_id)  # 获取token对应的字符串
            if bpe_token_id >= 151643: # 如果是特殊token（ID >=151643）
                if byte_buffer: # 先将之前累积的字节解码为文本
                    byte_buffer = "".join(byte_buffer)
                    decoded = bytearray([self.byte_decoder[c] for c in byte_buffer]).decode("utf-8")
                    text_parts.append(decoded)
                    byte_buffer = []  # 清空缓冲区
                text_parts.append(token_str)
            else: # 普通token
                byte_buffer.append(token_str)
        
        if byte_buffer: # 处理最后剩余的字节
            byte_buffer = "".join(byte_buffer)
            decoded = bytearray([self.byte_decoder[c] for c in byte_buffer]).decode("utf-8")
            text_parts.append(decoded)
        return "".join(text_parts)
    

    def decode_batch(self, bpe_token_ids_list:List[List[int]])->List[str]:
        return [self.decode(bpe_token_ids) for bpe_token_ids in bpe_token_ids_list]


    def decode_stream(self, bpe_token_ids: List[int]): # 我自己都觉得自己写的有点抽象 但是又挺对的
        try:
            if self.stream_buffer is None:
                text = self.decode(bpe_token_ids)
                print(text,end='',flush=True)
            else:
                text = self.decode(self.stream_buffer+bpe_token_ids)
                print(text,end='',flush=True)
                self.stream_buffer = []
        except UnicodeDecodeError as e:
            self.stream_buffer.extend(bpe_token_ids)
        



if __name__ == "__main__":
    token = Qwen2Tokenizer(vocab_file='Qwen-tokenizer/vocab.json',merges_file='Qwen-tokenizer/merges.txt')
    # print(token.encode("介绍一下你自己。Introduce yourself.",retID=False)) # ['ä»ĭç»įä¸Ģä¸ĭ', 'ä½łèĩªå·±', 'ãĢĤ', 'Int', 'roduce', 'Ġyourself', '.']
    # print(token.encode("介绍一下你自己。Introduce yourself.",retID=True)) # [109432, 107828, 1773, 1072, 47845, 6133, 13]
    # print(token.encode_batch(["介绍一下你自己","Introduce yourself"],retID=False)) # ([['<｜end▁of▁sentence｜>', 'ä»ĭç»įä¸Ģä¸ĭ', 'ä½łèĩªå·±'], ['Int', 'roduce', 'Ġyourself']], [[0, 1, 1], [1, 1, 1]])
    # print(token.encode_batch(["介绍一下你自己","Introduce yourself"],retID=True)) # ([[151643, 109432, 107828], [1072, 47845, 6133]], [[0, 1, 1], [1, 1, 1]])
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello"}, # tensor([[151646, 151644, 9707, 151645, 151648,    198]])
        {"role": "assistant", "content": "你好，有什么可以帮助你的吗？"},
        {"role": "user", "content": "某小学在“献爱心–为汶川地震区捐款”活动中，六年级五个班共捐款8000元，其中一班捐款1500元，二班比一班多捐款200元，三班捐款1600元，四班与五班捐款数之比是3:5。四班捐款多少元?"}    
    ]
    print(token.encode_template(messages,tokenize=False))
    ids = token.encode_template(messages,tokenize=True)
    print(token.decode(ids))