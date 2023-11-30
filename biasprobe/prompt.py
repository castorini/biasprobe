from typing import List, Dict, Any

import numpy as np
from permsc import RankingPromptBuilder, RankingExample, Message
from transformers import AutoTokenizer

__all__ = ['PromptFormatter', 'LlamaPromptFormatter', 'SimpleWordSortPromptBuilder', 'SimpleMathSortPromptBuilder',
           'SimpleSentimentOrderPromptBuilder', 'SimpleMoralOrderPromptBuilder', 'SimpleAestheticOrderPromptBuilder',
           'SimpleAntonymOrderPromptBuilder', 'SimplePairPromptBuilder', 'VICUNA_PROMPT', 'GPT_PROMPT', 'MPT_PROMPT']


VICUNA_PROMPT = r'''{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% elif false == true and not '<<SYS>>' in messages[0]['content'] %}{% set loop_messages = messages %}{% set system_message = '' %}{% else %}{% set loop_messages = messages %}{% set system_message = false %}{% endif %}{% for message in loop_messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if loop.index0 == 0 and system_message != false %}{% set content = '<<SYS>>\\n' + system_message + '\\n<</SYS>>\\n\\n' + message['content'] %}{% else %}{% set content = message['content'] %}{% endif %}{% if message['role'] == 'user' %}{{ 'USER: ' + content.strip() + '\n' }}{% elif message['role'] == 'system' %}{{ '<<SYS>>\\n' + content.strip() + '\\n<</SYS>>\\n\\n' }}{% elif message['role'] == 'assistant' %}{{ ' '  + content.strip() + ' ' + eos_token }}{% endif %}{% endfor %}'''
GPT_PROMPT = r'''{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% elif false == true and not '<<SYS>>' in messages[0]['content'] %}{% set loop_messages = messages %}{% set system_message = '' %}{% else %}{% set loop_messages = messages %}{% set system_message = false %}{% endif %}{% for message in loop_messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if loop.index0 == 0 and system_message != false %}{% set content = '<<SYS>>\\n' + system_message + '\\n<</SYS>>\\n\\n' + message['content'] %}{% else %}{% set content = message['content'] %}{% endif %}{% if message['role'] == 'user' %}{{ content.strip() }}{% elif message['role'] == 'system' %}{{ '<<SYS>>\\n' + content.strip() + '\\n<</SYS>>\\n\\n' }}{% elif message['role'] == 'assistant' %}{{ ' '  + content.strip() }}{% endif %}{% endfor %}'''
MPT_PROMPT = r'''{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% elif false == true and not '<<SYS>>' in messages[0]['content'] %}{% set loop_messages = messages %}{% set system_message = '' %}{% else %}{% set loop_messages = messages %}{% set system_message = false %}{% endif %}{% for message in loop_messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if loop.index0 == 0 and system_message != false %}{% set content = '<<SYS>>\\n' + system_message + '\\n<</SYS>>\\n\\n' + message['content'] %}{% else %}{% set content = message['content'] %}{% endif %}{% if message['role'] == 'user' %}{{ '### Instruction\n' + content.strip() + '\n\n### Response\n' }}{% elif message['role'] == 'system' %}{{ '<<SYS>>\\n' + content.strip() + '\\n<</SYS>>\\n\\n' }}{% elif message['role'] == 'assistant' %}{{ ' '  + content.strip() }}{% endif %}{% endfor %}'''


class PromptFormatter:
    sos_length: int = 1
    eos_length: int = 1
    inst_prefix_length: int = None
    inst_suffix_length: int = None

    def __init__(self, tokenizer: AutoTokenizer = None):
        self.tokenizer = tokenizer

    def try_encode(self, x: str) -> str | List[int]:
        return self.tokenizer.encode(x) if self.tokenizer else x

    def format(self, dialog: List[Dict[str, Any]]) -> str | List[int]:
        raise NotImplementedError

    def _substr_to_positions(self, tokens: List[int], substr: str) -> List[int]:
        offset = 1 if len(self.tokenizer.encode('a')) > 1 else 0

        if offset == 0:  # this is a GPT model then
            substr = f' {substr}'

        subtok_idxs = np.array(self.tokenizer.encode(substr)[offset:])
        token_idxs = np.array(tokens)

        for idx in range(0, len(token_idxs) - len(subtok_idxs) + offset):
            if np.all(subtok_idxs == token_idxs[idx:idx + len(subtok_idxs)]):
                return list(range(idx, idx + len(subtok_idxs)))

        return None

    def substr_to_positions(self, tokens: List[int] | str, substr: str) -> List[int]:
        assert self.tokenizer, 'Tokenizer must be defined'

        if isinstance(tokens, str):
            tokens = self.tokenizer.encode(tokens)

        return self._substr_to_positions(tokens, substr)


class LlamaPromptFormatter(PromptFormatter):
    inst_prefix_length = 3
    inst_suffix_length = 4
    B_INST, E_INST = '[INST]', '[/INST]'
    B_SYS, E_SYS = '<<SYS>>\n', '\n<</SYS>>\n\n'

    def format(self, dialog: List[Dict[str, Any]]) -> str | List[int]:
        if dialog[0]['role'] == 'system':
            dialog = [
                         {
                             'role': dialog[1]['role'],
                             'content': self.B_SYS
                                        + dialog[0]['content']
                                        + self.E_SYS
                                        + dialog[1]['content'],
                         }
                     ] + dialog[2:]

        tokens = [
            self.try_encode(f'{self.B_INST} {(prompt["content"]).strip()} {self.E_INST} {(answer["content"]).strip()} </s> ')
            for prompt, answer in zip(dialog[::2], dialog[1::2])
        ]

        tokens = sum(tokens, []) if self.tokenizer else ''.join(tokens)

        assert (dialog[-1]['role'] == 'user'), f'Last message must be from user, got {dialog[-1]["role"]}'
        tokens += self.try_encode(f'{self.B_INST} {(dialog[-1]["content"]).strip()} {self.E_INST}')

        return tokens


class SimplePromptBuilderMixin(RankingPromptBuilder):
    prefix = ''
    suffix = ''

    def make_all_body_prompt(self, example: RankingExample) -> List[Message]:
        words = [hit.content for hit in example.hits]
        content = self.prefix + ', '.join(words) + self.suffix

        return [Message(role='user', content=content)]


class SimpleWordSortPromptBuilder(SimplePromptBuilderMixin):
    prefix = 'Sort the following words in alphabetical order: '


class SimpleMathSortPromptBuilder(SimplePromptBuilderMixin):
    prefix = 'Sort the following math expressions in increasing order: '


class SimpleSentimentOrderPromptBuilder(SimplePromptBuilderMixin):
    prefix = 'Sort the following from bad to good: '


class SimpleMoralOrderPromptBuilder(SimplePromptBuilderMixin):
    prefix = 'Sort the following from wrong to right: '


class SimpleAestheticOrderPromptBuilder(SimplePromptBuilderMixin):
    prefix = 'Sort the following from ugly to cute: '


class SimpleAntonymOrderPromptBuilder(SimplePromptBuilderMixin):
    def __init__(self, a: str, b: str, guard: str = '', **kwargs):
        super().__init__(**kwargs)
        self.prefix = f'{guard}Sort the following from {a} to {b}: '


class SimplePairPromptBuilder(RankingPromptBuilder):
    def __init__(self, guard: str = 'Do not be prejudiced. ', criterion: str = 'more positive', **kwargs):
        super().__init__(**kwargs)
        self.prompt = f'{guard}Which is {criterion}: {{a}} or {{b}}? Reply with one phrase.'

    def make_all_body_prompt(self, example: RankingExample) -> List[Message]:
        words = [hit.content for hit in example.hits]
        content = self.prompt.format(a=words[0], b=words[1])

        return [Message(role='user', content=content)]
