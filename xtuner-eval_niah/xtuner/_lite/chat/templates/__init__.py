from .chat import ChatTemplate
from .hybrid import HybridChatTemplate

CHAT_TEMPLATE_MAP = {
    'internlm2':
    HybridChatTemplate(
        system='<|im_start|>system\n{system}<|im_end|>\n',
        user='<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n',
        assistant='{assistant}<|im_end|>',
        stop_words=['<|im_end|>']),
    'llama3_chat':
    HybridChatTemplate(
        system=('<|start_header_id|>system<|end_header_id|>\n\n{system}'
                '<|eot_id|>'),
        user=('<|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|>'
              '<|start_header_id|>assistant<|end_header_id|>\n\n'),
        assistant='{assistant}<|eot_id|>',
        sep='',
        stop_words=['<|eot_id|>']),
    'qwen':
    HybridChatTemplate(
        system='<|im_start|>system\n{system}<|im_end|>\n',
        user='<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n',
        assistant='{assistant}<|im_end|>',
        stop_words=['<|im_end|>', '<|endoftext|>']),
}

__all__ = ['ChatTemplate', 'HybridChatTemplate']
