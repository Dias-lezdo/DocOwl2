import torch
import os
from transformers import AutoTokenizer, AutoModel
from icecream import ic
import time

class DocOwlInfer():
    def __init__(self, ckpt_path):
        self.tokenizer = AutoTokenizer.from_pretrained(ckpt_path, use_fast=False)
        self.model = AutoModel.from_pretrained(ckpt_path, trust_remote_code=True, low_cpu_mem_usage=True, torch_dtype=torch.float16, device_map='auto')
        self.model.init_processor(tokenizer=self.tokenizer, basic_image_size=504, crop_anchors='grid_12')
        
    def inference(self, images, query):
        messages = [{'role': 'USER', 'content': '<|image|>'*len(images)+query}]
        answer = self.model.chat(messages=messages, images=images, tokenizer=self.tokenizer)
        return answer


docowl = DocOwlInfer(ckpt_path='mPLUG/DocOwl2')

images = [
        '/home/Screenshot (4).png',
        '/home/Screenshot (5).png',
    ]

answer = docowl.inference(images, query='what is this paper about? provide detailed information.')

answer = docowl.inference(images, query='what is the third page about? provide detailed information.')


# sudo apt update
# pip install torch transformers icecream
# pip install --upgrade transformers
# pip install sentencepiece
# pip install protobuf==4.23.3
# pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
# pip install einops
# pip install flash_attn
# pip install 'accelerate>=0.26.0'
# python doc.py



# As shown in the image 1, This paper is about a SQL Workbench and its features for designing, running, and testing SQL queries.
# It provides information on how to access and use the Workbench, as well as details on its various tools and connections.
# The paper also includes screenshots and step-by-step instructions on how to use the Workbench to connect to a database, create and run SQL queries, and view and modify query results. Overall, the paper provides a comprehensive guide for working with SQL and the Workbench.

# As shown in the image 2, I'm sorry, but there is no information provided about the third page.
# The texts in the image only mention the first and second pages, with the second page containing a list of resources and tools for working with MariaDB.
# There is no indication that there is a third page or what it might be about.
