import openai
import os
import sys

from trafilatura import fetch_url, bare_extraction

from chunking import chunk_element_tree
from ideas import chunks_to_propositions, propositions_to_ideas, only_salient_ideas

try:
  openai.api_key = os.environ['OPENAI_API_KEY']
except KeyError:
  sys.stderr.write("""
  You haven't set up your API key yet.
  
  If you don't have an API key yet, visit:
  
  https://platform.openai.com/signup

  1. Make an account or sign in
  2. Click "View API Keys" from the top right menu.
  3. Click "Create new secret key"

  Then, open the Secrets Tool and add OPENAI_API_KEY as a secret.
  """)
  exit(1)

url = 'https://www.pinecone.io/learn/chunking-strategies/'
doc = fetch_url(url)
d = bare_extraction(doc, output_format='xmltei')
chunks = chunk_element_tree(doc_id=url,
                            element=d['body'],
                            N=2000,
                            cache_key="pinecone")
chunk_max_len = 7000
for i, chunk in enumerate(chunks):
  if len(chunk.chunk) > chunk_max_len:
    print(
      f'Chunk "{chunk.chunk}" is longer than {chunk_max_len} symbols, trimming'
    )
    chunks[i].chunk = chunks[i].chunk[:chunk_max_len]
chunks[0].chunk = d['title'] + '\n' + chunks[0].chunk

chunks_and_propositions = chunks_to_propositions(chunks, cache_key="pinecone")
ideas_per_chunk = propositions_to_ideas(chunks_and_propositions, cache_key="pinecone")
# Currently cutting ideas per chunk is not needed because there are usually fewer than
# 5 ideas per chunk after clustering and ordering them by saliency doesn't make
# a lot of sence. However, maybe we should cut the least salient ideas on the document
# level.
# all_ideas = only_salient_ideas(ideas_per_chunk, cache_key="pinecone")
all_ideas = [i for ideas in ideas_per_chunk for i in ideas]
for idea in all_ideas:
  print(idea)
