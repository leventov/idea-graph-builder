from typing import List
from dataclasses import dataclass
import re

from trafilatura.xml import xmltotxt
import lxml

from cache import cache_in_file


@dataclass
class Chunk:
  chunk: str
  doc_id: str
  index: int


def chunk_by_size(text: str, N: int, delimiter: str) -> List[str]:
  '''
  Chunks the given text in parts of at most N symbols while splitting
  the text only at the given delimiter (such as newline character or dot character).
  
  Adjacent chunks overlap by one part.
  
  Args:
    text (str): Text to chunk
    N (int): maximum chunk length
  '''
  # Split the text by consecutive occurrences of the delimiter, capturing the delimiters
  parts = re.split(f'({delimiter}+)', text)
  chunks = []
  current_chunk = parts[0]
  for i in range(1, len(parts), 2):
    delimiter_part = parts[i]
    text_part = parts[i + 1]
    if len(current_chunk + delimiter_part + text_part) > N:
      chunks.append(current_chunk)
      # Attempt to overlap by one part
      current_chunk = parts[i - 1] + delimiter_part + text_part
      # Check if the overlap causes an overflow
      if len(current_chunk) > N:
        # Fall back to no overlap
        current_chunk = text_part
    else:
      current_chunk += delimiter_part + text_part
  if current_chunk:
    chunks.append(current_chunk)
  return chunks


def chunk_text(text: str, N: str) -> List[str]:
  '''
  Chunks the given text in parts of at most N symbols while trying to split
  the text only at the paragraph boundaries (i.e., newline symbols, assuming
  that the given text is plaintext or Markdown), and then if some paragraphs
  are still longer than N symbols, trying to split that paragraph at periods
  (i.e., sentence boundaries).

  Adjacent chunks overlap by one paragraph or sentence.

  Args:
    text (str): Text to chunk
    N (int): maximum chunk length
  '''
  if len(text) <= N:
    return [text]
  # Split by newline if text is too long
  chunks = chunk_by_size(text, N, '\n')
  final_chunks = []
  for i, chunk in enumerate(chunks[:-1]):
    # Split by period if chunk is still too long
    if len(chunk) > N:
      period_chunks = chunk_by_size(chunk, N, '.')
      final_chunks += period_chunks[:-1]
      # Overlap by one period sub-chunk
      if len(period_chunks[-1]) + 1 + len(chunks[i + 1]) < N:
        chunks[i + 1] = period_chunks[-1] + '\n' + chunks[i + 1]
    else:
      final_chunks.append(chunk)

  last_chunk = chunks[-1]
  if len(last_chunk) > N:
    final_chunks = chunk_by_size(last_chunk, N, '.')
  else:
    final_chunks.append(last_chunk)

  return final_chunks


@cache_in_file
def chunk_element_tree(doc_id: str, element: lxml.etree._Element,
                       N: int) -> List[Chunk]:
  '''
  Chunks the text contents of the given XML-TEI element so that chunks are as close
  to N characters as possible, while trying to respect the semantics of the XML-TEI
  tree.

  If the element (or one of the children) could be converted to plaintext which
  is shorter than N, that text is returned (within a 1-sized list).

  Otherwise, the children of the element are grouped in chunks, plaintext
  representations of children are joined with newlines, such that the size of
  the chunk is less than N. Adjacent groups overlap by one child element if possible.

  If the size of the innermost element (element without children) is still larger
  than N, resort to chunk_text().

  TODO: Don't overlap elements when a subsequent element is a header.

  Args:
    N (int): maximum chunk length
  '''
  text = xmltotxt(element, include_formatting=False)
  if len(text) <= N:
    return [Chunk(text, doc_id, 0)]

  children_texts_and_elements = []
  if element.text:
    children_texts_and_elements.append((element.text, None))
  for child in element.iterchildren():
    children_texts_and_elements.append(
      (xmltotxt(child, include_formatting=False), child))
    if child.tail:
      children_texts_and_elements.append((child.tail, None))

  chunks = []
  current_chunk = ""
  for idx, (child_text,
            child_element) in enumerate(children_texts_and_elements):
    if len(current_chunk + '\n' +
           child_text if current_chunk else child_text) > N:
      if current_chunk:
        chunks.append(Chunk(current_chunk, doc_id, len(chunks)))
      if len(child_text) > N:
        if child_element is not None:
          # Disregard the wrong indexes assigned to chunks in the recursive call
          sub_chunks = list(
            map(lambda c: c.chunk, chunk_element_tree(doc_id, child_element,
                                                      N)))
        else:
          sub_chunks = chunk_text(child_text, N)

        for sub_chunk in sub_chunks[:-1]:
          chunks.append(Chunk(sub_chunk, doc_id, len(chunks)))
        current_chunk = sub_chunks[-1]
      else:  # child_text is shorter than N
        current_chunk = child_text
        # Try to overlap by one previous child
        if idx > 0:
          prev_text, _ = children_texts_and_elements[idx - 1]
          if len(prev_text) + 1 + len(child_text) < N:
            current_chunk = prev_text + '\n' + current_chunk
    else:
      current_chunk += '\n' + child_text if current_chunk else child_text

  if current_chunk:
    chunks.append(Chunk(current_chunk, doc_id, len(chunks)))

  return chunks
