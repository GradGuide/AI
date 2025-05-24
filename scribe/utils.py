from nltk import sent_tokenize, word_tokenize


def smart_split(text, n):
    original_lines = text.split("\n")
    chunks = []
    paragraph_ids = []
    current_chunk = []
    current_length = 0

    for para_id, line in enumerate(original_lines):
        stripped_line = line.strip()
        if stripped_line:
            sentences = sent_tokenize(stripped_line)
            for sentence in sentences:
                sentence_length = len(word_tokenize(sentence))
                if current_length + sentence_length <= n:
                    current_chunk.append(sentence)
                    current_length += sentence_length
                else:
                    if current_chunk:
                        chunks.append(" ".join(current_chunk))
                        paragraph_ids.append(para_id)
                    current_chunk = [sentence]
                    current_length = sentence_length
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                paragraph_ids.append(para_id)
                current_chunk = []
                current_length = 0
        else:
            # Handle empty lines as a special chunk
            chunks.append("")
            paragraph_ids.append(para_id)

    return chunks, paragraph_ids


def smart_join(chunks, paragraph_ids):
    if len(chunks) != len(paragraph_ids):
        raise ValueError("Mismatch between chunks and paragraph IDs length")

    # Group chunks by their paragraph ID
    para_dict = {}
    for chunk, para_id in zip(chunks, paragraph_ids):
        if para_id not in para_dict:
            para_dict[para_id] = []
        para_dict[para_id].append(chunk)

    # Reconstruct lines in original order
    max_para_id = max(para_dict.keys()) if para_dict else 0
    reconstructed_lines = []
    for para_id in range(max_para_id + 1):
        if para_id in para_dict:
            line_chunks = para_dict[para_id]
            # Join chunks for this line, handling empty lines
            if not any(line_chunks):  # All chunks are empty
                reconstructed_lines.append("")
            else:
                reconstructed_lines.append(" ".join(line_chunks))
        else:
            # Handle missing para_id (unlikely if split correctly)
            reconstructed_lines.append("")

    return "\n".join(reconstructed_lines)
