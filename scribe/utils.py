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


def smart_rejoin_limited_full(chunks, paragraph_ids, limit):
    """
    Args:
        text (str): Original full text (used to preserve exact formatting).
        chunks (List[str]): List of sentence chunks (from smart_split).
        paragraph_ids (List[int]): Corresponding paragraph index per chunk.
        limit (int): Max number of tokens per output block.

    Returns:
        List[str]: Token-limited rejoined text blocks, preserving exact original spacing.
    """
    if len(chunks) != len(paragraph_ids):
        raise ValueError("Mismatch between chunks and paragraph IDs length")

    # Step 1: Recover original lines with their exact whitespace
    text = smart_join(chunks, paragraph_ids)
    original_lines = text.splitlines(
        keepends=True
    )  # includes \n or \r\n at end of each line

    # Step 2: Rebuild original structure using exact lines
    para_dict = {}
    for chunk, para_id in zip(chunks, paragraph_ids):
        para_dict.setdefault(para_id, []).append(chunk)

    max_para_id = max(para_dict.keys()) if para_dict else 0
    reconstructed_lines = []
    for para_id in range(max_para_id + 1):
        original_line = (
            original_lines[para_id] if para_id < len(original_lines) else "\n"
        )
        if para_id in para_dict:
            line_chunks = para_dict[para_id]
            if not any(line_chunks):
                reconstructed_lines.append(
                    original_line
                )  # preserve blank line with spacing
            else:
                # Replace stripped text with joined chunk but keep trailing space/newline
                stripped = original_line.strip()
                suffix = original_line[len(stripped) :] if stripped else original_line
                reconstructed_lines.append(" ".join(line_chunks) + suffix)
        else:
            # Empty or missing para
            reconstructed_lines.append(original_line)

    # Step 3: Group into token-limited blocks while preserving line formatting
    result = []
    current_block = []
    current_token_count = 0

    for line in reconstructed_lines:
        line_token_count = len(word_tokenize(line))

        if line.strip() == "":
            # Pure whitespace line — push it out on its own
            if current_block:
                result.append("".join(current_block))
                current_block = []
                current_token_count = 0
            result.append(line)
            continue

        if line_token_count > limit:
            if current_block:
                result.append("".join(current_block))
                current_block = []
                current_token_count = 0
            result.append(line)  # too big to merge — push standalone
            continue

        if current_token_count + line_token_count <= limit:
            current_block.append(line)
            current_token_count += line_token_count
        else:
            # Flush and start new block
            result.append("".join(current_block))
            current_block = [line]
            current_token_count = line_token_count

    if current_block:
        result.append("".join(current_block))

    return result
