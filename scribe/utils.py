from typing import Callable, List
from functools import wraps


def split_text(text: str, max_length: int) -> List[str]:
    """
    Splits the text into chunks based on a specified maximum length.

    Parameters:
    ----------
    text : str
        The input text to be split.
    max_length : int
        The maximum length for each chunk of text.

    Returns:
    -------
    List[str]
        A list of text chunks, each not exceeding max_length.
    """
    words = text.split()
    chunks = []
    current_chunk: List[str] = []

    for word in words:
        if len(" ".join(current_chunk + [word])) <= max_length:
            current_chunk.append(word)
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def process_in_batches(func: Callable, max_length: int = 512) -> Callable:
    """
    Decorator to handle batch processing of long text inputs for specific arguments.
    God help me.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        arg_names = func.__code__.co_varnames[: func.__code__.co_argcount]
        args = list(args)

        for arg_name in ["context", "input_text", "text"]:
            if arg_name in kwargs:
                if (
                    isinstance(kwargs[arg_name], str)
                    and len(kwargs[arg_name]) > max_length
                ):
                    chunks = [
                        kwargs[arg_name][i : i + max_length]
                        for i in range(0, len(kwargs[arg_name]), max_length)
                    ]
                    chunk_results = [
                        func(*args, **{**kwargs, arg_name: chunk}) for chunk in chunks
                    ]
                    kwargs[arg_name] = "".join(chunk_results)
            elif arg_name in arg_names:
                idx = arg_names.index(arg_name)
                if isinstance(args[idx], str) and len(args[idx]) > max_length:
                    chunks = [
                        args[idx][i : i + max_length]
                        for i in range(0, len(args[idx]), max_length)
                    ]
                    chunk_results = [
                        func(*args[:idx] + [chunk] + args[idx + 1 :], **kwargs)
                        for chunk in chunks
                    ]
                    args[idx] = "".join(chunk_results)

        for arg_name in ["sentences", "paragraphs"]:
            if arg_name in kwargs:
                if isinstance(kwargs[arg_name], list) and any(
                    len(s) > max_length for s in kwargs[arg_name]
                ):
                    chunks = [
                        item[i : i + max_length]
                        for item in kwargs[arg_name]
                        for i in range(0, len(item), max_length)
                    ]
                    chunk_results = [
                        func(*args, **{**kwargs, arg_name: chunk}) for chunk in chunks
                    ]
                    kwargs[arg_name] = [
                        item for sublist in chunk_results for item in sublist
                    ]
            elif arg_name in arg_names:
                idx = arg_names.index(arg_name)
                if isinstance(args[idx], list) and any(
                    len(s) > max_length for s in args[idx]
                ):
                    chunks = [
                        item[i : i + max_length]
                        for item in args[idx]
                        for i in range(0, len(item), max_length)
                    ]
                    chunk_results = [
                        func(*args[:idx] + [chunk] + args[idx + 1 :], **kwargs)
                        for chunk in chunks
                    ]
                    args[idx] = [item for sublist in chunk_results for item in sublist]

        return func(*args, **kwargs)

    return wrapper
