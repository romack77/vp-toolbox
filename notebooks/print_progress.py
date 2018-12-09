

def print_progress(iterable):
    """Prints a progress bar as something is iterated over.

    Yields:
        Each item in iterable.
    """
    for i in iterable:
        yield i
        print('.', end='')
    print('')
