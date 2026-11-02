"""Notebook/terminal display boundary. No typewriter sleeps in the workflow."""
import time
from IPython import get_ipython
from IPython.display import HTML, Markdown, Image, clear_output as _clear_output, display as _display

def in_notebook():
    shell=get_ipython()
    return shell is not None and getattr(shell, "kernel", None) is not None

def display(*objects, **kwargs):
    if in_notebook():
        return _display(*objects, **kwargs)
    for obj in objects:
        print(getattr(obj, "data", str(obj)))

def clear_output(*args, **kwargs):
    if in_notebook():
        _clear_output(*args, **kwargs)

class StreamDisplay:
    @staticmethod
    def write(text, speed=0.01, is_markdown=True, color=None):
        # Keep the notebook signature; formatting is presentation, not computation.
        display(Markdown(text) if is_markdown else HTML(text))

class BlockTimer:
    def __init__(self, name):
        self.name=name
    def __enter__(self):
        self.start_time=time.perf_counter()
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed=time.perf_counter()-self.start_time
        StreamDisplay.write(f"{self.name}: {'failed' if exc_type else 'completed'} in {elapsed:.2f}s")
