import os
import sys

class Logger(object):
    def __init__(self, fpath=None, title=None):
        self.console = sys.stdout
        self.file = None
        self.title = '' if title is None else title
        if fpath is not None:
            os.makedirs(os.path.dirname(fpath), exist_ok=True)
            self.file = open(fpath, 'w')

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()

    def set_names(self, names):
        self.names = names
        self.numbers = {}
        for name in names:
            self.numbers[name] = []

    def append(self, numbers):
        assert len(self.names) == len(numbers)
        for i, name in enumerate(self.names):
            self.numbers[name].append(numbers[i])

    def log(self):
        for name in self.names:
            print(f"{name}: {self.numbers[name]}")
