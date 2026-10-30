"""Exclusive wall-clock accounting; nested phases never count the same time twice."""
from collections import Counter
from contextlib import contextmanager
from functools import wraps
import time


class Ledger:
    def __init__(self):
        self.seconds = Counter()
        self.calls = Counter()
        self.stack = []

    @contextmanager
    def measure(self, name):
        now = time.perf_counter()
        if self.stack:
            parent = self.stack[-1]
            self.seconds[parent[0]] += now-parent[1]
        entry = [name, now]
        self.stack.append(entry)
        self.calls[name] += 1
        try:
            yield
        finally:
            now = time.perf_counter()
            self.seconds[name] += now-entry[1]
            assert self.stack.pop() is entry
            if self.stack:
                self.stack[-1][1] = now

    def summary(self):
        assert not self.stack, 'Only summarize after the timed operation returns'
        return dict(seconds=dict(self.seconds), calls=dict(self.calls),
                    accounted_seconds=sum(self.seconds.values()),
                    exclusive=True, uninstrumented_time_included=False)


def metered(name):
    def decorate(fn):
        @wraps(fn)
        def call(self, *args, **kwargs):
            with self.ledger.measure(name):
                return fn(self, *args, **kwargs)
        return call
    return decorate
