import heapq as hp


class ObjectHeap:
    def __init__(self, initial=None, keys: tuple = (lambda x: x)):
        self.keys = keys
        self.index = 0
        if initial:
            self._data = [(*(key(item) for key in self.keys), i, item) for i, item in enumerate(initial)]
            self.index = len(self._data)
            hp.heapify(self._data)
        else:
            self._data = []

    def push(self, item):
        hp.heappush(self._data, (*(key(item) for key in self.keys), self.index, item))
        self.index += 1

    def pop(self):
        return hp.heappop(self._data)[1 + len(self.keys)]
