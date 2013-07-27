class Bunch:
    def __init__(self, **kwds):
        self.__dict__.update(kwds)

    def __getstate__(self):
        return self

    def __setstate__(self, state):
        self.update(state)
        self.__dict__ = self
