class AttrDict(dict):
    def __getattr__(self, name):
        return self[name]

    def get_value(self, key, default=None):
        return self.get(key, default)
