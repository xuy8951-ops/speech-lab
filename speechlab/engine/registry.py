"""注册表：YAML 里写名字，引擎里拿类。新增算法不需要修改引擎。"""


class Registry:
    def __init__(self, kind: str):
        self.kind = kind
        self._items = {}

    def register(self, name: str):
        def deco(cls):
            self._items[name] = cls
            return cls
        return deco

    def get(self, name: str):
        if name not in self._items:
            raise KeyError(f"[{self.kind}] 未注册的名字: {name}；已有: {list(self._items)}")
        return self._items[name]


ENHANCERS = Registry("enhancer")
DATASETS = Registry("dataset")
METRICS = Registry("metric")
