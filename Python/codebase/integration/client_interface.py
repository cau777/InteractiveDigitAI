# noinspection PyUnresolvedReferences
from pyodide.console import Console


class ClientInterfaceBase:
    def __init__(self):
        self.console = Console()
        self.params = dict()

    @staticmethod
    def extract_bytes(s: str):
        return s.encode("latin-1")

    def echo(self):
        return self.params["message"]
