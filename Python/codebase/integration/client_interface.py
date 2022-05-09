# noinspection PyUnresolvedReferences
from pyodide.console import Console


class ClientInterfaceBase:
    def __init__(self):
        self.console = Console()

    @staticmethod
    def extract_bytes(s: str):
        return s.encode("latin-1")

    def echo(self):
        return self.message
