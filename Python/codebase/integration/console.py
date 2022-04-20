import __main__
from pyodide.console import Console


class ClientInterfaceBase:
    def create_console(self):
        return Console(self.__dict__)
