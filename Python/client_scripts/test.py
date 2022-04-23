from codebase.integration.client_interface import ClientInterfaceBase


class ClientInterface(ClientInterfaceBase):
    def __init__(self):
        super().__init__()


# noinspection PyUnresolvedReferences
instance = ClientInterface()
