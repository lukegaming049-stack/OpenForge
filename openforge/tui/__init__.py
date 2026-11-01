"""Terminal UI package. Importing it pulls in textual."""

__all__ = ["ForgeApp"]


def __getattr__(name: str):
    if name == "ForgeApp":
        from .app import ForgeApp

        return ForgeApp
    raise AttributeError(name)
