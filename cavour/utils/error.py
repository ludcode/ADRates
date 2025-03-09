class LibError(Exception):
    """ Class to understand if the error is coming from this library """

    def __init__(self,
                 message: str):
        """ Create error object """
        self._message = message

    def _print(self):
        print("LibError:",self._message)
