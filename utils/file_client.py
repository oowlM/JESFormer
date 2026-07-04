class HardDiskBackend:
    def get(self, filepath):
        with open(str(filepath), 'rb') as handle:
            return handle.read()


class FileClient:
    def __init__(self, backend='disk', **kwargs):
        if backend != 'disk':
            raise ValueError('Only disk backend is supported in this repository.')
        self.backend = backend
        self.client = HardDiskBackend()

    def get(self, filepath, client_key='default'):
        del client_key
        return self.client.get(filepath)

