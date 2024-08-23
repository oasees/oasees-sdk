# ipfs_endpoint = "http://10.160.1.209:5001/api/v0"
# indexer_endpoint ="http://10.160.1.209:31007"

class ENVs():
    def __init__(self):
        self.ipfs_endpoint= ""
        self.indexer_endpoint= ""

    def set_envs(self,ipfs_endpoint,indexer_endpoint):
        self.ipfs_endpoint = ipfs_endpoint
        self.indexer_endpoint = indexer_endpoint

Envs = ENVs()