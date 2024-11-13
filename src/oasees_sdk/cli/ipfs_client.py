import requests

def ipfs_upload(_file,OASEES_IPFS_IP):

    _file = _file.encode('utf-8')
    resp = requests.post(f"http://{OASEES_IPFS_IP}:5001/api/v0/add", files={"file": ("_file", _file)})
    response_data = resp.json()
    file_cid = response_data['Hash']

    return file_cid



def ipfs_get(cid,OASEES_IPFS_IP):
    resp = requests.post(f"http://{OASEES_IPFS_IP}:5001/api/v0/cat?arg={cid}")
    content = resp.content.decode("UTF-8")
    return content