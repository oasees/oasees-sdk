# from web3 import Web3
# from dotenv import load_dotenv
# import requests
# import json
# import ipfshttpclient
# import os
# import yaml
# from kubernetes import client, config
# from kubernetes.client import ApiException

# current_cluster = 0

# load_dotenv()
# _IPFS_HOST = os.getenv('IPFS_HOST')
# _BLOCK_CHAIN_IP = os.getenv('BLOCK_CHAIN_IP')
# _ACCOUNT_ADDRESS = os.getenv('ACCOUNT_ADDRESS')

# if(_ACCOUNT_ADDRESS):
#     _ACCOUNT_ADDRESS = Web3.to_checksum_address(str(_ACCOUNT_ADDRESS))


# ###### INITIALIZE THE CONNECTIONS TO THE SERVICES AND CONTRACTS INVOLVED ######

# __web3 = Web3(Web3.HTTPProvider(f"http://{_BLOCK_CHAIN_IP}:8545"))                    # BLOCKCHAIN
# __response = requests.get(f'http://{_IPFS_HOST}:6001/ipfs_portal_contracts')
# __data = __response.json()
# __ipfs_json = __data['portal_contracts']


# __nft_abi = __ipfs_json['nft_abi']             
# __nft_address = __ipfs_json['nft_address']
# __marketplace_abi = __ipfs_json['marketplace_abi']
# __marketplace_address = __ipfs_json['marketplace_address']


# __nft = __web3.eth.contract(address=__nft_address, abi=__nft_abi)                           # NFT contract
# __marketplace = __web3.eth.contract(address=__marketplace_address, 
#                                     abi=__marketplace_abi)    

# def _getDevices():
#     results = __marketplace.caller({'from': _ACCOUNT_ADDRESS}).getMyDevices()
#     devices = []

#     client = ipfshttpclient.connect(f"/ip4/{_IPFS_HOST}/tcp/5001")  

#     for r in results:
#         token_id = r[1]
#         content_hash = __nft.functions.tokenURI(token_id).call()

#         content = client.cat(content_hash)
#         content = content.decode("UTF-8")
#         content = json.loads(content)
        
#         devices.append({'name': content['name'], 'endpoint':content['device_endpoint'][7:]})
    
#     client.close()
    
#     return devices

# def _getPurchases():
#     if (_ACCOUNT_ADDRESS):

#         client = ipfshttpclient.connect(f"/ip4/{_IPFS_HOST}/tcp/5001")                    

#         results = __marketplace.caller({'from': _ACCOUNT_ADDRESS}).getMyNfts()
#         purchases=[]

#         for r in results:
#             token_id = r[1]
#             content_hash = __nft.functions.tokenURI(token_id).call()
#             metadata_hash = r[5]

            

#             metadata = client.cat(metadata_hash)
#             metadata = metadata.decode("UTF-8")
#             metadata = json.loads(json.loads(metadata))

#             purchases.append({'contentURI': content_hash, 'title':metadata['title']})

#         client.close()

#         return purchases
    
# def _getClusters():
#     daos = __marketplace.caller({'from':_ACCOUNT_ADDRESS}).getJoinedDaos()
#     devices = __marketplace.caller({'from':_ACCOUNT_ADDRESS}).getMyDevices()
#     clusters = []

   

#     for dao in daos:
#         if(dao[6]):
#             token_id = dao[5]
#             config_hash = __nft.functions.tokenURI(token_id).call()

#             resp = requests.post(f"http://{_IPFS_HOST}:5001/api/v0/cat?arg={dao[2]}")
#             cluster_description = json.loads(resp.content.decode("UTF-8"))

#             clusters.append({'name': cluster_description['dao_name'], 'config_hash':config_hash, 'cluster_ip': cluster_description['cluster_ip']})


#     return clusters

# def _getConfig():

#     clusters = _getClusters()

#     current_config_hash = clusters[current_cluster]['config_hash']

#     client = ipfshttpclient.connect(f"/ip4/{_IPFS_HOST}/tcp/5001") 

#     content = client.cat(current_config_hash)
#     content = content.decode("UTF-8")
#     config = yaml.safe_load(content)


#     with open('config', 'w') as f:
#         yaml.safe_dump(config,f)
    

#     client.close()

# def _get_cluster_from_node(node_name: str):
#     clusters = _getClusters()

#     for i in range (len(clusters)):
#         _switch_cluster(i)
#         _getConfig()
#         config.load_kube_config("./config")
#         k8s_api = client.CoreV1Api()
#         nodes = k8s_api.list_node()
#         for node in nodes.items:
#             if node.metadata.name == node_name:
#                 node_user = node.metadata.labels['user']
#                 return {"cluster_number": i, "node_user": node_user}
    
#     return {"cluster_number": -1 , "node_user": ''}

# def _switch_cluster(cluster_number:int):
#     clusters = _getClusters()
    
#     if(clusters):
#         global current_cluster
#         current_cluster = cluster_number
    
#     else:
#         print("You do not have any Kubernetes clusters registered at the moment.")