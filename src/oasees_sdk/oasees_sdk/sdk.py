import sys
from web3 import Web3
from dotenv import load_dotenv
import ipfshttpclient
import json
import requests
import yaml
import os
import json
import subprocess
import shutil
import zipfile
from .mlops.templating_functions import create_template, convert_to_pipeline 
from .mlops.templating_functions import create_quantum_template,convert_quantum_template
from .mlops.templating_functions import create_fl_template, convert_fl_template
from .mlops.env_vars import Envs

create_template = create_template
convert_to_pipeline = convert_to_pipeline
create_quantum_template = create_quantum_template
convert_quantum_template = convert_quantum_template

from kubernetes import client, config
from kubernetes.client import ApiException
# from .utility_functions import _getPurchases, _getDevices, _getClusters, _getConfig, _switch_cluster, _get_cluster_from_node, _IPFS_HOST

# def get_master_ip():
#     clusters = _getClusters()
#     master_ip = ""
#     if(len(clusters)):
#         master_ip = clusters[0]['cluster_ip']

#     return master_ip

# MASTER_IP = get_master_ip()


Envs.set_envs(f"http://oasees-ipfs.default.svc.cluster.local:5001/api/v0",f"http://oasees-mlops.default.svc.cluster.local:31007")


# def my_clusters():
#     clusters = _getClusters()


#     print("\nOwned clusters")
#     print("---------------------------------")
#     i=1
#     if(clusters):
#         for cluster in clusters:
#             print(str(i) + ") " + cluster['name'])
#             i+=1
    
#     else:
#         print("You do not have any Kubernetes clusters registered at the moment.")