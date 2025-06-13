import sys
from web3 import Web3
from pathlib import Path
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
from .mlops.util_files import *
from ipylab import JupyterFrontEnd


def ipfs_add():
    

    _name = "../{}".format(Path.cwd().name)

    cmd = ["oasees-sdk","mlops","ipfs-add",_name]

    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if (result.returncode):
        raise Exception("Please restart the kernel")
    else:
        print("Project created")


def convert():
    import subprocess
    from pathlib import Path

    app = JupyterFrontEnd()
    app.commands.execute('docmanager:save')

    _name = Path.cwd().name

    convert_files = [_name,"{}_svc.ipynb".format(_name)]

    for cf in convert_files:
        print(cf)
        nbconvert_cmd = [
            "jupyter",
            "nbconvert",
            cf,
            "--to",
            "script",
            "--TagRemovePreprocessor.enabled=True",
            "--TagRemovePreprocessor.remove_cell_tags=['skip-execution']"

        ]
        result = subprocess.run(nbconvert_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    # if (result.returncode == 0):
    ipfs_add()

def list_sample_data():
    cmd = ["oasees-sdk","mlops","ipfs-ls"]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out = result.stdout
    parsed = out.split('\n')[2:]
    for p in parsed:
        if(".npy" in p):
            print(p)

def get_sample_data(file1, file2):
   cmd1 = ["oasees-sdk","mlops","ipfs-get", file1]
   cmd2 = ["oasees-sdk","mlops","ipfs-get", file2]
   
   result1 = subprocess.run(cmd1, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
   result2 = subprocess.run(cmd2, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
   
   if result1.returncode == 0:
       print(f"Downloaded {file1}")
   else:
       print(f"Failed to download {file1}: {result1.stderr}")
       
   if result2.returncode == 0:
       print(f"Downloaded {file2}")
   else:
       print(f"Failed to download {file2}: {result2.stderr}")


def run_test_fl_server():

    from pathlib import Path
    import os
    _path = Path.cwd()

    fl_server_file = os.path.join(_path, 'fl_server.py')
    with open(fl_server_file, 'w') as f:
        f.write(fl_server)

    os.system("python fl_server.py --NUM_ROUNDS 1 --MIN_CLIENTS 1 --MODEL_NAME {} &".format(_path.name))






# from .mlops.templating_functions import create_template, convert_to_pipeline 
# from .mlops.templating_functions import create_quantum_template,convert_quantum_template
# from .mlops.templating_functions import create_fl_template, convert_fl_template
# from .mlops.env_vars import Envs

# create_template = create_template
# convert_to_pipeline = convert_to_pipeline
# create_quantum_template = create_quantum_template
# convert_quantum_template = convert_quantum_template

# from kubernetes import client, config
# from kubernetes.client import ApiException
# from .utility_functions import _getPurchases, _getDevices, _getClusters, _getConfig, _switch_cluster, _get_cluster_from_node, _IPFS_HOST

# def get_master_ip():
#     clusters = _getClusters()
#     master_ip = ""
#     if(len(clusters)):
#         master_ip = clusters[0]['cluster_ip']

#     return master_ip

# MASTER_IP = get_master_ip()


# Envs.set_envs(f"http://oasees-ipfs.default.svc.cluster.local:5001/api/v0",f"http://oasees-mlops.default.svc.cluster.local:31007")


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