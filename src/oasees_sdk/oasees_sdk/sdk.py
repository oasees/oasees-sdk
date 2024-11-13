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

create_ml_template = create_template
convert_to_pipeline = convert_to_pipeline
create_quantum_template = create_quantum_template
convert_quantum_template = convert_quantum_template

from kubernetes import client, config
from kubernetes.client import ApiException
# from .deploy_pipeline import create_job, create_load_job
# from .cluster_ipfs_upload import assets_to_ipfs
# from .training_workload import deploy_workload
from .utility_functions import _getPurchases, _getDevices, _getClusters, _getConfig, _switch_cluster, _get_cluster_from_node, _IPFS_HOST

def get_master_ip():
    clusters = _getClusters()
    master_ip = ""
    if(len(clusters)):
        master_ip = clusters[0]['cluster_ip']

    return master_ip

MASTER_IP = get_master_ip()


Envs.set_envs(f"http://{MASTER_IP}:31005/api/v0",f"http://{MASTER_IP}:31007")


# # Listing functions for the OASEES items that belong to the user (Algorithms, Devices, Clusters)

# def my_algorithms():
#     '''Returns a list with all the algorithms purchased from your account
#         on the OASEES Marketplace.''' 

#     purchases = _getPurchases()


#     print("\nOwned algorithms")
#     print("---------------------------------")
#     i=1
#     if(purchases):
#         for purchase in purchases:
#             print(str(i) + ") " + purchase['title'])
#             i+=1
    
#     else:
#         print("You have not bought any items from the marketplace yet.")



# def my_devices():
#     '''Returns a list with all the devices purchased / uploaded from your account
#         on the OASEES Marketplace.''' 

#     devices = _getDevices()

#     print("\nOwned devices")
#     print("---------------------------------")
#     i=1
#     if(devices):
#         for device in devices:
#             print(str(i) + ") " + device['name'] + " | " + device['endpoint'])
#             i+=1
    
#     else:
#         print("You have not bought any devices from the marketplace yet.")





def my_clusters():
    clusters = _getClusters()


    print("\nOwned clusters")
    print("---------------------------------")
    i=1
    if(clusters):
        for cluster in clusters:
            print(str(i) + ") " + cluster['name'])
            i+=1
    
    else:
        print("You do not have any Kubernetes clusters registered at the moment.")





# # Kubernetes object deployment functions

# def deploy_algorithm(algorithm_title:str,node_name):
#     '''Deploys a purchased algorithm on all your connected devices.

#         - algorithm_title: Needs to be provided in "string" form.
    
#         e.g. algorithm.py -> deploy_algorithm("algorithm.py")
#     '''

#     purchases = _getPurchases()
#     found = False
#     for purchase in purchases:
#         if found:
#             break

#         if(purchase['title']==algorithm_title):
#             found = True
#             node_info = _get_cluster_from_node(node_name)
    
#             if(node_info['cluster_number'] > -1):
#                 ipfs_cid = purchase['contentURI']

#                 _switch_cluster(node_info['cluster_number'])
#                 _getConfig()

#                 with open('config', 'r') as f:
#                     kube_config = yaml.safe_load(f)

#                 master_ip = kube_config['clusters'][0]['cluster']['server']
#                 master_ip = master_ip.split(':')


#                 ipfs_api_url = "http://{}:5001".format(_IPFS_HOST)
#                 app_name = algorithm_title
#                 print(app_name.split(".")[0])
#                 config.load_kube_config("./config")
#                 # config.load_kube_config()
#                 batch_v1 = client.BatchV1Api()
#                 resp = create_load_job(ipfs_api_url,batch_v1,ipfs_cid, app_name, node_name, node_info['node_user'])
#                 print(resp)


#             else:
#                 print("The specified node not was not found in your clusters.")
            

#     if not found:
#         print("The file you requested was not found in your purchases.")


# def deploy_manifest(manifest_file_path,node_name):
#     '''Deploys all the objects included in your specified manifest file, on the
#     Kubernetes cluster associated with your blockchain account.

#     - manifest_file_path: Needs to be providerd in "string" form.

#     e.g. manifest.yaml -> build_image("manifest.yaml")'''


#     node_info = _get_cluster_from_node(node_name)
    
#     if(node_info['cluster_number'] > -1):
#         _switch_cluster(node_info['cluster_number'])
#         _getConfig()

#         # Load kube config
#         config.load_kube_config("./config")

#         api_instance = client.CustomObjectsApi()

#         try:
#             # Read manifest file
#             with open(manifest_file_path, 'r') as f:
#                 manifest_documents = yaml.safe_load_all(f)

#                 # Iterate over each document and deploy it
#                 for manifest in manifest_documents:
#                     try:
#                         if manifest['kind'] == 'Service':
#                             # Deploy Service
#                             api_response = client.CoreV1Api().create_namespaced_service(
#                                 namespace="default",
#                                 body=manifest
#                             )
#                         elif manifest['kind'] == 'Ingress':
#                             # Deploy Ingress
#                             api_response = client.NetworkingV1Api().create_namespaced_ingress(
#                                 namespace="default",
#                                 body=manifest
#                             )
#                         else:
#                             # Deploy other resources
#                             api_response = api_instance.create_namespaced_custom_object(
#                                 group=manifest['apiVersion'].split('/')[0],
#                                 version=manifest['apiVersion'].split('/')[1],
#                                 namespace="default",
#                                 plural=manifest['kind'].lower() + 's',  # Convert resource kind to plural form
#                                 body=manifest
#                             )
#                         print("Manifest deployed successfully!")
#                     except Exception as e:
#                         print(f"Error deploying manifest: {e}")
#                         print("Problematic manifest:")
#                         print(yaml.dump(manifest))  # Print the problematic manifest
#         except Exception as e:
#             print(f"Error reading manifest file: {e}")

#     else: 
#         print("The specified node not was not found in your clusters.")


# # MLOps functions

# def create_ml_pipeline(filename):
#     # Create the directory if it doesn't exist
#     if os.path.exists(filename):
#         print(f"The ML pipeline '{filename}' already exists")
#         return

#     os.makedirs(filename, exist_ok=True)
#     os.makedirs("{}/assets".format(filename), exist_ok=True)

#     # Create the .ipynb file inside the directory
#     filepath = os.path.join(filename, filename + ".ipynb")
#     with open(filepath, 'w') as f:
#         json.dump(template, f, indent=4)  # Use json.dump() for proper JSON formatting
    
#     # Create the oasees_loggers.py file inside the directory
#     logger_filepath = os.path.join(filename, 'oasees_loggers.py')
#     with open(logger_filepath, 'w') as f:
#         f.write(logger_template)

#     print(f"Oasees ML Template '{filename}' created.")


# def convert_to_pipeline(filename):
#     notebook_path = os.path.join(filename, f"{filename}.ipynb")
#     assets_folder = os.path.join(filename, "assets")
#     exported_file_path = os.path.join(filename, "exported.py")
#     requirements_file = os.path.join(filename, "requirements.txt")

#     if not os.path.isfile(notebook_path):
#         raise FileNotFoundError(f"Notebook {notebook_path} not found.")
    

#     if os.path.isfile(requirements_file):
#         os.remove(requirements_file)

#     original_cwd = os.getcwd()
#     os.chdir(filename)
    
#     try:
#         command = ["soorgeon", "refactor", f"{filename}.ipynb"]
#         result = subprocess.run(command, check=True, text=True, capture_output=True)
#         print(f"ML pipeline '{filename}' created successfully.")
#     except subprocess.CalledProcessError as e:
#         print(f"Error running soorgeon refactor: {e.stderr}")
#         raise
#     finally:
#         os.chdir(original_cwd)

#     shutil.copy(exported_file_path, os.path.join(assets_folder, "exported.py"))


#     if os.path.isfile(requirements_file):
#         with open(requirements_file, 'r') as rf:
#             lines = rf.readlines()
#         with open(requirements_file, 'w') as nf:
#             for line in lines:
#                 if 'ploomber' not in line.lower():
#                     nf.write(line)

#     shutil.copy(requirements_file, os.path.join(assets_folder, "requirements.txt"))


#     try:
#         shutil.make_archive(filename, 'zip', original_cwd, filename)
#     except Exception as e:
#         print(f"Error creating zip archive: {e}")
#         raise


# def deploy_training_workload(filename,node_name):
#     _getConfig()

#     with open('config', 'r') as f:
#         kube_config = yaml.safe_load(f)

#     master_ip = kube_config['clusters'][0]['cluster']['server']
#     master_ip = master_ip.split(':')

#     ipfs_api_url = "http://{}:31005/api/v0/add".format(master_ip[1][2:])

#     deploy_workload(filename,ipfs_api_url,node_name)


# def download_trained_model(filename):
#     _getConfig()

#     with open('config', 'r') as f:
#         kube_config = yaml.safe_load(f)

#     master_ip = kube_config['clusters'][0]['cluster']['server']
#     master_ip = master_ip.split(':')
#     master_ip = master_ip[1][2:]

#     output_file = f"assets-{filename}.zip"

#     url = "http://{}/training-pod-{}/download_assets".format(master_ip,filename.replace("_","-").lower())

#     # Perform the GET request to download the file
#     try:
#         with requests.get(url, stream=True) as response:
#             response.raise_for_status()  # Check if the request was successful
#             with open(output_file, 'wb') as file:
#                 for chunk in response.iter_content(chunk_size=8192):
#                     file.write(chunk)
#         print(f"File downloaded successfully and saved as {output_file}")
#     except requests.exceptions.RequestException as e:
#         print(f"Training is still in progress.")
#         return

#     try:
#         with zipfile.ZipFile(output_file, 'r') as zip_ref:
#             # Extract all the contents into the specified directory
#             extract_dir = f"assets-{filename}"
#             os.makedirs(extract_dir, exist_ok=True)
#             zip_ref.extractall(extract_dir)
#     except zipfile.BadZipFile as e:
#         print(f"Error unzipping the file: {e}")

#     # Optionally, remove the downloaded zip file if you don't need it anymore
#     if os.path.exists(output_file):
#         os.remove(output_file)
#         print(f"Downloaded zip file {output_file} has been removed.")

#     namespace_name = "{}".format(filename.replace("_","-").lower())
#     print(namespace_name)

#     config.load_kube_config("./config")
#     v1 = client.CoreV1Api()
#     # Delete the namespace
#     try:
#         api_response = v1.delete_namespace(namespace_name)
#         print(f"Namespace {namespace_name} deleted successfully.")
#     except ApiException as e:
#         print(f"Exception when deleting namespace: {e}")


# def build_image(image_folder_path):

#     '''Deploys a job on the Kubernetes cluster associated with your blockchain
#     account, which builds a container image out of your specified folder.
#     The image will then be stored on your master node, and will be available
#     for deployment on any of the cluster's nodes specified in your manifest file. 
    
#         - image_folder_path: Needs to be provided in "string" form.

#     e.g. DApp_Image_Folder -> build_image("DApp_Image_Folder")

#     '''

#     _getConfig()

#     with open('config', 'r') as f:
#         kube_config = yaml.safe_load(f)

#     master_ip = kube_config['clusters'][0]['cluster']['server']
#     master_ip = master_ip.split(':')

#     ipfs_api_url = "http://{}:31005".format(master_ip[1][2:])
#     directory_path = image_folder_path
#     ipfs_cid = assets_to_ipfs(ipfs_api_url, directory_path)
#     print(ipfs_cid)
#     app_name = directory_path.split('/')[-1].lower()
#     print(app_name)
#     config.load_kube_config("./config")
#     # config.load_kube_config()
#     batch_v1 = client.BatchV1Api()
#     resp = create_job(batch_v1,ipfs_cid, app_name)
#     print(resp)

# def sync_clusters():

#     config_hashes = []
#     clusters = _getClusters()

#     if clusters:
#         url = f"http://{clusters[0]['cluster_ip']}:30000/sync_clusters"

#         for cluster in clusters:
#             config_hashes.append(cluster['config_hash'])

        
#         payload = {'config_hashes': config_hashes}
#         headers = {'Content-Type': 'application/json'}

#         response = requests.post(url, json=payload, headers=headers)

#         if response.ok:
#             print(f"Response: {response.json()}")
#         else:
#             print(f"Failed to send string. Status code: {response.status_code}")


# def instructions():
    
#     # \033[1m{deploy_algorithm.__name__}\033[0m(algorithm_title: str) \t \t
#     #     {deploy_algorithm.__doc__}


#     # \033[1m{deploy_local_file.__name__}\033[0m(path: str) \t   \t
#     #     {deploy_local_file.__doc__}

#     text = (f'''
#     \033[1m{my_algorithms.__name__}\033[0m() \t\t
#         {my_algorithms.__doc__}


#     \033[1m{my_devices.__name__}\033[0m() \t   \t
#         {my_devices.__doc__}
    
        
#     \033[1m{build_image.__name__}\033[0m() \t  \t
#         {build_image.__doc__}

        
#     \033[1m{deploy_manifest.__name__}\033[0m() \t      \t
#         {deploy_manifest.__doc__}

        
#     \033[1m{instructions.__name__}\033[0m() \t \t
#         Reprints the above documentation.
#         ''')
    
#     __print_msg_box(text,title="\033[1mOASEES SDK methods\033[0m \t \t")

 
# def __print_msg_box(msg, indent=1, width=None, title=None):
#     """Print message-box with optional title."""
#     lines = msg.split('\n')
#     space = " " * indent
#     if not width:
#         width = max(map(len, lines))
#     box = f'╔{"═" * (width + indent * 2)}╗\n'  # upper_border
#     if title:
#         box += f'║{space}{title:<{width}}{space}║\n'  # title
#         box += f'║{space}{"-" * len(title):<{width}}{space}║\n'  # underscore
#     box += ''.join([f'║{space}{line:<{width}}{space}║\n' for line in lines])
#     box += f'╚{"═" * (width + indent * 2)}╝'  # lower_border
#     print(box)

# instructions()



# # def get_label(node_name: str):
# #     clusters = __getClusters()

# #     for i in range (len(clusters)):
# #         __switch_cluster(i)
# #         __getConfig()
# #         config.load_kube_config("./config")
# #         k8s_api = client.CoreV1Api()
# #         nodes = k8s_api.list_node()
# #         for node in nodes.items:
# #             if node.metadata.name == node_name:
# #                 print(node.metadata.labels['user'])
            

# # def deploy_local_file(path:str):
# #     '''Deploys the file found in the specified path on all your connected devices.
    
# #         - path: -> Needs to be provided in "string" form.
# #                 -> Is equal to the filename when the file is located in
# #                    the Jupyter Notebook's directory.
    
# #         e.g. algorithm.py -> deploy_local_file("algorithm.py")
# #     '''

# #     devices = _getDevices()
# #     file = open(path,"rb")

# #     for device in devices:
# #         __response= requests.post("http://{}/deploy_file".format(device['endpoint']), files={'file': file})                 
# #         print(__response.text)
    
# #     file.close()
