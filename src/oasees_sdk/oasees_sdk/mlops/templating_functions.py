import os
import json
from .ml_template import *
from. util_files import *
from .env_vars import Envs
import subprocess
import shutil
import requests
import re


template = notebook_json

def create_template(filename):


    filepath = os.path.join(filename, filename + ".ipynb")
    if os.path.exists(filename):
        pass
        # print(f"Template '{filename}' already exists")
    else:
        
        os.makedirs(filename, exist_ok=True)
        # os.makedirs("{}/assets".format(filename), exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(template, f, indent=4) 





    helper_filepath = os.path.join(filename, 'oasees_helpers.py')
    with open(helper_filepath, 'w') as f:
        f.write(helper_template.replace("REPLACE_IPFS",Envs.ipfs_endpoint).replace("REPLACE_INDEXER",Envs.indexer_endpoint))


    runtime_params_filepath = os.path.join(filename, 'runtime_params.py')
    with open(runtime_params_filepath, 'w') as f:
        f.write(runtime_params)

    return_results_filepath = os.path.join(filename, 'return_results.py')
    with open(return_results_filepath, 'w') as f:
        f.write(return_results)

    dockerfile_filepath = os.path.join(filename, 'Dockerfile')
    with open(dockerfile_filepath, 'w') as f:
        f.write(DockerFile_template)


    deploy_filepath = os.path.join(filename, 'deploy.py')
    with open(deploy_filepath, 'w') as f:
        f.write(deploy_ml)



    print(f"Oasees ML Template '{filename}' created.")



def convert_to_pipeline(folder_name):
    notebook_path = os.path.join(folder_name, f"{folder_name}.ipynb")
    assets_folder = os.path.join(folder_name, "assets")
    requirements_path = os.path.join(folder_name,"requirements.txt")

    mod_notebook_path = modify_notebook(folder_name)




    command = ["soorgeon", "refactor", mod_notebook_path, "--single-task"]
    result = subprocess.run(command, capture_output=True, text=True,cwd=folder_name)


    # print(folder_name)
    subprocess.run(["pipreqs"], capture_output=True, text=True,cwd=folder_name)

    if(result.stderr):
        print("Error: ",result.stderr)
        return


    with open(requirements_path, 'r') as f:
        lines = f.readlines()
        cleaned_lines = []
        for line in lines:
            # Remove versioning information using regex
            cleaned_line = re.sub(r'([><=]=?[^,\s]+)', '', line).strip()
            if cleaned_line == 'opencv_python':
                continue
            cleaned_lines.append(cleaned_line)

    with open(requirements_path, 'w') as f:
        for cleaned_line in cleaned_lines:
            if cleaned_line:  # Avoid writing empty lines
                f.write(cleaned_line + '\n')






    result = subprocess.run(["ploomber","build","--force"], capture_output=True, text=True,cwd=folder_name)

    clean_up  = [
        f"{folder_name}/{folder_name}_exec_test.ipynb",
        f"{folder_name}/{folder_name}_exec_test-backup.ipynb",
        f"{folder_name}/mod_notebook_path",
        f"{folder_name}/__pycache__",
        f"{folder_name}/products",
        f"{folder_name}/model.pth",
        f"{folder_name}/training_output.log",
        f"{folder_name}/testing_output.log"
    ]

    for c in clean_up:
        if (os.path.exists(c)):
            if(os.path.isdir(c)):
                shutil.rmtree(c)
            else:
                os.remove(c)

    if(result.stdout):
        print("Template is a valid workload")
        upload_to_ipfs(folder_name,Envs.ipfs_endpoint,'ML')
        return

    print(f"{folder_name} Workload is ready! Go to SDK Manager...")

def upload_to_ipfs(folder_path, ipfs_api_url,_type):
    folder_name = os.path.basename(folder_path)
    files = []

    # Walk through the directory and gather files
    for root, _, filenames in os.walk(folder_path):
        for filename in filenames:
            file_path = os.path.join(root, filename)
            files.append(('file', (os.path.relpath(file_path, folder_path), open(file_path, 'rb'))))

    try:
        # Add folder to IPFS
        response = requests.post(f"{Envs.ipfs_endpoint}/add?recursive=true&wrap-with-directory=true", files=files)
        
        if response.status_code == 200:
            # Process response line by line
            folder_cid = None
            for line in response.iter_lines():
                if line:
                    entry = json.loads(line)
                    if entry['Name'] == "":
                        folder_cid = entry['Hash']
                        project_name = folder_path.split("/")[-1]
                        upload_to_indexer(project_name, folder_cid,_type)

            if folder_cid:
                return folder_cid
            else:
                raise Exception("Folder CID not found in the response.")
        else:
            raise Exception(f"Failed to add folder to IPFS: {response.text}")
    finally:
        for _, (_, file) in files:
            file.close()


def upload_to_indexer(project_name, cid,_type):
    url = f'{Envs.indexer_endpoint}/upload_project'
    headers = {
        'Content-Type': 'application/json',
    }
    payload = {
        'project_name': project_name,
        'cid': cid,
        'type': _type
    }

    response = requests.post(url, json=payload, headers=headers)
    print(response.json())
    if response.status_code == 200:
        # print('Workload is ready for deployment')
        print(f"{project_name} Workload is ready! Go to SDK Manager...")
        # print(response.json())
    else:
        print('Failed to add project folder version')


def modify_notebook(folder_name):


    notebook_path = os.path.join(folder_name, f"{folder_name}.ipynb")
    exported_file_path = os.path.join(folder_name, "exported.py")
    deployment_file_path = os.path.join(folder_name,"deploy.py")
    mod_notebook_path = os.path.join(folder_name, f"{folder_name}_exec_test.ipynb")
    params_path = os.path.join(folder_name,"exec_params.json")
    


    with open(f"{notebook_path}", 'r', encoding='utf-8') as file:
        notebook_content = file.read()

    notebook_json = json.loads(notebook_content)

    PARAMS = ['## Execution Parameters']
    IMPORTS = ['## Imports']
    MODEL_DEF = ['## Model Definition']

    _imports = ''
    _model_def = ''
    _deployment_code = ''



    for i in range(0, len(notebook_json['cells'])):
        if notebook_json['cells'][i]['source'] == PARAMS:
            params = notebook_json['cells'][i + 1]['source']
            source_code = ''.join(params)
            print(source_code)
            local_vars = {}
            exec(source_code, {}, local_vars)

            params = local_vars
            exec_params ={
                "epochs": 1,
                "learning_rate": 0.01,
                "ipfs_hash": local_vars["exec_params"]["ipfs_hash"],
                "retrain": 0,
                "sample_num": 1,
                "batch_size": 1           
            }


            modified_source = f'exec_params = {json.dumps(exec_params, indent=4)}\n'
            notebook_json['cells'][i + 1]['source'] = modified_source.splitlines(keepends=True)

            exec_params["ipfs_hash"] = ""

            with open(params_path, 'w') as json_file:
                json.dump(exec_params, json_file, indent=4)

     

        if notebook_json['cells'][i]['source'] == IMPORTS:
            _imports = notebook_json['cells'][i + 1]['source']
            _imports = ''.join(_imports)


        if notebook_json['cells'][i]['source'] == MODEL_DEF:
            _model_def = notebook_json['cells'][i + 1]['source']
            _model_def = ''.join(_model_def)

            # break



    with open(exported_file_path, 'w', encoding='utf-8') as file:
        file.write(_imports+"\n"+_model_def)



    with open(mod_notebook_path, 'w', encoding='utf-8') as file:
        json.dump(notebook_json, file, indent=4)



    return f"{folder_name}_exec_test.ipynb"



def create_quantum_template(filename):


    filepath = os.path.join(filename, filename + ".ipynb")
    if os.path.exists(filename):
        pass
        # print(f"Template '{filename}' already exists")
    else:
        
        os.makedirs(filename, exist_ok=True)
        # os.makedirs("{}/assets".format(filename), exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(quantum_notebook_json, f, indent=4)
    
    print(f"Oasees Quantum Template '{filename}' created.")


def convert_quantum_template(folder_name):
    upload_to_ipfs(folder_name,Envs.ipfs_endpoint,'Q')
    print(f"{folder_name} Workload is ready! Go to SDK Manager...")

def create_fl_template(filename):

    filepath = os.path.join(filename, filename + ".ipynb")
    if os.path.exists(filename):
        pass
        # print(f"Template '{filename}' already exists")
    else:
        
        os.makedirs(filename, exist_ok=True)

        with open(filepath, 'w') as f:
            json.dump(fl_json, f, indent=4)
        
    print(f"Oasees FL Template '{filename}' created")


def convert_fl_template(folder_name):


    notebook_path = os.path.join(folder_name, f"{folder_name}.ipynb")

    with open(f"{notebook_path}", 'r', encoding='utf-8') as file:
        notebook_content = file.read()


    notebook_json = json.loads(notebook_content)

    MODEL = ['## Federated Learning Model']
    SERVER = ['## Federated Learning Server']
    CLIENT = ['## Federated Learning Client']

    _model = ''
    _server = ''
    _client = ''

    for i in range(0, len(notebook_json['cells'])):
        if notebook_json['cells'][i]['source'] == MODEL:
            _model = notebook_json['cells'][i + 1]['source']
            _model = ''.join(_model)

        if notebook_json['cells'][i]['source'] == SERVER:
            _server = notebook_json['cells'][i + 1]['source']
            _server = ''.join(_server)

        if notebook_json['cells'][i]['source'] == CLIENT:
            _client = notebook_json['cells'][i + 1]['source']
            _client = ''.join(_client)


    model_file_path = os.path.join(folder_name,"FlModel.py")
    flower_server_file_path = os.path.join(folder_name,"flower_server.py")
    flower_client_file_path = os.path.join(folder_name,"flower_client.py")

    with open(model_file_path, 'w') as f:
        f.write(_model)


    with open(flower_server_file_path, 'w') as f:
        f.write(_server)


    with open(flower_client_file_path, 'w') as f:
        f.write(_client)



    exec_flower_server_sh_filepath = os.path.join(folder_name, 'exec_flower_server.sh')
    with open(exec_flower_server_sh_filepath, 'w') as f:
        f.write(exec_flower_server)

    exec_flower_client_sh_filepath = os.path.join(folder_name, 'exec_flower_client.sh')
    with open(exec_flower_client_sh_filepath, 'w') as f:
        f.write(exec_flower_client)


    return_results_filepath = os.path.join(folder_name, 'return_results.py')
    with open(return_results_filepath, 'w') as f:
        f.write(return_results_fl)

    deploy_filepath = os.path.join(folder_name, 'deploy.py')
    with open(deploy_filepath, 'w') as f:
        f.write(deploy_fl)




    upload_to_ipfs(folder_name,Envs.ipfs_endpoint,'FL')
    # print(f"{folder_name} Workload is ready! Go to SDK Manager...")
    
