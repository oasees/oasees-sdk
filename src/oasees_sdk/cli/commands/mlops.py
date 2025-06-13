import click
import subprocess
import json
import os
from pathlib import Path
from .jupyter_templating import *
import numpy as np
import argparse
import shutil

@click.group(name='mlops')
def mlops_commands():
    '''OASEES MLOPS Utilities'''
    pass


@mlops_commands.command()
@click.argument('data_file', default='')
@click.argument('target_file', default='')
def prepare_dataset(data_file,target_file):
    """Prepares datasets for training through OASEES"""
    data = np.load(data_file)
    target = np.load(target_file)
    n_samples = 100
    
    synth_data = np.zeros((n_samples, *data.shape[1:]), dtype=data.dtype)
    
    unique_classes = np.unique(target)
    n_classes = len(unique_classes)
    
    synth_target = np.zeros(n_samples, dtype=target.dtype)
    
    # Create equal distribution of classes
    samples_per_class = n_samples // n_classes
    remainder = n_samples % n_classes
    
    idx = 0
    for i, class_label in enumerate(unique_classes):
        # Give extra samples to first 'remainder' classes
        class_samples = samples_per_class + (1 if i < remainder else 0)
        synth_target[idx:idx + class_samples] = class_label
        idx += class_samples
    
    np.random.shuffle(synth_target)
    
    data_basename = os.path.splitext(os.path.basename(data_file))[0]
    target_basename = os.path.splitext(os.path.basename(target_file))[0]
    
    output_data_file = f'/var/tmp/synth_{data_basename}.npy'
    if os.path.exists(output_data_file):
        os.remove(output_data_file)

    output_target_file = f'/var/tmp/synth_{target_basename}.npy'
    if os.path.exists(output_target_file):
        os.remove(output_target_file)
    
    original_data_dest = f'/var/tmp/{os.path.basename(data_file)}'
    if os.path.exists(original_data_dest):
        os.remove(original_data_dest)    
    original_target_dest = f'/var/tmp/{os.path.basename(target_file)}'
    if os.path.exists(original_target_dest):
        os.remove(original_target_dest) 


    shutil.copy2(data_file, original_data_dest)
    shutil.copy2(target_file, original_target_dest)
    
    np.save(output_data_file, synth_data)
    np.save(output_target_file, synth_target)
    


def get_ipfs_api():
    """Get IPFS service IP from Kubernetes"""
    try:
        ipfs_svc = subprocess.run(['kubectl','get','svc','oasees-ipfs','-o','jsonpath={.spec.clusterIP}'], 
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        ipfs_ip = ipfs_svc.stdout.strip()
        return f"/ip4/{ipfs_ip}/tcp/5001/http"
    except subprocess.CalledProcessError as e:
        click.secho(f"Error getting IPFS service: {e.stderr}", fg="red", err=True)
        return None

def test_ipfs_connection(api_endpoint, quiet=False):
    """Test connection to IPFS node"""
    test_cmd = subprocess.run(['ipfs', f'--api={api_endpoint}', 'id'],  
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    if test_cmd.returncode != 0:
        click.secho(f"Cannot connect to IPFS node: {test_cmd.stderr}", fg="red", err=True)
        return False
    
    if not quiet:
        click.secho("Connected to IPFS node", fg="green")
    return True

def ensure_oasees_ml_ops_folder(api_endpoint, quiet=False):
    """Ensure /oasees-ml-ops folder exists in MFS"""
    # Check if folder exists
    stat_cmd = ['ipfs', f'--api={api_endpoint}', 'files', 'stat', '/oasees-ml-ops']
    stat_result = subprocess.run(stat_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    if stat_result.returncode != 0:

        
        mkdir_cmd = ['ipfs', f'--api={api_endpoint}', 'files', 'mkdir', '/oasees-ml-ops']
        mkdir_result = subprocess.run(mkdir_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        if mkdir_result.returncode != 0:
            return False
        
    
    return True

def normalize_mfs_path(path):
    """Convert relative path to absolute path under /oasees-ml-ops"""
    if path.startswith('/'):
        return path
    else:
        # Remove leading ./ if present
        if path.startswith('./'):
            path = path[2:]
        return f"/oasees-ml-ops/{path}"

@mlops_commands.command()
@click.argument('path', default='')
@click.option('--long', '-l', is_flag=True, help='Long format listing')
@click.option('--quiet', '-q', is_flag=True, help='Minimal output')
def ipfs_ls(path, long, quiet):
    '''List files in IPFS MFS (defaults to /oasees-ml-ops)'''
    
    try:
        api_endpoint = get_ipfs_api()
        if not api_endpoint:
            return
        
        if not test_ipfs_connection(api_endpoint, quiet):
            return
        
        if not ensure_oasees_ml_ops_folder(api_endpoint, quiet):
            return
        
        # Normalize path
        if not path:
            mfs_path = '/oasees-ml-ops'
        else:
            mfs_path = normalize_mfs_path(path)
        
        # Build ls command
        ls_cmd = ['ipfs', f'--api={api_endpoint}', 'files', 'ls']
        
        if long:
            ls_cmd.append('-l')
        
        ls_cmd.append(mfs_path)
        
        if not quiet:
            click.echo(f"Listing: {mfs_path}")
        
        # Execute ls
        result = subprocess.run(ls_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        if result.returncode == 0:
            if result.stdout.strip():
                click.echo(result.stdout.strip())
            else:
                if not quiet:
                    click.echo("Directory is empty")
        else:
            click.secho(f"List failed: {result.stderr}", fg="red", err=True)
            
    except Exception as e:
        click.secho(f"Unexpected error: {str(e)}", fg="red", err=True)

@mlops_commands.command()
@click.argument('source')
@click.argument('dest')
@click.option('--quiet', '-q', is_flag=True, help='Minimal output')
def ipfs_cp(source, dest, quiet):
    '''Copy files in IPFS MFS (paths relative to /oasees-ml-ops)'''
    
    try:
        api_endpoint = get_ipfs_api()
        if not api_endpoint:
            return
        
        if not test_ipfs_connection(api_endpoint, quiet):
            return
        
        if not ensure_oasees_ml_ops_folder(api_endpoint, quiet):
            return
        
        # Normalize paths
        source_path = normalize_mfs_path(source)
        dest_path = normalize_mfs_path(dest)
        
        # Build cp command
        cp_cmd = ['ipfs', f'--api={api_endpoint}', 'files', 'cp', source_path, dest_path]
        
        if not quiet:
            click.echo(f"Copying: {source_path} -> {dest_path}")
        
        # Execute cp
        result = subprocess.run(cp_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        if result.returncode == 0:
            if not quiet:
                click.secho("Copy successful", fg="green")
        else:
            click.secho(f"Copy failed: {result.stderr}", fg="red", err=True)
            
    except Exception as e:
        click.secho(f"Unexpected error: {str(e)}", fg="red", err=True)

@mlops_commands.command()
@click.argument('mfs_path')
@click.option('--output', '-o', help='Custom local filename/directory (default: use MFS filename)')
@click.option('--directory', '-d', default='.', help='Local directory to save to (default: current directory)')
@click.option('--quiet', '-q', is_flag=True, help='Minimal output')
def ipfs_get(mfs_path, output, directory, quiet):
    '''Download file/folder from IPFS MFS (paths relative to /oasees-ml-ops)'''
    
    try:
        api_endpoint = get_ipfs_api()
        if not api_endpoint:
            return
        
        if not test_ipfs_connection(api_endpoint, quiet):
            return
        
        if not ensure_oasees_ml_ops_folder(api_endpoint, quiet):
            return
        
        # Normalize MFS path
        full_mfs_path = normalize_mfs_path(mfs_path)
        
        # Get the base name from MFS path
        base_name = os.path.basename(full_mfs_path.rstrip('/'))
        
        # Use custom output name if provided, otherwise use the original name
        local_name = output if output else base_name
        
        # Build the full local path
        local_path = os.path.join(directory, local_name)
        
        if not quiet:
            click.echo(f"Downloading: {full_mfs_path} -> {local_path}")
        
        # Step 1: Get the hash from MFS
        stat_cmd = ['ipfs', f'--api={api_endpoint}', 'files', 'stat', '--hash', full_mfs_path]
        
        if not quiet:
            click.echo("Getting file hash from MFS...")
        
        stat_result = subprocess.run(stat_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        if stat_result.returncode != 0:
            click.secho(f"Failed to get file hash: {stat_result.stderr}", fg="red", err=True)
            return
        
        file_hash = stat_result.stdout.strip()
        
        if not quiet:
            click.echo(f"File hash: {file_hash}")
            click.echo("Downloading from IPFS...")
        
        # Step 2: Download using ipfs get
        get_cmd = ['ipfs', f'--api={api_endpoint}', 'get', file_hash, '-o', local_path]
        
        get_result = subprocess.run(get_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        if get_result.returncode == 0:
            if not quiet:
                click.secho("Download successful", fg="green")
                click.echo(f"Local path: {local_path}")
                click.echo(f"IPFS Hash: {file_hash}")
                
                # Show file/directory info
                if os.path.isdir(local_path):
                    click.echo("Downloaded as directory")
                elif os.path.isfile(local_path):
                    file_size = os.path.getsize(local_path)
                    click.echo(f"Downloaded file ({file_size} bytes)")
            else:
                click.echo(local_path)
        else:
            click.secho(f"Download failed: {get_result.stderr}", fg="red", err=True)
            
    except Exception as e:
        click.secho(f"Unexpected error: {str(e)}", fg="red", err=True)

@mlops_commands.command()
@click.argument('path')
@click.option('--parents', '-p', is_flag=True, help='Create parent directories')
@click.option('--quiet', '-q', is_flag=True, help='Minimal output')
def ipfs_mkdir(path, parents, quiet):
    '''Create directory in IPFS MFS (paths relative to /oasees-ml-ops)'''
    
    try:
        api_endpoint = get_ipfs_api()
        if not api_endpoint:
            return
        
        if not test_ipfs_connection(api_endpoint, quiet):
            return
        
        if not ensure_oasees_ml_ops_folder(api_endpoint, quiet):
            return
        
        # Normalize path
        full_path = normalize_mfs_path(path)
        
        # Build mkdir command
        mkdir_cmd = ['ipfs', f'--api={api_endpoint}', 'files', 'mkdir']
        
        if parents:
            mkdir_cmd.append('-p')
        
        mkdir_cmd.append(full_path)
        
        if not quiet:
            click.echo(f"Creating directory: {full_path}")
        
        # Execute mkdir
        result = subprocess.run(mkdir_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        if result.returncode == 0:
            if not quiet:
                click.secho("Directory created", fg="green")
        else:
            click.secho(f"Create directory failed: {result.stderr}", fg="red", err=True)
            
    except Exception as e:
        click.secho(f"Unexpected error: {str(e)}", fg="red", err=True)

@mlops_commands.command()
@click.argument('path')
@click.option('--hash', is_flag=True, help='Show only hash')
@click.option('--size', is_flag=True, help='Show only size')
@click.option('--quiet', '-q', is_flag=True, help='Minimal output')
def ipfs_stat(path, hash, size, quiet):
    '''Get file/directory stats from IPFS MFS (paths relative to /oasees-ml-ops)'''
    
    try:
        api_endpoint = get_ipfs_api()
        if not api_endpoint:
            return
        
        if not test_ipfs_connection(api_endpoint, quiet):
            return
        
        if not ensure_oasees_ml_ops_folder(api_endpoint, quiet):
            return
        
        # Normalize path
        full_path = normalize_mfs_path(path)
        
        # Build stat command
        stat_cmd = ['ipfs', f'--api={api_endpoint}', 'files', 'stat']
        
        if hash:
            stat_cmd.append('--hash')
        elif size:
            stat_cmd.append('--size')
        
        stat_cmd.append(full_path)
        
        if not quiet and not (hash or size):
            click.echo(f"Stats for: {full_path}")
        
        # Execute stat
        result = subprocess.run(stat_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        if result.returncode == 0:
            click.echo(result.stdout.strip())
        else:
            click.secho(f"Stat failed: {result.stderr}", fg="red", err=True)
            
    except Exception as e:
        click.secho(f"Unexpected error: {str(e)}", fg="red", err=True)

@mlops_commands.command()
@click.argument('local_path', type=click.Path(exists=True))
@click.option('--name', help='Custom name in MFS (default: use original filename)')
@click.option('--path', help='Custom MFS path (default: root of /oasees-ml-ops)')
@click.option('--recursive', '-r', is_flag=True, help='Upload directory recursively')
@click.option('--quiet', '-q', is_flag=True, help='Minimal output')
def ipfs_add(local_path, name, path, recursive, quiet):
    '''Upload local file/directory to IPFS MFS (/oasees-ml-ops)'''
    
    try:
        api_endpoint = get_ipfs_api()
        if not api_endpoint:
            return
        
        if not test_ipfs_connection(api_endpoint, quiet):
            return
        
        if not ensure_oasees_ml_ops_folder(api_endpoint, quiet):
            return
        
        # Get the base name of the file/directory
        base_name = os.path.basename(local_path.rstrip('/'))
        
        # Use custom name if provided, otherwise use the original name
        target_name = name if name else base_name
        
        # Build the MFS path
        if path:
            mfs_path = normalize_mfs_path(f"{path.rstrip('/')}/{target_name}")
        else:
            mfs_path = f"/oasees-ml-ops/{target_name}"
        
        # Always remove existing content if it exists
        if not quiet:
            click.echo(f"Checking for existing content at {mfs_path}...")
        
        check_cmd = ['ipfs', f'--api={api_endpoint}', 'files', 'stat', mfs_path]
        check_result = subprocess.run(check_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        if check_result.returncode == 0:
            if not quiet:
                click.echo(f"Removing existing content at {mfs_path}...")
            
            rm_cmd = ['ipfs', f'--api={api_endpoint}', 'files', 'rm', '-r', mfs_path]
            rm_result = subprocess.run(rm_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            
            if rm_result.returncode != 0:
                if not quiet:
                    click.secho(f"Warning: Could not remove existing content: {rm_result.stderr}", fg="yellow")
        
        if not quiet:
            if os.path.isdir(local_path):
                click.echo(f"Uploading directory: {base_name} -> {mfs_path}")
            else:
                click.echo(f"Uploading file: {base_name} -> {mfs_path}")
        
        # Step 1: Add to IPFS and get hash
        add_cmd = ['ipfs', f'--api={api_endpoint}', 'add', '-q']
        
        # Auto-detect if it's a directory or use explicit recursive flag
        if recursive or os.path.isdir(local_path):
            add_cmd.append('-r')
            if not quiet:
                click.echo("Uploading recursively...")
        
        add_cmd.append(local_path)
        
        if not quiet:
            click.echo("Adding to IPFS...")
        
        # Execute add command
        add_result = subprocess.run(add_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        if add_result.returncode != 0:
            click.secho(f"IPFS add failed: {add_result.stderr}", fg="red", err=True)
            return
        
        # Get the hash
        hash_lines = add_result.stdout.strip().split('\n')
        if not hash_lines or not hash_lines[-1].strip():
            click.secho("No hash returned from IPFS add", fg="red", err=True)
            return
            
        file_hash = hash_lines[-1].strip()
        
        if not quiet:
            click.echo(f"Added to IPFS with hash: {file_hash}")
            click.echo("Copying to MFS...")
        
        # Step 2: Copy from IPFS to MFS using the hash
        cp_cmd = ['ipfs', f'--api={api_endpoint}', 'files', 'cp', f'/ipfs/{file_hash}', mfs_path]
        
        cp_result = subprocess.run(cp_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        if cp_result.returncode == 0:
            if not quiet:
                click.secho("Successfully uploaded to MFS", fg="green")
                click.echo(f"IPFS Hash: {file_hash}")
                click.echo(f"MFS Path: {mfs_path}")
            else:
                click.echo(file_hash)
        else:
            click.secho(f"Failed to copy to MFS: {cp_result.stderr}", fg="red", err=True)
            if not quiet:
                click.echo(f"File is available in IPFS with hash: {file_hash}")
            
    except Exception as e:
        click.secho(f"Unexpected error: {str(e)}", fg="red", err=True)

@mlops_commands.command()
@click.argument('name')
def init(name):
    """Create a new project folder with an empty OASEES notebook"""
    

    folder_path = Path(name)
    folder_path.mkdir(exist_ok=True)
    

    training_notebook = notebook([
        code_cell("""
        from oasees_sdk import oasees_sdk
        oasees_sdk.ipfs_add()
        """, readonly=True,tags=["readonly", "skip-execution"]),



        code_cell("""       
        import time
        import argparse
        import flwr as fl
        import numpy as np
        import warnings
        import logging
        logging.getLogger("flwr").setLevel(logging.ERROR)
        warnings.filterwarnings("ignore", category=UserWarning, module="flwr")
        """, readonly=True,tags=["readonly"]),
        
        markdown_cell("## IMPORTS"),

        code_cell("""

        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import log_loss


        """, editable=True,tags=[]),


        code_cell("oasees_sdk.list_sample_data()", readonly=True,tags=["readonly", "skip-execution"]),

        code_cell("""
        parser = argparse.ArgumentParser(description='Flower Federated Learning Client')
        parser.add_argument('--SERVER_ADDRESS', type=str, default='127.0.0.1:9999', help='Server address')
        parser.add_argument('--DATA_PATH', type=str, default='/var/tmp/samples.npy', help='Path to data file')
        parser.add_argument('--TARGET_PATH', type=str, default='/var/tmp/labels.npy', help='Path to target file')
        parser.add_argument('--TRAIN_SPLIT', type=float, default=0.8, help='Training split ratio')
        parser.add_argument('--CLIENT_ID', type=str, default='0', help='Client ID')
        parser.add_argument('--EPOCHS', type=int, default=10, help='Number of epochs')
        args = parser.parse_args()
        SERVER_ADDRESS = args.SERVER_ADDRESS
        DATA_PATH = args.DATA_PATH
        TARGET_PATH = args.TARGET_PATH
        TRAIN_SPLIT = args.TRAIN_SPLIT
        CLIENT_ID = args.CLIENT_ID
        EPOCHS = args.EPOCHS                  
        """,readonly=True,tags=["readonly"]),


        code_cell("""
        DATA_PATH = ""
        TARGET_PATH = "" 
        oasees_sdk.get_sample_data(DATA_PATH, TARGET_PATH)
        """, editable=True,tags=["skip-execution"]),

        code_cell("""
        SERVER_ADDRESS = "localhost:9999"
        TRAIN_SPLIT = 0.8
        CLIENT_ID = 0
        EPOCHS = 10
        """,editable=False,tags=["skip-execution"]),

        code_cell("""
        def model_definition():
            model = None
            ################USER INPUT############################
            model = RandomForestClassifier() 



            ######################################################
            return model
        model = model_definition()
        """,editable=True,tags=[]),

        code_cell("""
        def data_load():
            X, y = np.load(DATA_PATH), np.load(TARGET_PATH)
            if X.ndim > 2:
                X = X.reshape(X.shape[0], -1)

            unique_classes = np.unique(y)
            train_indices = []
            test_indices = []
            
            for class_label in unique_classes:
                class_indices = np.where(y == class_label)[0]
                np.random.shuffle(class_indices)
                
                n_train = int(len(class_indices) * TRAIN_SPLIT)
                train_indices.extend(class_indices[:n_train])
                test_indices.extend(class_indices[n_train:])
            
            train_indices = np.array(train_indices)
            test_indices = np.array(test_indices)
            
            return X[train_indices], X[test_indices], y[train_indices], y[test_indices] 

        X_train, X_test, y_train, y_test = data_load()
        """,readonly=True,tags=[]),

        code_cell("""
        class Fl_client(fl.client.NumPyClient):

            def __init__(self, model, X_train, X_test, y_train, y_test, client_id, epochs):
                self.model = model
                self.X_train = X_train
                self.X_test = X_test
                self.y_train = y_train
                self.y_test = y_test
                self.client_id = client_id
                self.epochs = epochs

            def get_parameters(self, config):
                params = None
                if hasattr(self.model, 'get_weights'):
                    params = self.model.get_weights()
                elif hasattr(self.model, 'state_dict'):
                    params = [p.detach().cpu().numpy() for p in self.model.state_dict().values()]
                else:
                    if hasattr(self.model, 'estimators_'):
                        params = [np.array([len(self.model.estimators_)])]
                    else:
                        params = [np.array([1.0])] 
                return params

            def fit(self, parameters, config):
                train_metrics = None
                ################USER INPUT############################
                self.model.fit(self.X_train, self.y_train)
                accuracy = self.model.score(self.X_train, self.y_train)
                loss = log_loss(self.y_train, self.model.predict_proba(self.X_train))
                
                train_metrics = {
                    "train_accuracy": accuracy,
                    "train_loss": loss
                }
                  


                ######################################################
                train_metrics = {
                    "train_accuracy": accuracy,
                    "train_loss": loss
                }
                
                updated_params = self.get_parameters(config)
                return updated_params, len(self.X_train), train_metrics

            def evaluate(self, parameters, config):
                eval_metrics = None
                ################USER INPUT############################

                accuracy = self.model.score(self.X_test, self.y_test)
                loss = log_loss(self.y_test, self.model.predict_proba(self.X_test))  


                ######################################################
                eval_metrics = {
                    "eval_accuracy": accuracy,
                    "eval_loss": loss
                }
                
                
                return loss, len(self.X_test), eval_metrics
                  
        client = Fl_client(
            model=model,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            client_id=CLIENT_ID,
            epochs=EPOCHS
        )

        """,editable=True,tags=[]),
        
        code_cell("""
        oasees_sdk.run_test_fl_server()
        time.sleep(1)
        fl.client.start_client(
            server_address=SERVER_ADDRESS,
            client=client.to_client()
        )
        """,readonly=True,tags=["skip-execution"]),

        code_cell("""
        def main():
            fl.client.start_client(
                server_address=SERVER_ADDRESS,
                client=client.to_client()
            )

        if __name__ == "__main__":
            main()
        """, readonly=True,tags=["readonly"]),


        code_cell("""
        oasees_sdk.convert()
        """,readonly=True,tags=["skip-execution"]),


    ])

    training_notebook_path = folder_path / f"{name}.ipynb"

    with open(training_notebook_path, 'w', encoding='utf-8') as f:
        json.dump(training_notebook, f, indent=2)




    deployment_notebook = notebook([

        code_cell("""
        from oasees_sdk import oasees_sdk
        """, readonly=True,tags=["skip-execution"]),


        code_cell("""
        from flask import Flask, request, jsonify
        import pickle
        import io
        import numpy as np
        """, readonly=True,tags=["readonly"]),

        code_cell("""
        import torch
        import torch.nn as nn
        """),


        code_cell("""
        def model_definition():
            class SimpleNN(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.fc1 = nn.Linear(4, 64)
                    self.fc2 = nn.Linear(64, 32)
                    self.fc3 = nn.Linear(32, 3)
                    self.relu = nn.ReLU()
                    
                def forward(self, x):
                    x = self.relu(self.fc1(x))
                    x = self.relu(self.fc2(x))
                    return self.fc3(x)
            
            model = SimpleNN()
            return model
        """),

        code_cell("""
        def load_fl_model(pkl_path):
            with open(pkl_path, 'rb') as f:
                model_data = pickle.load(f)
            
            parameters = model_data['parameters']
            model = model_definition()
            param_list = list(model.parameters())
            
            for i, param_array in enumerate(parameters):
                if i < len(param_list):
                    param_tensor = torch.from_numpy(param_array).float()
                    param_list[i].data = param_tensor
            
            model.eval()
            return model
        """),

        code_cell("""
        model = load_fl_model('test_model')
        """),


        code_cell("""
        import threading
        app = Flask(__name__)
        @app.route('/predict', methods=['POST'])
        def predict():
            try:
                file = request.files['file']
                
                data = np.load(io.BytesIO(file.read()))
                
                if data.ndim == 1:
                    data = data.reshape(1, -1)
                
                input_tensor = torch.from_numpy(data).float()
                
                with torch.no_grad():
                    output = model(input_tensor)
                    probabilities = torch.softmax(output, dim=1)
                    predicted_classes = torch.argmax(output, dim=1)
                
                results = {
                    'predictions': predicted_classes.tolist(),
                    'probabilities': probabilities.tolist(),
                    'raw_output': output.tolist()
                }
                
                return jsonify(results)
                
            except Exception as e:
                return jsonify({'error': str(e)}), 400
                """),

        code_cell("""
        def run_server():
            app.run(debug=False, host='0.0.0.0', port=5005, use_reloader=False)
        """,readonly=True,tags=["readonly"]),

        code_cell("""
        threading.Thread(target=run_server, daemon=True).start()
        print("Server running at http://localhost:5000")
        """,readonly=True,tags=["readonly"]),

        code_cell("""
        oasees_sdk.convert()
        """,readonly=True,tags=["skip-execution"]),



    ])    
  
    deployment_notebook_path = folder_path / f"{name}_svc.ipynb"

    with open(deployment_notebook_path, 'w', encoding='utf-8') as f:
        json.dump(deployment_notebook, f, indent=2)


    
    click.echo(f"Created project '{name}' with OASEES notebook")