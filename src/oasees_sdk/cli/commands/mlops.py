import click
import subprocess
import json
import os
from pathlib import Path
from .jupyter_templating import *
import numpy as np
import argparse
import shutil
import re

@click.group(name='mlops')
def mlops_commands():
    '''OASEES MLOPS Utilities'''
    pass


def sanitize_k8s_name(name):
    """Sanitize names for Kubernetes (lowercase, replace underscores with hyphens, remove invalid chars)"""
    sanitized = name.lower()
    sanitized = re.sub(r'_', '-', sanitized)
    sanitized = re.sub(r'[^a-z0-9-]', '', sanitized)
    sanitized = re.sub(r'^-+|-+$', '', sanitized)
    return sanitized




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
    
    samples_per_class = n_samples // n_classes
    remainder = n_samples % n_classes
    
    idx = 0
    for i, class_label in enumerate(unique_classes):
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
    

@mlops_commands.command()
@click.option('--model', required=True, help='Model filename (required)')
@click.option('--project-name', help='Project name (optional, derived from model name if not provided)')
@click.option('--pod-name', help='Pod name (optional, defaults to sanitized model name)')
def deploy_model(model, project_name, pod_name):
    """Deploy a trained model for inference"""
    
    # Get IPFS IP
    ipfs_api = get_ipfs_api()
    if not ipfs_api:
        return
    
    # Extract IPFS IP from API string
    ipfs_ip = ipfs_api.split('/')[2]
    
    # Extract project name from model if not provided
    if not project_name:
        # Extract project name from model filename (everything before the first underscore)
        project_name = model.split('_')[0] if '_' in model else None
        if not project_name:
            click.secho(f"Error: Could not derive project name from model filename '{model}'", fg="red")
            click.secho("Please provide --project-name explicitly", fg="red")
            return
        click.echo(f"Derived project name from model: {project_name}")
    
    # Set pod name if not provided (use model name without extension)
    if not pod_name:
        # Extract base name from model file (remove .pkl extension)
        model_base = os.path.splitext(os.path.basename(model))[0]
        pod_name = sanitize_k8s_name(model_base)
    
    # Sanitize all names for Kubernetes
    k8s_pod_name = sanitize_k8s_name(pod_name)
    k8s_app_label = sanitize_k8s_name(model)
    
    # Display configuration
    click.echo("Deploying Kubernetes resources with the following configuration:")
    click.echo(f"  Project Name: {project_name}")
    click.echo(f"  IPFS IP: {ipfs_ip}")
    click.echo(f"  Model: {model}")
    click.echo(f"  Pod Name: {k8s_pod_name}")
    click.echo(f"  App Label: {k8s_app_label}")
    click.echo()
    
    # Create YAML content
    yaml_content = f"""apiVersion: v1
kind: Pod
metadata:
  name: {k8s_pod_name}
  labels:
    app: {k8s_app_label}
    tag: model
spec:
  restartPolicy: Never
  containers:
  - name: model-deploy
    image: ghcr.io/oasees/ml-base-image:latest
    env:
      - name: PROJECT_NAME
        value: "{project_name}"
      - name: IPFS_IP
        value: "{ipfs_ip}"
      - name: MODEL
        value: "{model}"
    command: ["/bin/bash", "-c"]
    args:
    - |
      HASH=$(ipfs --api=/ip4/${{IPFS_IP}}/tcp/5001/http files stat --hash /oasees-ml-ops/projects/ml/${{PROJECT_NAME}}) &&
      ipfs --api=/ip4/${{IPFS_IP}}/tcp/5001/http get ${{HASH}} -o ${{PROJECT_NAME}} &&
      cd $PROJECT_NAME &&
      python  ${{PROJECT_NAME}}_deploy.py --model-path $MODEL
    ports:
    - containerPort: 5005

---
apiVersion: v1
kind: Service
metadata:
  name: {k8s_pod_name}-service
  labels:
    tag: model
spec:
  type: NodePort
  selector:
    app: {k8s_app_label}
    tag: model
  ports:
  - port: 30005
    targetPort: 5005
"""

    # Display generated manifest
    click.echo("Generated Kubernetes manifest:")
    click.echo("=" * 32)
    click.echo(yaml_content)
    click.echo("=" * 32)
    click.echo()
    
    # Apply to Kubernetes
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as temp_file:
            temp_file.write(yaml_content)
            temp_file_path = temp_file.name
        
        click.echo("Applying Kubernetes configuration...")
        result = subprocess.run(['kubectl', 'apply', '-f', temp_file_path], 
                              capture_output=True, text=True, check=True)
        
        click.secho("✅ Deployment successful!", fg="green")
        
    except subprocess.CalledProcessError as e:
        click.secho("❌ Deployment failed!", fg="red")
        click.secho(f"Error: {e.stderr}", fg="red")
    finally:
        # Clean up temporary file
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)





def get_ipfs_api():
    """Get IPFS service IP from Kubernetes"""
    try:
        ipfs_svc = subprocess.run(['kubectl','get','svc','oasees-ipfs','-o','jsonpath={.spec.clusterIP}'], 
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        ipfs_ip = ipfs_svc.stdout.strip()
        print(ipfs_ip)
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
    
    # if not quiet:
    #     click.secho("Connected to IPFS node", fg="green")
    return True

def ensure_oasees_ml_ops_folder(api_endpoint, quiet=False):
    """Ensure /oasees-ml-ops folder exists in MFS"""

    folders_to_create = [
        '/oasees-ml-ops/projects',
        '/oasees-ml-ops/projects/ml',
        '/oasees-ml-ops/projects/quantum',
        '/oasees-ml-ops/data_paths',
        '/oasees-ml-ops/synthetic_data'
    ]


    stat_cmd = ['ipfs', f'--api={api_endpoint}', 'files', 'stat', '/oasees-ml-ops']
    stat_result = subprocess.run(stat_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    if stat_result.returncode != 0:

        
        mkdir_cmd = ['ipfs', f'--api={api_endpoint}', 'files', 'mkdir', '/oasees-ml-ops']
        mkdir_result = subprocess.run(mkdir_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        if mkdir_result.returncode != 0:
            return False
        
    for folder in folders_to_create:
        stat_cmd = ['ipfs', f'--api={api_endpoint}', 'files', 'stat', folder]
        stat_result = subprocess.run(stat_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        if stat_result.returncode == 0:
            continue
        
        mkdir_cmd = ['ipfs', f'--api={api_endpoint}', 'files', 'mkdir', '-p', folder]
        mkdir_result = subprocess.run(mkdir_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


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

import click
import subprocess
import tempfile
import os
from datetime import datetime

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

@mlops_commands.command()
@click.option('--project-name', required=True, help='Project name for the FL system')
@click.option('--data-files', required=True, 
              help='Comma-separated list of data:target:node triplets (data1.npy,target1.npy,node1:data2.npy,target2.npy,node2:...)')
@click.option('--min-clients', type=int, help='Minimum number of clients (default: number of data file pairs)')
@click.option('--num-rounds', type=int, default=5, help='Number of rounds (default: 5)')
@click.option('--epochs', type=int, default=5, help='Number of epochs per client (default: 5)')
def start_fl(project_name, data_files, min_clients, num_rounds, epochs):
    """Start Federated Learning system with server and clients"""
    
    def wait_for_pod_ready(pod_name, timeout=300, check_interval=5):
        """Wait for a pod to be ready"""
        import time
        start_time = time.time()
        click.echo(f"Waiting for {pod_name} to be ready...")
        
        while time.time() - start_time < timeout:
            try:
                result = subprocess.run(['kubectl', 'get', 'pod', pod_name, '-o', 'jsonpath={.status.phase}'], 
                                      capture_output=True, text=True, check=True)
                if result.stdout.strip() == 'Running':
                    # Additional check to ensure container is ready
                    result = subprocess.run(['kubectl', 'get', 'pod', pod_name, '-o', 'jsonpath={.status.containerStatuses[0].ready}'], 
                                          capture_output=True, text=True, check=True)
                    if result.stdout.strip() == 'true':
                        click.secho(f"✅ {pod_name} is ready!", fg="green")
                        return True
                elif result.stdout.strip() == 'Failed':
                    click.secho(f"❌ {pod_name} failed to start!", fg="red")
                    return False
                
                click.echo(f"Pod status: {result.stdout.strip()}, waiting...")
                time.sleep(check_interval)
            except subprocess.CalledProcessError:
                click.echo(f"Pod {pod_name} not found yet, waiting...")
                time.sleep(check_interval)
        
        click.secho(f"❌ Timeout waiting for {pod_name} to be ready", fg="red")
        return False
    
    # Get IPFS IP
    ipfs_api = get_ipfs_api()
    if not ipfs_api:
        return
    
    # Extract IPFS IP from API string
    ipfs_ip = ipfs_api.split('/')[2]
    
    # Parse data files
    try:
        file_triplets = data_files.split(':')
        data_array = []
        target_array = []
        node_array = []
        
        for triplet in file_triplets:
            files = triplet.split(',')
            if len(files) != 3:
                click.secho(f"Error: Each data file triplet must be in format 'data.npy,target.npy,node-name'", fg="red")
                click.secho(f"Invalid triplet: {triplet}", fg="red")
                return
            data_array.append(files[0])
            target_array.append(files[1])
            node_array.append(files[2])
    except Exception as e:
        click.secho(f"Error parsing data files: {e}", fg="red")
        return
    
    num_clients = len(data_array)
    if min_clients is None:
        min_clients = num_clients
    
    # Generate timestamp and names
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    server_pod_name = f"fl-server-{project_name}-{timestamp}"
    service_name = f"fl-server-{project_name}-{timestamp}"
    
    # Display configuration
    click.echo(f"\nDeploying FL System with the following configuration:")
    click.echo(f"  Project Name: {project_name}")
    click.echo(f"  IPFS IP: {ipfs_ip}")
    click.echo(f"  Server Pod: {server_pod_name} (on master node)")
    click.echo(f"  Service Name: {service_name}")
    click.echo(f"  Number of Clients: {num_clients}")
    click.echo(f"  Min Clients: {min_clients}")
    click.echo(f"  Num Rounds: {num_rounds}")
    click.echo(f"  Epochs: {epochs}")
    click.echo(f"\nClient Configuration:")
    for i, (data, target, node) in enumerate(zip(data_array, target_array, node_array)):
        click.echo(f"  Client {i}: {data} -> {target} (Node: {node})")
    click.echo()
    
    # Create Server YAML content
    server_yaml_content = f"""# FL Server Pod (always on master node)
apiVersion: v1
kind: Pod
metadata:
  name: {server_pod_name}
  labels:
    app: fl-server
    project: {project_name}
    component: server
    tag: fl-system
spec:
  restartPolicy: Never
  nodeSelector:
    node-role.kubernetes.io/master: "true"
  containers:
  - name: fl-server
    image: ghcr.io/oasees/ml-base-image:latest
    command: ["/bin/bash", "-c"]
    env:
      - name: PROJECT_NAME
        value: "{project_name}"
      - name: IPFS_IP
        value: "{ipfs_ip}"
      - name: MIN_CLIENTS
        value: "{min_clients}"
      - name: NUM_ROUNDS
        value: "{num_rounds}"        
    args:
    - |
      HASH=$(ipfs --api=/ip4/${{IPFS_IP}}/tcp/5001/http files stat --hash /oasees-ml-ops/projects/ml/${{PROJECT_NAME}}) &&
      ipfs --api=/ip4/${{IPFS_IP}}/tcp/5001/http get ${{HASH}} -o ${{PROJECT_NAME}} &&
      python ${{PROJECT_NAME}}/fl_server.py \\
        --NUM_ROUNDS ${{NUM_ROUNDS}} \\
        --MODEL_NAME ${{PROJECT_NAME}} \\
        --MIN_CLIENTS ${{MIN_CLIENTS}} &&
      TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S") &&
      HASH=$(ipfs --api=/ip4/$IPFS_IP/tcp/5001/http add $PROJECT_NAME.pkl -q) &&
      echo ${{HASH}} &&
      ipfs --api=/ip4/${{IPFS_IP}}/tcp/5001/http files cp  /ipfs/${{HASH}} /oasees-ml-ops/projects/ml/${{PROJECT_NAME}}/${{PROJECT_NAME}}_${{TIMESTAMP}}.pkl 
    ports:
    - containerPort: 9999

---
# FL Server Service
apiVersion: v1
kind: Service
metadata:
  name: {service_name}
  labels:
    project: {project_name}
    component: server
    tag: fl-system
spec:
  selector:
    app: fl-server
    project: {project_name}
    component: server
  ports:
  - port: 9999
    targetPort: 9999
"""

    # Create Clients YAML content
    clients_yaml_content = ""
    for i, (data_file, target_file, node_name) in enumerate(zip(data_array, target_array, node_array)):
        client_pod_name = f"fl-client-{project_name}-{i}-{timestamp}"
        data_path = f"/var/tmp/{data_file}"
        target_path = f"/var/tmp/{target_file}"
        
        clients_yaml_content += f"""---
# FL Client Pod {i} (on {node_name} node)
apiVersion: v1
kind: Pod
metadata:
  name: {client_pod_name}
  labels:
    app: fl-client
    project: {project_name}
    component: client
    tag: fl-system
    client-id: "{i}"
spec:
  restartPolicy: Never
  nodeSelector:
    kubernetes.io/hostname: {node_name}
  containers:
  - name: fl-client
    image: ghcr.io/oasees/ml-base-image:latest
    env:
      - name: PROJECT_NAME
        value: "{project_name}"
      - name: SERVER_ADDRESS
        value: "{service_name}:9999"
      - name: IPFS_IP
        value: "{ipfs_ip}"
      - name: MIN_CLIENTS
        value: "{min_clients}"
      - name: EPOCHS
        value: "{epochs}"
      - name: CLIENT_ID
        value: "{i}"
      - name: DATA_PATH
        value: "{data_path}"
      - name: TARGET_PATH
        value: "{target_path}"
        
    command: ["/bin/bash", "-c"]
    args:
    - |
      HASH=$(ipfs --api=/ip4/${{IPFS_IP}}/tcp/5001/http files stat --hash /oasees-ml-ops/projects/ml/${{PROJECT_NAME}}) &&
      ipfs --api=/ip4/${{IPFS_IP}}/tcp/5001/http get ${{HASH}} -o ${{PROJECT_NAME}} &&
      sleep 1 &&
      python $PROJECT_NAME/${{PROJECT_NAME}}_client.py \\
        --SERVER_ADDRESS $SERVER_ADDRESS \\
        --DATA_PATH $DATA_PATH \\
        --TARGET_PATH $TARGET_PATH \\
        --EPOCHS $EPOCHS --CLIENT_ID $CLIENT_ID

    volumeMounts:
    - name: data-files
      mountPath: {data_path}
    - name: target-files
      mountPath: {target_path}
  volumes:
  - name: data-files
    hostPath:
      path: {data_path}
      type: File
  - name: target-files
    hostPath:
      path: {target_path}
      type: File
"""

    server_temp_file_path = None
    clients_temp_file_path = None
    
    try:
        # Step 1: Deploy Server and Service
        click.echo("\n🚀 Step 1: Deploying FL Server...")
        with tempfile.NamedTemporaryFile(mode='w', suffix='_server.yaml', delete=False) as temp_file:
            temp_file.write(server_yaml_content)
            server_temp_file_path = temp_file.name
        
        result = subprocess.run(['kubectl', 'apply', '-f', server_temp_file_path], 
                              capture_output=True, text=True, check=True)
        click.secho("✅ Server deployment initiated!", fg="green")
        
        # Step 2: Wait for Server to be ready
        click.echo("\n⏳ Step 2: Waiting for FL Server to be ready...")
        if not wait_for_pod_ready(server_pod_name, timeout=300):
            click.secho("❌ Server failed to start. Aborting client deployment.", fg="red")
            return
        
        # Step 3: Deploy Clients
        click.echo("\n🚀 Step 3: Deploying FL Clients...")
        with tempfile.NamedTemporaryFile(mode='w', suffix='_clients.yaml', delete=False) as temp_file:
            temp_file.write(clients_yaml_content)
            clients_temp_file_path = temp_file.name
        
        result = subprocess.run(['kubectl', 'apply', '-f', clients_temp_file_path], 
                              capture_output=True, text=True, check=True)
        
        click.secho("\n✅ Deployment successful!", fg="green")
        click.echo(f"\nFL System Information:")
        click.echo(f"  Server Pod: {server_pod_name}")
        click.echo(f"  Server Service: {service_name}")
        click.echo(f"  Number of Clients: {num_clients}")
        
        click.echo(f"\nClient Pods:")
        for i, (data, target, node) in enumerate(zip(data_array, target_array, node_array)):
            client_pod_name = f"fl-client-{project_name}-{i}-{timestamp}"
            click.echo(f"  Client {i}: {client_pod_name} ({data} -> {target}, Node: {node})")
        
        click.echo(f"\nUseful commands:")
        click.echo(f"  # Check all pods status")
        click.echo(f"  kubectl get pods -l project={project_name}")
        click.echo(f"\n  # View server logs")
        click.echo(f"  kubectl logs {server_pod_name} -f")
        click.echo(f"\n  # View client logs")
        for i in range(num_clients):
            client_pod_name = f"fl-client-{project_name}-{i}-{timestamp}"
            click.echo(f"  kubectl logs {client_pod_name} -f  # Client {i}")
        click.echo(f"\n  # Delete entire FL Deployment")
        click.echo(f"  kubectl delete pods,services -l project={project_name}")
        
    except subprocess.CalledProcessError as e:
        click.secho(f"\n❌ Deployment failed!", fg="red")
        click.secho(f"Error: {e.stderr}", fg="red")
    finally:
        # Clean up temporary files
        if server_temp_file_path and os.path.exists(server_temp_file_path):
            os.unlink(server_temp_file_path)
        if clients_temp_file_path and os.path.exists(clients_temp_file_path):
            os.unlink(clients_temp_file_path)














@mlops_commands.command()
@click.argument('path', default='')
@click.option('--long', '-l', is_flag=True, help='Long format listing')
@click.option('--quiet', '-q', is_flag=True, help='Minimal output')
def ipfs_ls(path, long, quiet):
    '''List files in IPFS MFS (defaults to /oasees-ml-ops)'''

    api_endpoint = get_ipfs_api()
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
def fl_data_nodes():
    '''Retrieve nodes with data for Federated Learning through OASEES '''
    api_endpoint = get_ipfs_api()
    ls_cmd = ['ipfs', f'--api={api_endpoint}', 'files', 'ls','/oasees-ml-ops/data_paths']
    result = subprocess.run(ls_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    if result.returncode == 0:
        if result.stdout.strip():
            files = result.stdout.strip().split('\n')
            
            datasets = {}
            
            for file in files:
                file = file.strip()
                if not file:
                    continue
                
                parts = file.split('-')
                if len(parts) >= 2:
                    filename = '-'.join(parts[:-1])  
                    node = parts[-1]   
                    
                    if '_' in filename:
                        dataset_prefix = filename.split('_')[0]
                        
                        key = f"{dataset_prefix}-{node}"
                        
                        if key not in datasets:
                            datasets[key] = []
                        
                        datasets[key].append(filename)
            
            for key, filenames in datasets.items():
                dataset_prefix, node = key.split('-', 1)
                files_str = ','.join(sorted(filenames))
                print(f"{files_str},{node}")

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
        
        full_path = normalize_mfs_path(path)
        
        mkdir_cmd = ['ipfs', f'--api={api_endpoint}', 'files', 'mkdir']
        
        if parents:
            mkdir_cmd.append('-p')
        
        mkdir_cmd.append(full_path)
        
        if not quiet:
            click.echo(f"Creating directory: {full_path}")
        
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
@click.option('--quiet', '-q', is_flag=True, help='Minimal output')
def ipfs_rm(path, quiet):
    '''Remove file or directory from IPFS MFS (paths relative to /oasees-ml-ops)'''
    
    try:
        api_endpoint = get_ipfs_api()
        if not api_endpoint:
            return
        
        if not test_ipfs_connection(api_endpoint, quiet):
            return
        
        full_path = normalize_mfs_path(path)
        
        rm_cmd = ['ipfs', f'--api={api_endpoint}', 'files', 'rm', '-r', full_path]
        
        if not quiet:
            click.echo(f"Removing: {full_path}")
        
        result = subprocess.run(rm_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        if result.returncode == 0:
            if not quiet:
                click.secho("Successfully removed", fg="green")
        else:
            click.secho(f"Remove failed: {result.stderr}", fg="red", err=True)
            
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
def init_project(name):
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
        import os
        import torch
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  
        logging.getLogger("flwr").setLevel(logging.ERROR)
        warnings.filterwarnings("ignore", category=UserWarning, module="flwr")
                

        """, readonly=True,tags=["readonly"]),
        
        markdown_cell("## IMPORTS"),

        code_cell("""

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
             
            def set_parameters(self, parameters):
                if hasattr(self.model, 'get_weights'):
                    weights = self.model.get_weights()
                    for i, param in enumerate(parameters):
                        weights[i].assign(param)
                
                elif hasattr(self.model, 'state_dict'):
                    state_dict = self.model.state_dict()
                    param_keys = list(state_dict.keys())
                    for i, param in enumerate(parameters):
                        if i < len(param_keys):
                            state_dict[param_keys[i]] = torch.tensor(param)
                    self.model.load_state_dict(state_dict)
                else:
                    pass

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
                self.set_parameters(parameters)
                ################USER INPUT############################

                  


                ######################################################
                train_metrics = {
                    "train_accuracy": accuracy,
                    "train_loss": loss
                }
                
                updated_params = self.get_parameters(config)
                return updated_params, len(self.X_train), train_metrics

            def evaluate(self, parameters, config):
                eval_metrics = None
                self.set_parameters(parameters)
                ################USER INPUT############################


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
        time.sleep(2)
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

    training_notebook_path = folder_path / f"{name}_client.ipynb"

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
        import argparse
        import os
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

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
        parser = argparse.ArgumentParser(description='Flask ML Model Service')
        parser.add_argument('--model-path', required=True, help='Path to the pickled model file')
        args = parser.parse_args()
        """, readonly=True,tags=["readonly"]),


        code_cell("""
        model = load_fl_model('test_model')
        """,tags=["skip-execution"]),

        code_cell("""
        model = load_fl_model(args.model_path)
        """, readonly=True,tags=["readonly"]),



        code_cell("""
        import threading
        app = Flask(__name__)
                  
        @app.route('/status', methods=['GET'])
        def status():
            return jsonify({'status': 'up'})


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
        print("Server running at http://localhost:5005")
        """,readonly=True,tags=["readonly","skip-execution"]),

        code_cell("""
        run_server()
        """,readonly=True,tags=["readonly"]),


        code_cell("""
        oasees_sdk.convert(deploy=True)
        """,readonly=True,tags=["skip-execution"]),



    ])    
  
    deployment_notebook_path = folder_path / f"{name}_deploy.ipynb"

    with open(deployment_notebook_path, 'w', encoding='utf-8') as f:
        json.dump(deployment_notebook, f, indent=2)


    
    click.echo(f"Created project '{name}' with OASEES notebook")


@mlops_commands.command()
@click.argument('name')
def init_example_pytorch(name):
    """Create a new project folder with an empty OASEES notebook"""
    

    folder_path = Path(name)
    folder_path.mkdir(exist_ok=True)
    
    fl_server_file = os.path.join(folder_path, 'fl_server.py')
    with open(fl_server_file, 'w') as f:
        f.write(fl_server)



    training_notebook = notebook([
        code_cell("""
        from oasees_sdk import oasees_sdk
        oasees_sdk.ipfs_add()
        """, readonly=True,tags=["readonly", "skip-execution"]),



        code_cell("""
        import os
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'                  
        import time
        import argparse
        import flwr as fl
        import numpy as np
        import warnings
        import logging
        logging.getLogger("flwr").setLevel(logging.ERROR)
        warnings.filterwarnings("ignore", category=UserWarning, module="flwr")
        """, readonly=True,tags=["readonly"]),

        code_cell("""
        %%capture
        try:
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
        except:
            pass                  
        """,readonly=True,tags=["readonly"]),



        code_cell("""
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import TensorDataset, DataLoader
        """, editable=True,tags=[]),


        code_cell("oasees_sdk.list_sample_data()", readonly=True,tags=["readonly", "skip-execution"]),




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
            ################USER INPUT############################
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
                self.set_parameters(parameters)
                ################USER INPUT############################
                X_tensor = torch.FloatTensor(self.X_train)
                y_tensor = torch.LongTensor(self.y_train)
                
                # Create DataLoader
                dataset = TensorDataset(X_tensor, y_tensor)
                dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
                
                # Define optimizer and loss
                optimizer = optim.Adam(self.model.parameters(), lr=0.001)
                criterion = nn.CrossEntropyLoss()
                
                # Training loop
                self.model.train()
                for epoch in range(self.epochs):
                    for batch_X, batch_y in dataloader:
                        optimizer.zero_grad()
                        outputs = self.model(batch_X)
                        loss = criterion(outputs, batch_y)
                        loss.backward()
                        optimizer.step()


                self.model.eval()
                with torch.no_grad():
                    outputs = self.model(X_tensor)
                    loss = criterion(outputs, y_tensor).item()
                    _, predicted = torch.max(outputs, 1)
                    accuracy = (predicted == y_tensor).float().mean().item()
                
                  
                ######################################################
                train_metrics = {
                    "train_accuracy": accuracy,
                    "train_loss": loss
                }
                
                updated_params = self.get_parameters(config)
                return updated_params, len(self.X_train), train_metrics

            def evaluate(self, parameters, config):
                eval_metrics = None
                self.set_parameters(parameters)
                ################USER INPUT############################

                X_test_tensor = torch.FloatTensor(self.X_test)
                y_test_tensor = torch.LongTensor(self.y_test)
                
                # Evaluation
                self.model.eval()
                with torch.no_grad():
                    outputs = self.model(X_test_tensor)
                    loss = nn.CrossEntropyLoss()(outputs, y_test_tensor).item()
                    
                    # Calculate accuracy
                    _, predicted = torch.max(outputs.data, 1)
                    total = y_test_tensor.size(0)
                    correct = (predicted == y_test_tensor).sum().item()
                    accuracy = correct / total
                
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
        time.sleep(2)
        fl.client.start_client(
            server_address=SERVER_ADDRESS,
            client=client.to_client()
        )
        """,readonly=True,tags=["skip-execution"]),

        code_cell("""
        %%capture
        try:
            def main():
                fl.client.start_client(
                    server_address=SERVER_ADDRESS,
                    client=client.to_client()
                )

            if __name__ == "__main__":
                main()
        except:
            pass
        """, readonly=True,tags=["readonly"]),


        code_cell("""
        oasees_sdk.convert()
        """,readonly=True,tags=["skip-execution"]),


    ])

    training_notebook_path = folder_path / f"{name}_client.ipynb"

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
        import argparse
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
        %%capture
        try:
            parser = argparse.ArgumentParser(description='Flask ML Model Service')
            parser.add_argument('--model-path', required=True, help='Path to the pickled model file')
            args = parser.parse_args()
        except:
            pass
        """, readonly=True,tags=["readonly"]),


        code_cell("""
        model = load_fl_model('')
        """,tags=["skip-execution"]),

        code_cell("""
        %%capture
        try:
            model = load_fl_model(args.model_path)
        except:
            pass
        """, readonly=True,tags=["readonly"]),



        code_cell("""
        import threading
        app = Flask(__name__)
                  
        @app.route('/status', methods=['GET'])
        def status():
            return jsonify({'status': 'up'})


        @app.route('/predict', methods=['POST'])
        def predict():
            try:
                json_data = request.get_json()
                data = np.array(json_data['data'])
                
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
        """,readonly=False,tags=["readonly"]),

        code_cell("""
        threading.Thread(target=run_server, daemon=True).start()
        print("Server running at http://localhost:5005")
        """,readonly=True,tags=["readonly","skip-execution"]),

        code_cell("""
        import requests
        import numpy as np
                  
        data = np.load('')[0]
        response = requests.post('http://localhost:5005/predict', 
                        json={'data': data.tolist()})
        print(response.json())           
        """,readonly=False,tags=["skip-execution"]),


        code_cell("""
        %%capture
        try:
            run_server()
        except:
            pass
        """,readonly=True,tags=["readonly"]),


        code_cell("""
        oasees_sdk.convert(deploy=True)
        """,readonly=True,tags=["skip-execution"]),



    ])    
  
    deployment_notebook_path = folder_path / f"{name}_deploy.ipynb"

    with open(deployment_notebook_path, 'w', encoding='utf-8') as f:
        json.dump(deployment_notebook, f, indent=2)


    
    click.echo(f"Created project '{name}' with OASEES notebook")


@mlops_commands.command()
@click.argument('name')
def init_example_tensorflow(name):
    """Create a tensorflow example project"""
    

    folder_path = Path(name)
    folder_path.mkdir(exist_ok=True)
    
    fl_server_file = os.path.join(folder_path, 'fl_server.py')
    with open(fl_server_file, 'w') as f:
        f.write(fl_server)



    training_notebook = notebook([
        code_cell("""
        from oasees_sdk import oasees_sdk
        oasees_sdk.ipfs_add()
        """, readonly=True,tags=["readonly", "skip-execution"]),



        code_cell("""
        import os
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'                  
        import time
        import argparse
        import flwr as fl
        import numpy as np
        import warnings
        import logging
        import torch
        logging.getLogger("flwr").setLevel(logging.ERROR)
        warnings.filterwarnings("ignore", category=UserWarning, module="flwr")
        """, readonly=True,tags=["readonly"]),

        code_cell("""
        %%capture
        try:
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
        except:
            pass                  
        """,readonly=True,tags=["readonly"]),



        code_cell("""
        import tensorflow as tf
        tf.config.set_visible_devices([], 'GPU')
        from sklearn.metrics import accuracy_score
        """, editable=True,tags=[]),


        code_cell("oasees_sdk.list_sample_data()", readonly=True,tags=["readonly", "skip-execution"]),




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
            ################USER INPUT############################
            class SimpleModel:
                def __init__(self):
                    self.w1 = tf.Variable(tf.random.normal([4, 64]), trainable=True)
                    self.b1 = tf.Variable(tf.zeros([64]), trainable=True)
                    self.w2 = tf.Variable(tf.random.normal([64, 32]), trainable=True)
                    self.b2 = tf.Variable(tf.zeros([32]), trainable=True)
                    self.w3 = tf.Variable(tf.random.normal([32, 3]), trainable=True)
                    self.b3 = tf.Variable(tf.zeros([3]), trainable=True)
                    
                def __call__(self, x):
                    x = tf.nn.relu(tf.matmul(x, self.w1) + self.b1)
                    x = tf.nn.relu(tf.matmul(x, self.w2) + self.b2)
                    return tf.matmul(x, self.w3) + self.b3
                    
                def get_weights(self):
                    return [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3]
            
            model = SimpleModel()
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

            def set_parameters(self, parameters):
                if hasattr(self.model, 'get_weights'):
                    weights = self.model.get_weights()
                    for i, param in enumerate(parameters):
                        weights[i].assign(param)
                
                elif hasattr(self.model, 'state_dict'):
                    state_dict = self.model.state_dict()
                    param_keys = list(state_dict.keys())
                    for i, param in enumerate(parameters):
                        if i < len(param_keys):
                            state_dict[param_keys[i]] = torch.tensor(param)
                    self.model.load_state_dict(state_dict)
                else:
                    pass                  

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
                self.set_parameters(parameters)
                ################USER INPUT############################
                optimizer = tf.optimizers.Adam(0.001)
                
                for epoch in range(self.epochs):
                    with tf.GradientTape() as tape:
                        logits = self.model(tf.cast(self.X_train, tf.float32))
                        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                            labels=self.y_train, logits=logits
                        )
                        loss = tf.reduce_mean(loss)
                    
                    gradients = tape.gradient(loss, self.model.get_weights())
                    optimizer.apply_gradients(zip(gradients, self.model.get_weights()))
                
                predictions = tf.argmax(logits, axis=1)
                correct_predictions = tf.equal(predictions, tf.cast(self.y_train, tf.int64))
                accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32)).numpy()
                
                train_metrics = {
                    "train_accuracy": float(accuracy),
                    "train_loss": float(loss.numpy())
                }

                
                updated_params = self.get_parameters(config)
                return updated_params, len(self.X_train), train_metrics

            def evaluate(self, parameters, config):
                eval_metrics = None
                self.set_parameters(parameters)  
                ################USER INPUT############################

                logits = self.model(tf.cast(self.X_test, tf.float32))
                predictions = tf.argmax(logits, axis=1).numpy()
                loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=self.y_test, logits=logits
                )).numpy()
                predictions = tf.argmax(logits, axis=1)
                correct_predictions = tf.equal(predictions, tf.cast(self.y_test, tf.int64))
                accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32)).numpy()
                
                ######################################################
                
                eval_metrics = {
                    "eval_accuracy": float(accuracy),
                    "eval_loss": float(loss)
                }
                
                
                return float(loss), len(self.X_test), eval_metrics
                  
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
        time.sleep(2)
        fl.client.start_client(
            server_address=SERVER_ADDRESS,
            client=client.to_client()
        )
        """,readonly=True,tags=["skip-execution"]),

        code_cell("""
        %%capture
        try:
            def main():
                fl.client.start_client(
                    server_address=SERVER_ADDRESS,
                    client=client.to_client()
                )

            if __name__ == "__main__":
                main()
        except:
            pass
        """, readonly=True,tags=["readonly"]),


        code_cell("""
        oasees_sdk.convert()
        """,readonly=True,tags=["skip-execution"]),


    ])

    training_notebook_path = folder_path / f"{name}_client.ipynb"

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
        import argparse
        import os
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  
        """, readonly=True,tags=["readonly"]),

        code_cell("""
        import tensorflow as tf
        """),


        code_cell("""
        def model_definition():
            model = None
            class SimpleModel:
                def __init__(self):
                    self.w1 = tf.Variable(tf.random.normal([4, 64]), trainable=True)
                    self.b1 = tf.Variable(tf.zeros([64]), trainable=True)
                    self.w2 = tf.Variable(tf.random.normal([64, 32]), trainable=True)
                    self.b2 = tf.Variable(tf.zeros([32]), trainable=True)
                    self.w3 = tf.Variable(tf.random.normal([32, 3]), trainable=True)
                    self.b3 = tf.Variable(tf.zeros([3]), trainable=True)
                    
                def __call__(self, x):
                    x = tf.nn.relu(tf.matmul(x, self.w1) + self.b1)
                    x = tf.nn.relu(tf.matmul(x, self.w2) + self.b2)
                    return tf.matmul(x, self.w3) + self.b3
                    
                def get_weights(self):
                    return [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3]
            
            model = SimpleModel()
            return model
        """),

        code_cell("""
        def load_fl_model(pkl_path):
            with open(pkl_path, 'rb') as f:
                model_data = pickle.load(f)
            
            model = model_definition()
            parameters = model_data['parameters']


            model_weights = model.get_weights()
            for weight, param in zip(model_weights, parameters):
                weight.assign(param)

            return model
        """),

        code_cell("""
        %%capture
        try:
            parser = argparse.ArgumentParser(description='Flask ML Model Service')
            parser.add_argument('--model-path', required=True, help='Path to the pickled model file')
            args = parser.parse_args()
        except:
            pass
        """, readonly=True,tags=["readonly"]),


        code_cell("""
        model = load_fl_model('')
        """,tags=["skip-execution"]),

        code_cell("""
        %%capture
        try:
            model = load_fl_model(args.model_path)
        except:
            pass
        """, readonly=True,tags=["readonly"]),



        code_cell("""
        import threading
        app = Flask(__name__)
                  
        @app.route('/status', methods=['GET'])
        def status():
            return jsonify({'status': 'up'})


        @app.route('/predict', methods=['POST'])
        def predict():
            try:
                json_data = request.get_json()
                data = np.array(json_data['data'])
                
                if data.ndim == 1:
                    data = data.reshape(1, -1)
                
                input_tensor = tf.constant(data, dtype=tf.float32)
                predictions = model(input_tensor)


                logits = predictions.numpy()
                probabilities = tf.nn.softmax(logits, axis=1)
                predicted_classes = tf.argmax(probabilities, axis=1).numpy()
                confidence = tf.reduce_max(probabilities, axis=1).numpy()


                results = {
                    'predictions': predicted_classes.tolist(),
                    'probabilities': probabilities.numpy().tolist()
                }
                
                return jsonify(results)
                
            except Exception as e:
                return jsonify({'error': str(e)}), 400
                """),

        code_cell("""
        def run_server():
            app.run(debug=False, host='0.0.0.0', port=5005, use_reloader=False)
        """,readonly=False,tags=["readonly"]),

        code_cell("""
        threading.Thread(target=run_server, daemon=True).start()
        print("Server running at http://localhost:5005")
        """,readonly=True,tags=["readonly","skip-execution"]),

        code_cell("""
        import requests
        import numpy as np
                  
        data = np.load('')[0]
        response = requests.post('http://localhost:5005/predict', 
                        json={'data': data.tolist()})
        print(response.json())          
        """,readonly=False,tags=["skip-execution"]),


        code_cell("""
        %%capture
        try:
            run_server()
        except:
            pass
        """,readonly=True,tags=["readonly"]),


        code_cell("""
        oasees_sdk.convert(deploy=True)
        """,readonly=True,tags=["skip-execution"]),



    ])    
  
    deployment_notebook_path = folder_path / f"{name}_deploy.ipynb"

    with open(deployment_notebook_path, 'w', encoding='utf-8') as f:
        json.dump(deployment_notebook, f, indent=2)


    
    click.echo(f"Created project '{name}' with OASEES notebook")








