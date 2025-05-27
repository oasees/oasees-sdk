import shutil
import subprocess
import time
import click
import getpass
import json
import requests
# import web3
import socket
import os
# from dotenv import load_dotenv
import yaml
# from kubernetes import client,config
from .oasees_ml_ops import sdk_manager_manifest
from .ipfs_client import ipfs_get,ipfs_upload
import platform
from pathlib import Path
import sys
import re


@click.group(invoke_without_command=True)
@click.pass_context
def cli(ctx):
    if not ctx.invoked_subcommand:
        bash_script = r'''
        Y='\x1b[38;2;252;220;132m'
        G='\x1b[38;2;60;148;140m'
        O='\x1b[38;2;244;172;92m'
        N='\033[0m' # No Color
        B='\x1b[38;2;4;188;252m'

        echo -e "                ${Y}.......           ${O}=======${N}                "             
        echo -e "               ${Y}.........        ${O}==========${N}               "            
        echo -e "               ${Y}.......... ${O}================${N}               "             
        echo -e "               ${Y}.........${O}==================${N}               "             
        echo -e "                ${Y}....... ${O}========= ======${N}                 "               
        echo -e "             ${Y}.......     ${O}=======     ${Y}.......${N}             "           
        echo -e "     ${O}=====  ${Y}.........${O}=========${Y}...............  .....${N}     "
        echo -e "   ${O}=========${Y}.........${O}======   ${Y}........................${N}   "
        echo -e "   ${O}==========${Y}.......  ${O}===== ${Y}........ .................   "
        echo -e "   ${O}==========  ${Y}...  ${O}====   ${Y}....  ${O}==== ===   ${Y}..........   "
        echo -e "   ${O} ========  ${Y}......${O}====  ${B}${G}**${B}  ${G}**${N} ${O}=========   ${Y}........    "
        echo -e "   ${O}   ======= ${Y}......     ${B}${G}**${B}****${G}**${N}    ${O}===== =====${Y}...      "   
        echo -e "   ${O}    ======== ${Y}.. ..  ${G}**${B}*******${G}**${N}   ${Y}..${O}== ========       "    
        echo -e "   ${O}    ========   ${Y}....${G}**${B}*********${G}**${Y}..... ${O}=========       "    
        echo -e "   ${O}    ============${Y}..  ${G}**${B}*******${G}**${N}   ${Y}.....${O}========       "    
        echo -e "   ${O}   ${Y}...${O}==== ======     ${G}**${B}****${G}**${O}====${Y}......${O}=========${N}     "  
        echo -e "   ${Y}.........  ${O}========== ${G} **  ** ${N} ${O}====${Y}......  ${O}=========   "
        echo -e "   ${Y}..........   .${O}=   ===  ${Y}....  ${O}====   ${Y}...  ${O}==========   "
        echo -e "   ${Y}................. ......... ${O}======${Y}........${O}=========   "
        echo -e "   ${Y}........................    ${O}=====${Y}.........${O}========    "
        echo -e "     ${Y}.....  ..............${O}========   ${Y}........${O}   ===      "   
        echo -e "             ${Y}.......     ${O}=========    ${Y}......             "
        echo -e "                   ${O}===============${Y}.......                "
        echo -e "                  ${O}===============${Y}.........               "
        echo -e "                 ${O}==========     ${Y}..........               " 
        echo -e "                 ${O}==========      ${Y}.........               "
        echo -e "                  ${O}========        ${Y}.......                "
                                                
        echo -e ${B}"  ___    _    ____  _____ _____ ____    ____  ____  _  __ "
        echo -e ${B}" / _ \  / \  / ___|| ____| ____/ ___|  / ___||  _ \| |/ / "
        echo -e ${B}"| | | |/ _ \ \___ \|  _| |  _| \___ \  \___ \| | | | ' /  "
        echo -e ${B}"| |_| / ___ \ ___) | |___| |___ ___) |  ___) | |_| | . \  "
        echo -e ${B}" \___/_/   \_\____/|_____|_____|____/  |____/|____/|_|\_\ "${N}
        '''

        # Execute the Bash script and capture its output
        result = subprocess.run(['bash', '-c', bash_script], capture_output=True, text=True)
        check_required_tools()
        click.echo(result.stdout)
        cli(['--help'])


# @click.option('--price', required=False, type=float, default=0, help="")

def ipfs_client_install():
   '''Installs IPFS CLI on the user's path'''
   
   # Detect system architecture
   machine = platform.machine().lower()
   
   # Determine download URL based on architecture
   if machine in ('x86_64', 'amd64'):
       url = "https://dist.ipfs.tech/kubo/v0.30.0/kubo_v0.30.0_linux-amd64.tar.gz"
       filename = "kubo_v0.30.0_linux-amd64.tar.gz"
   elif machine in ('arm64', 'aarch64'):
       url = "https://dist.ipfs.tech/kubo/v0.30.0/kubo_v0.30.0_linux-arm64.tar.gz"
       filename = "kubo_v0.30.0_linux-arm64.tar.gz"
   else:
       print(f"Unsupported architecture: {machine}")
       sys.exit(1)
   
   print(f"Detected architecture: {machine}")
   print(f"Downloading IPFS Kubo from: {url}")
   
   try:
       # Download IPFS Kubo
       subprocess.run(["curl", "-L", "-o", filename, url], check=True)
       
       # Extract the tar.gz file
       subprocess.run(["tar", "-xzf", filename], check=True)
       
       # Make ipfs executable (should already be, but just in case)
       subprocess.run(["chmod", "+x", "kubo/ipfs"], check=True)
       
       # Move to /usr/local/bin (requires sudo)
       try:
            subprocess.run(["sudo", "cp", "kubo/ipfs", "/usr/local/bin/"], check=True)
       except:
            subprocess.run(["cp", "kubo/ipfs", "/usr/local/bin/"], check=True)

       
       # Clean up downloaded files
       subprocess.run(["rm", "-rf", filename, "kubo"], check=True)
       
       click.secho("IPFS CLI installed successfully", fg="green")
       return True
   except subprocess.CalledProcessError as e:
       click.secho(f"Error during installation: {e}", fg="red")
       sys.exit(1)



def kompose_install():
    '''Installs kompose on the user's path'''

    # Detect system architecture
    machine = platform.machine().lower()
    
    # Determine download URL based on architecture
    if machine in ('x86_64', 'amd64'):
        url = "https://github.com/kubernetes/kompose/releases/download/v1.35.0/kompose-linux-amd64"
    elif machine in ('arm64', 'aarch64'):
        url = "https://github.com/kubernetes/kompose/releases/download/v1.35.0/kompose-linux-arm64"
    else:
        print(f"Unsupported architecture: {machine}")
        sys.exit(1)
    
    print(f"Detected architecture: {machine}")
    print(f"Downloading Kompose from: {url}")
    
    try:
        # Download Kompose
        subprocess.run(["curl", "-L", "-o", "kompose", url], check=True)
        
        # Make it executable
        subprocess.run(["chmod", "+x", "kompose"], check=True)
        
        # Move to /usr/local/bin (requires sudo)
        try:
            subprocess.run(["sudo", "mv", "kompose", "/usr/local/bin/"], check=True)
        except:
            subprocess.run(["mv", "kompose", "/usr/local/bin/"], check=True)


        click.secho("Kompose installed successfully", fg="green")
        return True
    except subprocess.CalledProcessError as e:
        click.secho(f"Error during installation: {e}",fg="red")
        sys.exit(1)
def helm_install():
    '''Installs helm on the user's path'''

    try:
        print("Installing Helm...")
        subprocess.run([
            "curl", "-fsSL", "-o", "get_helm.sh",
            "https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3"
        ], check=True)
        subprocess.run(["chmod", "700", "get_helm.sh"], check=True)
        subprocess.run(["./get_helm.sh"], check=True)
        Path("get_helm.sh").unlink()  # Cleanup
        
        click.secho("Helm installed successfully", fg="green")
        return True
    except subprocess.CalledProcessError as e:
        click.secho(f"Failed to install Helm: {e}", fg="red")
        sys.exit(1)

def helm_add_oasees_repo():
    '''Adds and / or updates the repository containing all of OASEES' charts'''

    add_cmd = ["repo", "add", "oasees-charts", "https://oasees.github.io/helm-charts"]
    update_cmd = ["repo", "update", "oasees-charts"]
    
    click.echo("*Adding OASEES helm chart repository...*")
    click.echo("------------------------------------")
    run_helm_command(add_cmd)

    click.echo('\n')
    click.echo("*Updating OASEES chart repository...*")
    click.echo("------------------------------------")
    run_helm_command(update_cmd)


def helm_install_oasees_user(expose_ip):
    cmd = ["upgrade", "--install", "oasees-user", "oasees-charts/oasees-user", "--create-namespace", "--set", f"exposedIP={expose_ip}"]
    run_helm_command(cmd)


def run_helm_command(cmd:list):
    
    try:
        result = subprocess.run(
            ["helm"] + cmd,
            check=True,
            text=True
        )

    except subprocess.CalledProcessError as e:
        return e.stderr


def check_required_tools():
    """Check if helm and kompose are installed and available in PATH."""
    helm_installed = shutil.which("helm") is not None
    kompose_installed = shutil.which("kompose") is not None
    ipfs_client_installed = shutil.which("ipfs") is not None
    
    # click.echo("Checking if the required tools are installed on your platform:")
    # click.echo(f"Helm: {'✅ installed' if helm_installed else '❌ not found'}")
    # click.echo(f"Kompose: {'✅ installed' if kompose_installed else '❌ not found'}")
    # click.echo(f"Ipfs Client: {'✅ installed' if ipfs_client_installed else '❌ not found'}")

    if helm_installed and kompose_installed and ipfs_client_installed:
        return True
    else:
        if click.confirm("\nHelm and Kompose are needed in order to use all of the OASEES SDK features. Would you like to install them now?"):
            helm_install()
            kompose_install()
            ipfs_client_install()
            return True
        else:
            return False

def get_nodes():
    '''Retrieves information on the cluster's nodes.'''
    try:
        result = subprocess.run(["kubectl", "get", "nodes", "-o", "json"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return result.stdout
    except FileNotFoundError:
        click.echo("Error: No connection to cluster found.")

@cli.command()
@click.option('--expose-ip', required=False, help="Expose a node's IP if you plan accessing the stack remotely.")
@click.option('--update', required=False, is_flag=True, help="Use this to skip cluster initialization.")
def init(expose_ip,update):
    '''Sets up the OASEES cluster.'''

    if(check_required_tools()):
        if not update:
            try:
                curl = subprocess.Popen(['curl','-sfL', 'https://get.k3s.io'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                result = subprocess.check_output(['sh','-s','-','--write-kubeconfig-mode','644', '--write-kubeconfig', '/home/'+getpass.getuser()+'/.kube/config', '--node-label', 'user='+getpass.getuser()], stdin=curl.stdout)
                click.echo(result)
                curl.wait()

            except subprocess.CalledProcessError as e:
                return e.stderr
        
        try:
            helm_add_oasees_repo()
            click.echo('\n')
            helm_install_oasees_user(expose_ip)
            click.echo('\n')
        except Exception as e:
            click.secho("Failed to install OASEES user chart.", fg="red")
            sys.exit(1)

        click.secho("Kubernetes cluster initialized successfully!",fg="green")
    else:
        click.secho("Please install Helm and / or Kompose before trying again.", fg="red")

@cli.command()
@click.argument('role', type=str)
def uninstall(role):
    '''Runs the appropriate k3s uninstallation script based on the role provided.'''
    try:
        if(role=='master'):
            result = subprocess.run(['/usr/local/bin/k3s-uninstall.sh'],  stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            click.echo("Cluster uninstalled successfully.")
        elif (role=='agent'):
            result = subprocess.run(['/usr/local/bin/k3s-agent-uninstall.sh'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            click.echo("Agent uninstalled successfully.")
        else:
            click.echo("Enter a valid role (master / agent)")
    except FileNotFoundError:
        click.echo("Error: Uninstall executable not found.")


@cli.command()
def get_token():
    '''Retrieves the token required to join the cluster.'''

    try:
        token = subprocess.run(['sudo', 'cat','/var/lib/rancher/k3s/server/token'],  stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        click.echo(token.stdout)
    except FileNotFoundError:
        click.echo("Error: No token found.")


@cli.command()
@click.option('--ip', required=True, help="The cluster's master ip address.")
@click.option('--token', required=True, help="The cluster's master token.")
@click.option('--iface', required=False, help="The network interface you want to join with (tun0 if you're using a VPN connection).")
def join(ip,token,iface):
    '''Joins the current machine to the specified cluster, using the specified interface if one is provided.'''
    try:
        curl = subprocess.Popen(['curl','-sfL', 'https://get.k3s.io'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        if(iface):
            result = subprocess.check_output(['sh','-s','-','--flannel-iface', iface, '--node-label', 'user='+getpass.getuser()], env={'K3S_URL' : 'https://'+ip+':6443', 'K3S_TOKEN': token}, stdin=curl.stdout)
        else:
            result = subprocess.check_output(['sh','-s','-','--node-label', 'user='+getpass.getuser()], env={'K3S_URL' : 'https://'+ip+':6443', 'K3S_TOKEN': token}, stdin=curl.stdout)
        click.echo(result)
        curl.wait()
    except FileNotFoundError:
        click.echo("Error: K3S cluster could not be joined.\n")


@cli.command()
@click.argument('compose_file', type=str)
@click.argument('output_dir', type=str)
def convert_app(compose_file, output_dir):
    """Convert Docker Compose to Kubernetes manifests using kompose"""
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    
    print("Converting Docker Compose using Kompose...")
    subprocess.run(["kompose", "convert", "-f", compose_file, "-o", output_dir], check=True)

    process_yaml_files(output_dir)

def process_yaml_files(output_dir):
    """Process generated YAML files to add nodeSelectors and other modifications"""
    yaml_files = list(Path(output_dir).glob("*.yaml")) + list(Path(output_dir).glob("*.yml"))
    
    for yaml_file in yaml_files:
        print(f"Processing {yaml_file}...")
        
        with open(yaml_file, 'r') as f:
            data = list(yaml.safe_load_all(f))
        
        modified = False
        
        for doc in data:
            if not doc:
                continue
                
            kind = doc.get('kind', '')
            
            # Process Deployments
            if kind == 'Deployment':
                metadata = doc.get('metadata', {})
                annotations = metadata.get('annotations', {})
                
                # Handle nodeSelector
                node_selector_value = annotations.get('oasees.device')
                if node_selector_value:
                    if 'spec' not in doc:
                        doc['spec'] = {}
                    if 'template' not in doc['spec']:
                        doc['spec']['template'] = {}
                    if 'spec' not in doc['spec']['template']:
                        doc['spec']['template']['spec'] = {}
                    
                    doc['spec']['template']['spec']['nodeSelector'] = {
                        'kubernetes.io/hostname': node_selector_value
                    }
                    modified = True
                
                # Handle sensor resources
                sensor_value = annotations.get('oasees.sensor')
                if sensor_value:
                    if 'spec' not in doc:
                        doc['spec'] = {}
                    if 'template' not in doc['spec']:
                        doc['spec']['template'] = {}
                    if 'spec' not in doc['spec']['template']:
                        doc['spec']['template']['spec'] = {}
                    if 'containers' not in doc['spec']['template']['spec']:
                        continue
                    
                    resource_map = {
                        'sound': {'smarter-devices/snd': 1},
                        'video': {'smarter-devices/video0': 1},
                        'bluetooth': {'smarter-devices/hci0': 1}
                    }
                    
                    if sensor_value in resource_map:
                        doc['spec']['template']['spec']['containers'][0]['resources'] = {
                            'limits': resource_map[sensor_value]
                        }
                        modified = True
            
            # Process Services
            elif kind == 'Service':
                expose_value = doc.get('metadata', {}).get('annotations', {}).get('oasees.expose')
                if expose_value:
                    doc['spec']['type'] = 'NodePort'
                    doc['spec']['ports'][0]['nodePort'] = int(expose_value)
                    modified = True
        
        if modified:
            with open(yaml_file, 'w') as f:
                yaml.dump_all(data, f, default_flow_style=False)
@cli.command()
@click.argument('folder', type=str)
def deploy_app(folder):
    """Deploy all YAML/JSON manifests in a folder using kubectl"""
    files = list(Path(folder).glob('*.yaml')) + list(Path(folder).glob('*.yml'))
    
    if not files:
        print(f"No manifests found in {folder}")
        return False

    success = True
    for f in files:
        try:
            print(f"Deploying {f.name}...")
            subprocess.run(['kubectl', 'apply', '-f', str(f)], check=True)
        except subprocess.CalledProcessError:
            print(f"Failed to deploy {f.name}")
            success = False
    
    if success:
        click.secho("All manifests deployed successfully!", fg="green")
    else:
        click.secho("Some manifests failed to deploy.", fg="red")

@cli.command()
@click.argument('folder', type=str)
def delete_app(folder):
    """Delete all YAML/JSON manifests in a folder using kubectl"""
    files = list(Path(folder).glob('*.yaml')) + list(Path(folder).glob('*.yml'))
    
    if not files:
        print(f"No manifests found in {folder}")
        return False

    success = True
    for f in files:
        try:
            print(f"Deploying {f.name}...")
            subprocess.run(['kubectl', 'delete', '-f', str(f)], check=True)
        except subprocess.CalledProcessError:
            print(f"Failed to deploy {f.name}")
            success = False
    
    if success:
        click.secho("All manifests deleted successfully!", fg="green")
    else:
        click.secho("Some manifests failed to delete.", fg="red")

@cli.command()
def enable_gpu_operator():
    '''Enables the GPU Operator on the cluster'''

    click.secho("Enabling GPU Operator...", fg="yellow")

    command = [
        "helm", "install", "gpu-operator", "-n", "gpu-operator", "--create-namespace", "nvidia/gpu-operator",
        "--version=v25.3.0",
        "--set", "toolkit.env[0].name=CONTAINERD_CONFIG",
        "--set", "toolkit.env[0].value=/var/lib/rancher/k3s/agent/etc/containerd/config.toml",
        "--set", "toolkit.env[1].name=CONTAINERD_SOCKET",
        "--set", "toolkit.env[1].value=/run/k3s/containerd/containerd.sock",
        "--set", "toolkit.env[2].name=CONTAINERD_RUNTIME_CLASS",
        "--set", "toolkit.env[2].value=nvidia",
        "--set", "toolkit.env[3].name=CONTAINERD_SET_AS_DEFAULT",
        "--set-string", "toolkit.env[3].value=true"
    ]

    try:
        result = subprocess.run(command, check=True, text=True)
        click.secho("GPU Operator enabled successfully!", fg="green")
    except subprocess.CalledProcessError as e:
        click.secho(f"Failed to enable GPU Operator: {e}", fg="red")
        sys.exit(1)

@cli.command()
def mlops_add_node():
    '''Install the components needed to execute ML workloads on the specified node.'''
    node_name = ''
    mount_path = ''
    gpu_enabled = 0
    node_infos = json.loads(get_nodes())
    nodes = []

    click.echo("Available nodes:")
    for idx, node_info in enumerate(node_infos['items']):
        name = node_info['metadata']['name']
        gpu = any(re.match(r'nvidia.*',label) for label in node_info['metadata']['labels'])

        click.echo(f"{idx+1}. {name} (GPU: {'Yes' if gpu else 'No'})")
        nodes.append({'name': name, 'gpu': gpu})

    click.echo('\n')

    selection = int(click.prompt("Choose the node you want to use for training", type=click.Choice([str(i+1) for i in range(len(nodes))]), show_choices=False)) - 1
    node_name = nodes[selection]['name']

    if(nodes[selection]['gpu']):
        gpu_enabled = 1 if click.confirm("Would you like to enable GPU support for this node?", default=True) else 0

    click.echo()
    click.secho(f'The current MLOPs image requires the designated node ({node_name}) to have a venv with all the required ML libraries already set up.', fg="yellow")
    mount_path = click.prompt("Enter the path your venv is in (e.g. /home/<username>/venv)", type=str)

    command = [
        "helm", "upgrade" , "--install", f"training-worker-{node_name}", "oasees-charts/training-worker", "--set", f"nodeName={node_name}", "--set", f"mountPath={mount_path}", "--set", f"gpu={gpu_enabled}"
    ]

    try:
        subprocess.run(command, check=True, text=True)
        click.secho(f"Training components deployed successfully on {node_name}!", fg="green")
    except subprocess.CalledProcessError as e:
        click.secho(f"Error during the deployment of the training components.", fg="red")
        sys.exit(1)


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
        # Folder doesn't exist, create it
        # if not quiet:
            # click.echo("Creating /oasees-ml-ops folder...")
        
        mkdir_cmd = ['ipfs', f'--api={api_endpoint}', 'files', 'mkdir', '/oasees-ml-ops']
        mkdir_result = subprocess.run(mkdir_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        if mkdir_result.returncode != 0:
            # click.secho(f"Failed to create /oasees-ml-ops folder: {mkdir_result.stderr}", fg="red", err=True)
            return False
        
        # if not quiet:
        #     click.secho("Created /oasees-ml-ops folder", fg="green")
    
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

@cli.command()
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

@cli.command()
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

@cli.command()
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

@cli.command()
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

@cli.command()
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

@cli.command()
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
                
                # Show gateway URL
                ipfs_ip = api_endpoint.split('/')[2]
                # click.echo(f"Gateway URL: http://{ipfs_ip}:8080/ipfs/{file_hash}")
            else:
                click.echo(file_hash)
        else:
            click.secho(f"Failed to copy to MFS: {cp_result.stderr}", fg="red", err=True)
            if not quiet:
                click.echo(f"File is available in IPFS with hash: {file_hash}")
            
    except Exception as e:
        click.secho(f"Unexpected error: {str(e)}", fg="red", err=True)
