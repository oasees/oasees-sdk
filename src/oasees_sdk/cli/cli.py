import shutil
import subprocess
import time
import click
import getpass
import json
import requests
import socket
# import web3
import socket
import os
# from dotenv import load_dotenv
import yaml
# from kubernetes import client,config
# from .oasees_ml_ops import sdk_manager_manifest
# from .ipfs_client import ipfs_get,ipfs_upload
import platform
from pathlib import Path
import sys
import re
from .commands.mlops import mlops_commands
from .commands.telemetry import telemetry_commands
from .commands.display_banner import display_banner


def get_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.settimeout(0)
    try:
        # doesn't even have to be reachable
        s.connect(('10.254.254.254', 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP






@click.group(invoke_without_command=True)
@click.pass_context
def cli(ctx):
    if not ctx.invoked_subcommand:
        display_banner()
        check_required_tools()
        cli(['--help'])


cli.add_command(mlops_commands)
cli.add_command(telemetry_commands)
# cli.add_command(display_banner)

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


def helm_install_oasees(expose_ip):
    cmd = ["upgrade", "--install", "oasees-user", "oasees-charts/oasees-user", "--create-namespace", "--set", f"exposedIP={expose_ip}"]
    run_helm_command(cmd)
    cmd = ["upgrade", "--install", "oasees-telemetry", "oasees-charts/oasees-telemetry", "--create-namespace"]
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
# def init(update):    
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
        

        # expose_ip = get_ip()


        try:
            helm_add_oasees_repo()
            click.echo('\n')
            helm_install_oasees(expose_ip)
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
                

                oasees_ui_label = annotations.get('oasees.ui')
                if oasees_ui_label:
                    metadata['labels']['oasees-ui'] = 'true'

                
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
                        'sound': {'oasees.dev/snd': 1},
                        'video': {'oasees.dev/video0': 1},
                        'bluetooth': {'oasees.dev/hci0': 1}
                    }
                    
                    if sensor_value in resource_map:
                        doc['spec']['template']['spec']['containers'][0]['resources'] = {
                            'limits': resource_map[sensor_value]
                        }
                        modified = True
            
            elif kind == 'Service':
                metadata = doc.get('metadata', {})
                annotations = doc.get('metadata', {}).get('annotations', {})
                oasees_ui_label = annotations.get('oasees.ui')
                if oasees_ui_label:
                    metadata['labels']['oasees-ui'] = 'true'
                    oasees_ui_port = annotations.get('oasees.ui.port')
                    metadata['labels']['oasees-ui-port'] = oasees_ui_port
                
                expose_values = []
                for key, value in annotations.items():
                    if key.startswith('oasees.expose'):
                        if isinstance(value, str):
                            ports = [port.strip() for port in value.split() if port.strip()]
                            expose_values.extend(ports)
                        else:
                            expose_values.append(str(value))
                
                if expose_values:
                    doc['spec']['type'] = 'NodePort'
                    
                    if 'ports' not in doc['spec']:
                        doc['spec']['ports'] = []
                    
                    existing_ports = doc['spec']['ports']
                    
                    for i, expose_value in enumerate(expose_values):
                        try:
                            node_port = int(expose_value)
                            
                            if i < len(existing_ports):
                                existing_ports[i]['nodePort'] = node_port
                            else:
                                print(f"Warning: More expose values ({len(expose_values)}) than existing ports ({len(existing_ports)}) for service")
                                break
                        
                        except ValueError:
                            print(f"Warning: Invalid port value '{expose_value}' in oasees.expose annotation")
                            continue
                    
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

# @cli.command()
# def enable_gpu_operator():
#     '''Enables the GPU Operator on the cluster'''

#     click.secho("Enabling GPU Operator...", fg="yellow")

#     command = [
#         "helm", "install", "gpu-operator", "-n", "gpu-operator", "--create-namespace", "nvidia/gpu-operator",
#         "--version=v25.3.0",
#         "--set", "toolkit.env[0].name=CONTAINERD_CONFIG",
#         "--set", "toolkit.env[0].value=/var/lib/rancher/k3s/agent/etc/containerd/config.toml",
#         "--set", "toolkit.env[1].name=CONTAINERD_SOCKET",
#         "--set", "toolkit.env[1].value=/run/k3s/containerd/containerd.sock",
#         "--set", "toolkit.env[2].name=CONTAINERD_RUNTIME_CLASS",
#         "--set", "toolkit.env[2].value=nvidia",
#         "--set", "toolkit.env[3].name=CONTAINERD_SET_AS_DEFAULT",
#         "--set-string", "toolkit.env[3].value=true"
#     ]

#     try:
#         result = subprocess.run(command, check=True, text=True)
#         click.secho("GPU Operator enabled successfully!", fg="green")
#     except subprocess.CalledProcessError as e:
#         click.secho(f"Failed to enable GPU Operator: {e}", fg="red")
#         sys.exit(1)

# @cli.command()
# def mlops_add_node():
#     '''Install the components needed to execute ML workloads on the specified node.'''
#     node_name = ''
#     mount_path = ''
#     gpu_enabled = 0
#     node_infos = json.loads(get_nodes())
#     nodes = []

#     click.echo("Available nodes:")
#     for idx, node_info in enumerate(node_infos['items']):
#         name = node_info['metadata']['name']
#         gpu = any(re.match(r'nvidia.*',label) for label in node_info['metadata']['labels'])

#         click.echo(f"{idx+1}. {name} (GPU: {'Yes' if gpu else 'No'})")
#         nodes.append({'name': name, 'gpu': gpu})

#     click.echo('\n')

#     selection = int(click.prompt("Choose the node you want to use for training", type=click.Choice([str(i+1) for i in range(len(nodes))]), show_choices=False)) - 1
#     node_name = nodes[selection]['name']

#     if(nodes[selection]['gpu']):
#         gpu_enabled = 1 if click.confirm("Would you like to enable GPU support for this node?", default=True) else 0

#     click.echo()
#     click.secho(f'The current MLOPs image requires the designated node ({node_name}) to have a venv with all the required ML libraries already set up.', fg="yellow")
#     mount_path = click.prompt("Enter the path your venv is in (e.g. /home/<username>/venv)", type=str)

#     command = [
#         "helm", "upgrade" , "--install", f"training-worker-{node_name}", "oasees-charts/training-worker", "--set", f"nodeName={node_name}", "--set", f"mountPath={mount_path}", "--set", f"gpu={gpu_enabled}"
#     ]

#     try:
#         subprocess.run(command, check=True, text=True)
#         click.secho(f"Training components deployed successfully on {node_name}!", fg="green")
#     except subprocess.CalledProcessError as e:
#         click.secho(f"Error during the deployment of the training components.", fg="red")
#         sys.exit(1)

