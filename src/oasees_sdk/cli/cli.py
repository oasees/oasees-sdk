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
        click.echo(result.stdout)
        cli(['--help'])


# @click.option('--price', required=False, type=float, default=0, help="")
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
        
        print("Helm installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"Failed to install Helm: {e}")
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


def helm_install_oasees_user():
    cmd = ["upgrade", "--install", "oasees-user", "oasees-charts/oasees-user", "--create-namespace"]
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

@cli.command()
def init_cluster():

    try:
        curl = subprocess.Popen(['curl','-sfL', 'https://get.k3s.io'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        result = subprocess.check_output(['sh','-s','-','--write-kubeconfig-mode','644', '--write-kubeconfig', '/home/'+getpass.getuser()+'/.kube/config', '--node-label', 'user='+getpass.getuser()], stdin=curl.stdout)
        click.echo(result)
        curl.wait()

    except subprocess.CalledProcessError as e:
        return e.stderr
    
    helm_install()
    click.echo('\n')
    helm_add_oasees_repo()
    click.echo('\n')
    helm_install_oasees_user()


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
def join_cluster(ip,token,iface):
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