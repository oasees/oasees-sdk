# Oasees SDK

The OASEES SDK is a Python package that consists of two modules: the **Command Line Interface (CLI)** module and the **Python Environment** module.

<br>

## CLI

The CLI module's purpose is to handle the Kubernetes aspect of the OASEES framework. It provides the user with a few simple commands to quickly provision and configure a Kubernetes cluster, as well as facilitate and automate its nodes' connection to the OASEES blockchain
<br>

### <ins>Installation</ins>
Like any other user Python package published on the PyPi repository, the user can install the OASEES SDK using pip.

<ol>
<li>

Ensure that pip is present on your machine. If not, install it:
    
    sudo apt install python3-pip

</li>

<br>

<li>

Install with pip:

    pip install git+https://github.com/oasees/oasees-sdk.git
  

</li>

<br>

<li>

Set up the CLI module for use by adding the executable's installation folder to your system's PATH (example **for Ubuntu**):
    
    export PATH="/home/{USERNAME}/.local/bin:$PATH"

</li>

<br>

<li>
Make sure that it is working properly by executing:

    oasees-sdk

Which will also provide you with the CLI's available commands and a very short description for each one.


</li>

</ol>

<br><br>

### <ins>Usage</ins>

#### <ins>Important note:</ins> Before using the CLI module, ensure that an instance of the OASEES stack is up and running.

The SDK is designed to work in parallel with the rest of the stack. Before provisioning a cluster, <ins>the user will be prompted to enter the stack's IP address and their blockchain account information</ins> (if you're using a test account, make sure it's the same one that you've imported on MetaMask).

Should any typing mistake happen during the input prompts, you can edit the configuration file located in `/home/{username}/.oasees_sdk/config.json`, or use `oasees-sdk config-full-reset` to completely reset the configuration and force the prompts to reappear.

<br>

***

<br>

#### <ins>Available commands</ins>

As mentioned above, executing `oasees-sdk` in your terminal will give you a list of the available commands. To get the full description of one of these commands, simply execute:

    oasees-sdk [COMMAND] --help

<br>

***

<br>

#### <ins>The CLI's typical usage flow:</ins>

<ol>

<li>

Install the OASEES SDK on all the machines that will participate in the cluster (**both the master node and the worker nodes**) using the installation commands mentioned above.

</li>

<br>

<li>

Provision a K3S Cluster on the machine that represents the cluster's **master node**:

    oasees-sdk init-cluster

The cluster should be visible on your Portal's home page almost immediately.
</li>

<br>

<li>

Retrieve your cluster's token:

    oasees-sdk get-token

</li>

<br>

<li>

Execute the join-cluster command **on each of your worker devices** to join them with your cluster, providing the master's IP address and the retrieved token:

    oasees-sdk join-cluster --ip {K3S_MASTER_IP_ADDRESS} --token {K3S_MASTER_TOKEN}

**NOTE: If you're using a VPN connection, make sure to specify that connection's network interface by setting an extra flag:**

    oasees-sdk join-cluster --ip {K3S_MASTER_IP_ADDRESS} --token {K3S_MASTER_TOKEN} --iface {VPN_INTERFACE}


</li>

<br>

<li>

Execute the register-new-nodes command **on your master node** to create blockchain accounts for each of your unregistered worker devices, register them to the blockchain and associate them with your cluster:

    oasees-sdk register-new-nodes

**NOTE:** This command detects and handles only the nodes that aren't already registered on the blockchain, so you can use it multiple times as you scale your K3S Cluster with new devices / nodes.

</li>

<br>

<li>

If you intend to associate your cluster with a blockchain DAO, execute the apply-dao-logic command **on your master node**, providing the IPFS hash of your uploaded DAO contracts:

    oasees-sdk apply-dao-logic {DAO_CONTRACTS_IPFS_HASH}

After the DAO logic is applied, devices that are already registered on the blockchain, as well as devices that get registered at a later point, will automatically be able to perform DAO actions such as creating proposals and voting.

</li>

</ol>

<br>

### [Click here for a video demo of the above steps.](https://nocncsrd.sharepoint.com/:v:/r/sites/OASEES2/Shared%20Documents/WP4/Meetings/OASEES%20stack%20%26%20sdk%20new%20installation%20guide/cli_demo_2.mp4?csf=1&web=1&e=gkYTbE)


<br>

***

<br>

#### <ins>Uninstalling</ins>

If you want to uninstall your cluster (e.g. you've restarted your OASEES stack) run the following **on the master node**:

    oasees-sdk uninstall master # UNINSTALLS K3S SERVER ON MASTER NODE

and **on <ins>each</ins> of your worker nodes**:

    oasees-sdk uninstall agent #UNINSTALLS K3S AGENT ON WORKER NODE

<br>

***

<br>

## Python Module SDK

The Python Environment portion of the SDK is embedded into the Jupyter Notebook image which is built upon the creation of the OASEES stack. This means that no installation of the SDK and its requirements is needed from the user's side. It only needs to be imported either into a Jupyter Notebook or Python Console by running:

    from oasees_sdk import oasees_sdk

<br/>

Or in the stack's Jupyter Terminal (all three options are found in the portal's <i>Notebook</i> tab), where you first need to launch the python shell by running:

    python3

And then import the SDK as indicated.

<br/>


### Usage / Instructions
As soon as the SDK gets successfully imported into the chosen python environment, the following brief documentation about its functions and their usage is printed:

```
╔════════════════════════════════════════════════════════════════════════════════════╗
║ OASEES SDK methods                                                                 ║
║ ------------------------------                                                     ║
║                                                                                    ║
║     my_algorithms()                                                                ║
║         Returns a list with all the algorithms purchased from your account         ║
║         on the OASEES Marketplace.                                                 ║
║                                                                                    ║
║                                                                                    ║
║     my_devices()                                                                   ║
║         Returns a list with all the devices purchased / uploaded from your account ║
║         on the OASEES Marketplace.                                                 ║
║                                                                                    ║
║                                                                                    ║
║     build_image()                                                                  ║
║         Deploys a job on the Kubernetes cluster associated with your blockchain    ║
║     account, which builds a container image out of your specified folder.          ║
║     The image will then be stored on your master node, and will be available       ║
║     for deployment on any of the cluster's nodes specified in your manifest file.  ║
║                                                                                    ║
║         - image_folder_path: Needs to be providerd in "string" form.               ║
║                                                                                    ║
║     e.g. DApp_Image_Folder -> build_image("DApp_Image_Folder")                     ║
║                                                                                    ║
║                                                                                    ║
║                                                                                    ║
║                                                                                    ║
║     deploy_manifest()                                                              ║
║         Deploys all the objects included in your specified manifest file, on the   ║
║     Kubernetes cluster associated with your blockchain account.                    ║
║                                                                                    ║
║     - manifest_file_path: Needs to be providerd in "string" form.                  ║
║                                                                                    ║
║     e.g. manifest.yaml -> build_image("manifest.yaml")                             ║
║                                                                                    ║
║                                                                                    ║
║     instructions()                                                                 ║
║         Reprints the above documentation.                                          ║
║                                                                                    ║
╚════════════════════════════════════════════════════════════════════════════════════╝
```
