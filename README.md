# Oasees SDK

Despite its name, the OASEES SDK is a Rapid Development Toolkit (RDK) developed in the context of the OASEES Project, that:

1. Facilitates the provisioning of a Kubernetes cluster that comes with the OASEES framework's core components pre-installed.
2. Provides templates and commands for the deployment of user applications in a way that enables immediate interaction with said components, as well as the OASEES Blockchain.
3. Provides templates and commands for the development and deployment of Machine Learning and Federated Learning workflows.

The SDK comes in the form of a Python Package, and its functionality is split between two core modules: a **Command Line Interface** module and a **Python module SDK**.

<br>

## CLI

The CLI module's purpose is to handle the Kubernetes aspect of the OASEES framework.
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

Then either install with pip through its PyPi release:
    
    pip install oasees-sdk

Or through its official GitHub repository:

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

### Usage

<br>

#### <ins>Available commands</ins>

As mentioned above, executing `oasees-sdk` in your terminal will give you a list of the available commands. To get the full description of one of these commands, simply execute:

    oasees-sdk [COMMAND] --help

<br>

***

<br>

### I. Stack and Node Management

Commands for initializing the OASEES stack on the master node and managing worker nodes.

* #### `oasees-sdk init`
    * **Description**: Provisions the Oasees stack on the master node.
    * **Argument**: `--expose-ip <YOUR_MASTERS_IP>` (Required): The IP address of the master node to expose the Oasees portal.
    * **Usage**:
        ```bash
        oasees-sdk init --expose-ip <YOUR MASTERS's IP>
        ```

* #### `oasees-sdk get-token`
    * **Description**: Generates a token required for joining worker-agent nodes to the master.
    * **Usage**:
        ```bash
        oasees-sdk get-token
        ```

* #### `oasees-sdk join`
    * **Description**: Joins a device to the cluster as a worker-agent node. This must be run on each device that will be a worker.
    * **Arguments**:
        * `--ip <YOUR MASTER's IP>` (Required): The IP address of the master node.
        * `--token <TOKEN>` (Required): The token generated from the `oasees-sdk get-token` command.
    * **Usage**:
        ```bash
        oasees-sdk join --ip <YOUR MASTER's IP> --token <TOKEN>
        ```

***

### II. Application Management

Commands for converting and deploying applications on the OASEES stack.

* #### `oasees-sdk convert-app`
    * **Description**: Converts a Docker Compose file with OASEES-specific labels into a format that can be deployed on the stack.
    * **Arguments**:
        * `<docker-compose-file>` (Required): The path to the OASEES-compatible `docker-compose.yaml` file.
        * `<app-name>` (Required): A name for the new application.
    * **Usage**:
        ```bash
        oasees-sdk convert-app docker-compose_oasees.yaml my-app
        ```

* #### `oasees-sdk deploy-app`
    * **Description**: Deploys a converted application to the cluster.
    * **Argument**: `<app-name>` (Required): The name of the application you want to deploy.
    * **Usage**:
        ```bash
        oasees-sdk deploy-app my-app
        ```

* #### `oasees-sdk get-app`
    * **Description**: Displays a summary of a deployed application, showing which service is running on which node and on which port.
    * **Usage**:
        ```bash
        oasees-sdk get-app
        ```

***

### III. Telemetry

Commands for managing metrics collection and agent configuration for DAO interactions. All telemetry commands can be listed by running `$ oasees-sdk telemetry`.

* #### `oasees-sdk telemetry deploy-collector`
    * **Description**: Deploys a collector to scrape metrics from a specified endpoint.
    * **Arguments**:
        * `-i <metric_index>` (Required): The name for the metric index to group metrics together[.
        * `-s <device_name>` (Required): The device where the collector will be deployed.
        * `-se <scrape_endpoint>` (Required): The URL endpoint to scrape for metrics.
    * **Usage**:
        ```bash
        oasees-sdk telemetry deploy-collector -i swarm_metrics -s device4 -se http://192.168.88.239:32691/metrics
        ```

* #### `oasees-sdk telemetry metric-index`
    * **Description**: Lists all created metric indices.
    * **Usage**:
        ```bash
        oasees-sdk telemetry metric-index
        ```

* #### `oasees-sdk telemetry metrics-list`
    * **Description**: Lists all metrics being ingested for a specific metric index.
    * **Argument**: `-i <metric_index>` (Required): The name of the metric index.
    * **Usage**:
        ```bash
        oasees-sdk telemetry metrics-list -i swarm_metrics
        ```

* #### `oasees-sdk telemetry gen-config`
    * **Description**: Generates a configuration template file (`.json`) in the current path. This file is used to program the behavior of agents to propose and vote on DAO actions based on metrics.

***

### IV. MLOps & Federated Learning

Commands to support federated learning workflows, including data preparation, pipeline creation, training, and model deployment. An overview of commands is available via `$ oasees-sdk mlops`.

* #### `oasees-sdk mlops init-project`
    * **Description**: Creates project templates for federated learning.
    * **Argument**: `<project-name>` (Required): The name of the new ML project.
    * **Usage**:
        ```bash
        oasees-sdk mlops init-project <project-name>
        ```

* #### `oasees-sdk mlops prepare-dataset`
    * **Description**: Prepares a device for federated learning. It creates and uploads synthetic data to IPFS while keeping the original data on the device.
    * **Arguments**:
        * `<samples.npy>` (Required): The `.npy` file containing the data samples/features.
        * `<labels.npy>` (Required): The `.npy` file containing the data labels.
    * **Usage**:
        ```bash
        oasees-sdk mlops prepare-dataset iris_data.npy iris_target.npy
        ```

* #### `oasees-sdk mlops ipfs-ls`
    * **Description**: Lists files stored in IPFS for a given project path.
    * **Usage**:
        ```bash
        oasees-sdk mlops ipfs-ls projects/ml
        ```

* #### `oasees-sdk mlops fl-data-nodes`
    * **Description**: Lists the devices that have data ready for federated learning.
    * **Usage**:
        ```bash
        oasees-sdk mlops fl-data-nodes
        ```

* #### `oasees-sdk mlops start-fl`
    * **Description**: Starts a federated learning process with a server and clients.
    * **Arguments**:
        * `--project-name TEXT` (Required): The name of the ML project.
        * `--data-files TEXT` (Required): A colon-separated list of `data,target,node` triplets (e.g., "file.npy,target.npy,node1:file.npy,target.npy,node2").
        * `--num-rounds INTEGER` (Optional): The number of training rounds (default: 5).
        * `--epochs INTEGER` (Optional): The number of epochs per client per round (default: 5).
    * **Usage**:
        ```bash
        oasees-sdk mlops start-fl --project-name example1 --data-files "iris_data.npy,iris_target.npy,device1:iris_data.npy,iris_target.npy,device2"
        ```

* #### `oasees-sdk mlops deploy-model`
    * **Description**: Deploys a trained model for inference.
    * **Arguments**:
        * `--project=<project-name>` (Required): The name of the project the model belongs to.
        * `--model=<model-file.pkl>` (Required): The name of the trained model file.
    * **Usage**:
        ```bash
        oasees-sdk mlops deploy-model --project-name=example1 --model=example1_2025-07-24_06-16-15.pkl
        ```

#### <ins>Uninstalling</ins>

If you want to uninstall your cluster (e.g. you've restarted your OASEES stack) run the following **on the master node**:

    oasees-sdk uninstall master # UNINSTALLS K3S SERVER ON MASTER NODE

and **on <ins>each</ins> of your worker nodes**:

    oasees-sdk uninstall agent #UNINSTALLS K3S AGENT ON WORKER NODE


<br>

## Python Module SDK

Information about the Python module SDK to be added.