helper_template = '''import logging


ipfs_endpoint = "REPLACE_IPFS"
indexer_endpoint = "REPLACE_INDEXER"

def setup_logger(name, log_file, level=logging.DEBUG):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    file_handler = logging.FileHandler(log_file)
    console_handler = logging.StreamHandler()

    file_handler.setLevel(level)
    console_handler.setLevel(level)

    formatter = logging.Formatter('%(asctime)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
'''

runtime_params = '''
import json
import os
import shutil
import sys

notebook = sys.argv[1]
epochs = int(sys.argv[2])
learning_rate = float(sys.argv[3])
ipfs_hash = sys.argv[4]
retrain = int(sys.argv[5])
sample_num = int(sys.argv[6])
batch_size = int(sys.argv[7])
source_cid = sys.argv[8]


with open("exec_params.json", 'r') as json_file:
    exec_params = json.load(json_file)

    exec_params['epochs'] = epochs
    exec_params['learning_rate'] = learning_rate
    exec_params['batch_size'] = batch_size
    exec_params['ipfs_hash'] = ipfs_hash
    exec_params['sample_num'] = sample_num
    exec_params["source_cid"] = source_cid
    exec_params['retrain'] = retrain


with open("exec_params.json", 'w') as json_file:
    json.dump(exec_params, json_file)



with open(f"{notebook}.ipynb", 'r', encoding='utf-8') as file:
    notebook_content = file.read()

notebook_json = json.loads(notebook_content)

PARAMS = ['## Execution Parameters']
IMPORTS = ['## Imports']
MODEL_DEF = ['## Model Definition']


_imports = ''
_model_def = ''


for i in range(0, len(notebook_json['cells'])):
    if notebook_json['cells'][i]['source'] == PARAMS:
        params = notebook_json['cells'][i + 1]['source']
        source_code = ''.join(params)
        local_vars = {}
        exec(source_code, {}, local_vars)

        modified_source = f'exec_params = {json.dumps(exec_params, indent=4)}\\n'
        notebook_json['cells'][i + 1]['source'] = modified_source.splitlines(keepends=True)
 

    if notebook_json['cells'][i]['source'] == IMPORTS:
    	_imports = notebook_json['cells'][i + 1]['source']
    	_imports = ''.join(_imports)


    if notebook_json['cells'][i]['source'] == MODEL_DEF:
    	_model_def = notebook_json['cells'][i + 1]['source']
    	_model_def = ''.join(_model_def)

    	break


with open("exported.py", 'w', encoding='utf-8') as file:
    file.write(_imports+"\\n"+_model_def)


with open(f"{notebook}_exec.ipynb", 'w', encoding='utf-8') as file:
    json.dump(notebook_json, file, indent=4)

clean_up  = [
    f"training_output.log",
    f"testing_output.log"
]

for c in clean_up:
    if (os.path.exists(c)):
        if(os.path.isdir(c)):
            shutil.rmtree(c)
        else:
            os.remove(c)
'''



return_results='''

import requests
import os
import json
from oasees_helpers import *
import argparse
import shutil

parser = argparse.ArgumentParser(description='Modify exec_params in a Jupyter notebook.')
parser.add_argument('--notebook', type=str, required=True, help='Path to the notebook file')

args = parser.parse_args()

folder_path = "."

files = []
folder_cid = None

clean_up  = [
    f"__pycache__",
    f".ipynb_checkpoints"
]

for c in clean_up:
    if (os.path.exists(c)):
        if(os.path.isdir(c)):
            shutil.rmtree(c)
        else:
            os.remove(c)

for root, _, filenames in os.walk(folder_path):
    for filename in filenames:
        file_path = os.path.join(root, filename)
        
        files.append(('file', (os.path.relpath(file_path, folder_path), open(file_path, 'rb'))))

try:
    response = requests.post(f"{ipfs_endpoint}/add?recursive=true&wrap-with-directory=true", files=files)
    
    if response.status_code == 200:
        folder_cid = None
        for line in response.iter_lines():
            if line:
                entry = json.loads(line)
                if entry['Name'] == "":
                    folder_cid = entry['Hash']
                    project_name = folder_path.split("/")[-1]
    else:
        raise Exception(f"Failed to add folder to IPFS: {response.text}")
finally:
    for _, (_, file) in files:
        file.close()



headers = {
    'Content-Type': 'application/json',
}
payload = {
    'project_name': args.notebook,
    'cid': folder_cid,
}


response = requests.post(f"{indexer_endpoint}/upload_results", json=payload, headers=headers)

'''


pipeline_exec_sh = '''

epochs=$EPOCHS
lr=$LR
dataset=$DATASET
samples=$SAMPLES
retrain=$RETRAIN
batch_size=$BATCH_SIZE
source_cid=$SOURCE_CID


bash runtime_pipeline.sh $epochs $lr $dataset $samples $retrain $batch_size $source_cid

'''

runtime_pipeline_sh = '''

#!/bin/bash

ploomber_path=$(which ploomber)
soorgeon_path=$(which soorgeon)



for IPYNB in *.ipynb; do
    NOTEBOOK_NAME=${IPYNB%.ipynb}
done


python3 runtime_params.py $NOTEBOOK_NAME $1 $2 $3 $4 $5 $6 $7


touch training_output.log
touch testing_output.log


tail -f training_output.log &
TAIL_PID=$!

"$soorgeon_path" refactor $NOTEBOOK_NAME"_exec.ipynb" --single-task >> /dev/null 2>&1


echo "Starting Training $NOTEBOOK_NAME"
"$ploomber_path" build --force >> /dev/null 2>&1


kill $TAIL_PID
echo "Testing Results of $NOTEBOOK_NAME"
cat testing_output.log

rm $NOTEBOOK_NAME"_exec.ipynb"
rm $NOTEBOOK_NAME"_exec-backup.ipynb"
rm -rf __pycache__
rm -rf products
rm -rf pipeline.yaml


python3 return_results.py --notebook "$NOTEBOOK_NAME"

'''

DockerFile_template='''
FROM andreasoikonomakis/ml-base-image:latest

COPY deploy.py model.pth requirements.txt oasees_helpers.py /

RUN pip3 install --no-cache-dir -r requirements.txt

CMD ["python3","deploy.py"]

'''


deploy_ml ='''

import torch
from torch import optim
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader , Dataset
import sys
import numpy as np
import io
from io import BytesIO
import requests
import cv2
from flask import Flask, request, jsonify 
from exported import Model



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.load('model.pth',map_location=device)
model.eval()

app = Flask(__name__)

@app.route('/status', methods=['GET'])
def status():
    return jsonify({'status': 'up'})


@app.route('/output', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        # Read the image file to memory
        image_stream = io.BytesIO(file.read())
        image_stream.seek(0)
        image = np.frombuffer(image_stream.read(), np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)  # Read as grayscale

        # Resize image to 28x28 pixels
        image = cv2.resize(image, (28, 28))

        # Convert image to PyTorch tensor
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions

        image = image.to(device)  # Move tensor to the device (CPU/GPU)

        with torch.no_grad():
            output, _ = model(image)
            predicted = output.argmax(1).item()

        return jsonify({'prediction': predicted})

@app.route('/predict_numpy', methods=['POST'])
def predict_numpy():
    data = request.get_json()

    if 'image' not in data:
        return jsonify({'error': 'No image in request'}), 400

    image = np.array(data['image'], dtype=np.float32)

    if image.shape != (28, 28):
        return jsonify({'error': 'Invalid image shape, expected (28, 28)'}), 400

    image = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    image = image.to(device)

    with torch.no_grad():
        output, _ = model(image)
        predicted = output.argmax(1).item()

    return jsonify({'prediction': predicted})


if __name__ == '__main__':
    app.run(host="0.0.0.0")

'''





exec_flower_server ='''
#!/bin/bash

args=("$@")

NUM_CLIENTS=${args[0]}
ROUNDS=${args[1]}
PROJECT_NAME=${args[2]}
IPFS_ENDPOINT=${args[3]}
INDEXER_ENDPOINT=${args[4]}

rm training_output.log >> /dev/null 2>&1
rm testing_output.log >> /dev/null 2>&1

touch training_output.log 
touch testing_output.log

tail -f training_output.log & >> /dev/null 2>&1
TAIL_PID=$!

cleanup() {
    echo "Cleaning up..."
    kill $TAIL_PID
    if [[ -n "$FLOWER_PID" ]]; then
        kill $FLOWER_PID
    fi
    exit 0
}

trap cleanup SIGINT SIGTERM

python3 flower_server.py $NUM_CLIENTS $ROUNDS >> /dev/null 2>&1 &
FLOWER_PID=$!

wait $FLOWER_PID

kill $TAIL_PID


cat testing_output.log 

python3 return_results.py $PROJECT_NAME $IPFS_ENDPOINT $INDEXER_ENDPOINT
'''



exec_flower_client = '''

args=("$@")
python3 flower_client.py "${args[@]}"

'''


return_results_fl='''

import requests
import os
import json
import shutil
import sys

project_name = sys.argv[1]
ipfs_endpoint = sys.argv[2]
indexer_endpoint = sys.argv[3]



folder_path = "."

files = []
folder_cid = None

clean_up  = [
    f"__pycache__",
    f".ipynb_checkpoints"
]

for c in clean_up:
    if (os.path.exists(c)):
        if(os.path.isdir(c)):
            shutil.rmtree(c)
        else:
            os.remove(c)

for root, _, filenames in os.walk(folder_path):
    for filename in filenames:
        file_path = os.path.join(root, filename)
        
        files.append(('file', (os.path.relpath(file_path, folder_path), open(file_path, 'rb'))))

try:
    response = requests.post(f"{ipfs_endpoint}/add?recursive=true&wrap-with-directory=true", files=files)
    
    if response.status_code == 200:
        folder_cid = None
        for line in response.iter_lines():
            if line:
                entry = json.loads(line)
                if entry['Name'] == "":
                    folder_cid = entry['Hash']
                    
    else:
        raise Exception(f"Failed to add folder to IPFS: {response.text}")
finally:
    for _, (_, file) in files:
        file.close()



headers = {
    'Content-Type': 'application/json',
}
payload = {
    'project_name': project_name,
    'cid': folder_cid,
}


response = requests.post(f"{indexer_endpoint}/upload_results_fl", json=payload, headers=headers)
'''

deploy_fl = '''

import torch
from torch import optim
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader , Dataset
import sys
import numpy as np
import io
from io import BytesIO
import requests
import cv2
from flask import Flask, request, jsonify 
import FlModel


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = FlModel.Net().to(device)
model.load_state_dict(torch.load('model.pth', map_location=device))
model.eval()

app = Flask(__name__)

@app.route('/status', methods=['GET'])
def status():
    return jsonify({'status': 'up'})


@app.route('/output', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        # Read the image file to memory
        image_stream = io.BytesIO(file.read())
        image_stream.seek(0)
        image = np.frombuffer(image_stream.read(), np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)  # Read as grayscale

        # Resize image to 28x28 pixels
        image = cv2.resize(image, (28, 28))

        # Convert image to PyTorch tensor
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions

        image = image.to(device)  # Move tensor to the device (CPU/GPU)

        with torch.no_grad():
            output, _ = model(image)
            predicted = output.argmax(1).item()

        return jsonify({'prediction': predicted})


@app.route('/predict_numpy', methods=['POST'])
def predict_numpy():
    data = request.get_json()

    if 'image' not in data:
        return jsonify({'error': 'No image in request'}), 400

    image = np.array(data['image'], dtype=np.float32)

    if image.shape != (28, 28):
        return jsonify({'error': 'Invalid image shape, expected (28, 28)'}), 400

    image = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    image = image.to(device)

    with torch.no_grad():
        output, _ = model(image)
        predicted = output.argmax(1).item()

    return jsonify({'prediction': predicted})



if __name__ == '__main__':
    app.run(host="0.0.0.0")

'''


fl_server = '''
import flwr as fl
from typing import List, Tuple
from flwr.common import Metrics
import pickle
import warnings
import logging
import argparse

logging.getLogger("flwr").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning, module="flwr")

# Parse command line arguments
parser = argparse.ArgumentParser(description='Flower Federated Learning Server')
parser.add_argument('--NUM_ROUNDS', type=int, default=15, help='Number of federated learning rounds')
parser.add_argument('--MIN_CLIENTS', type=int, default=1, help='Minimum number of clients required')
parser.add_argument('--MODEL_NAME', type=str, default='my_model', help='Model name for saving')

args = parser.parse_args()

NUM_ROUNDS = args.NUM_ROUNDS
MIN_CLIENTS = args.MIN_CLIENTS
MODEL_NAME = args.MODEL_NAME

print(f"Server Configuration:")
print(f"  - NUM_ROUNDS: {NUM_ROUNDS}")
print(f"  - MIN_CLIENTS: {MIN_CLIENTS}")
print(f"  - MODEL_NAME: {MODEL_NAME}")
print()

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    if not metrics:
        return {}
    
    accuracies = [num_examples * m.get("accuracy", m.get("eval_accuracy", 0)) 
                  for num_examples, m in metrics 
                  if "accuracy" in m or "eval_accuracy" in m]
    examples = [num_examples for num_examples, m in metrics 
                if "accuracy" in m or "eval_accuracy" in m]
    
    if not accuracies:
        return {}
    
    return {"accuracy": sum(accuracies) / sum(examples)}

def weighted_average_fit(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregate training metrics from fit() calls"""
    if not metrics:
        return {}
    
    train_accuracies = [num_examples * m["train_accuracy"] for num_examples, m in metrics if "train_accuracy" in m]
    train_losses = [num_examples * m["train_loss"] for num_examples, m in metrics if "train_loss" in m]
    examples = [num_examples for num_examples, m in metrics if "train_accuracy" in m]
    
    aggregated = {}
    if train_accuracies and examples:
        aggregated["train_accuracy"] = sum(train_accuracies) / sum(examples)
    if train_losses and examples:
        aggregated["train_loss"] = sum(train_losses) / sum(examples)
    
    return aggregated

class ServerStrategy(fl.server.strategy.FedAvg):
    def __init__(self, model_save_path='final_model.pkl', **kwargs):
        super().__init__(**kwargs)
        self.model_save_path = model_save_path
        self.final_parameters = None
    
    def aggregate_fit(self, server_round, results, failures):
        parameters, metrics = super().aggregate_fit(server_round, results, failures)
        
        if results:
            print(f"=== Round {server_round} Training Results ===")
            
            for i, (client_proxy, fit_res) in enumerate(results):
                client_metrics = fit_res.metrics
                num_examples = fit_res.num_examples
                client_id = client_proxy.cid 
                
                print(f"Client {client_id}: {num_examples} samples")
                if "train_accuracy" in client_metrics:
                    print(f"  - Training accuracy: {client_metrics['train_accuracy']:.4f}")
                if "train_loss" in client_metrics:
                    print(f"  - Training loss: {client_metrics['train_loss']:.4f}")
        
        if server_round == NUM_ROUNDS and parameters is not None:
            self.final_parameters = parameters
            self._save_parameters(parameters)
        
        return parameters, metrics

    def aggregate_evaluate(self, server_round, results, failures):
        """Collect evaluation metrics"""
        aggregated_metrics = super().aggregate_evaluate(server_round, results, failures)
        
        if results:
            print(f"=== Round {server_round} Evaluation Results ===")
            for i, (client_proxy, eval_res) in enumerate(results):
                client_metrics = eval_res.metrics
                print(f"Client {i+1}: Loss={eval_res.loss:.4f}")
                if "eval_accuracy" in client_metrics:
                    print(f"  - Eval accuracy: {client_metrics['eval_accuracy']:.4f}")
                if "eval_loss" in client_metrics:
                    print(f"  - Eval loss: {client_metrics['eval_loss']:.4f}")
        
        return aggregated_metrics

    def _save_parameters(self, parameters):
        params_list = fl.common.parameters_to_ndarrays(parameters)
        
        model_data = {
            'parameters': params_list,
            'parameter_shapes': [param.shape for param in params_list],
            'parameter_dtypes': [str(param.dtype) for param in params_list],
            'num_parameters': len(params_list),
            'total_size': sum(param.size for param in params_list)
        }
        
        with open(self.model_save_path, 'wb') as f:
            pickle.dump(model_data, f,protocol=pickle.HIGHEST_PROTOCOL)
        
        print(f"Model saved to {self.model_save_path}")

strategy = ServerStrategy(
    model_save_path='{}.pkl'.format(MODEL_NAME),
    fraction_fit=1.0,
    fraction_evaluate=1.0,
    min_fit_clients=MIN_CLIENTS,
    min_evaluate_clients=MIN_CLIENTS,
    min_available_clients=MIN_CLIENTS, 
    evaluate_metrics_aggregation_fn=weighted_average,
    fit_metrics_aggregation_fn=weighted_average_fit  
)

def main():
    fl.server.start_server(
        server_address="0.0.0.0:9999",
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
    )

if __name__ == "__main__":
    main()
'''

