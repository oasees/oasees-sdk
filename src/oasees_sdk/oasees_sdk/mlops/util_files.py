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