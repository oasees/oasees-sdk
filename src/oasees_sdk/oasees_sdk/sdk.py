import sys
# from web3 import Web3
from pathlib import Path
from dotenv import load_dotenv
import json
import subprocess
from .mlops.util_files import *
from ipylab import JupyterFrontEnd
import re
import ast
import time


def wait_for_file(file_path, timeout=30, check_interval=0.5):
    """Wait for a file to be created, with timeout."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        if file_path.exists():
            # Additional check to ensure file is not empty and fully written
            try:
                if file_path.stat().st_size > 0:
                    return True
            except OSError:
                pass
        time.sleep(check_interval)
    return False


def extract_and_reconstruct_magic_code(content):
    """Extract code from get_ipython().run_cell_magic() calls and reconstruct it"""
    
    """Simple approach to extract code from magic calls"""
    
    # Find all get_ipython().run_cell_magic calls
    lines = content.split('\n')
    result_lines = []
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Check if this line starts a magic call
        if 'get_ipython().run_cell_magic(' in line:
            # print(f"Found magic call starting at line {i}: {line[:50]}...")
            
            # Find the complete magic call (might span multiple lines)
            magic_call = line
            paren_count = line.count('(') - line.count(')')
            
            # Continue reading lines until parentheses are balanced
            j = i + 1
            while paren_count > 0 and j < len(lines):
                magic_call += '\n' + lines[j]
                paren_count += lines[j].count('(') - lines[j].count(')')
                j += 1
            
            print(f"Complete magic call: {magic_call[:100]}...")
            
            # Extract the code string (third argument)
            try:
                # Use regex to find the third argument
                pattern = r"get_ipython\(\)\.run_cell_magic\(\s*['\"][^'\"]*['\"],\s*['\"][^'\"]*['\"],\s*(.*)\s*\)$"
                match = re.search(pattern, magic_call, re.DOTALL)
                
                if match:
                    code_str = match.group(1).strip()
                    # print(f"Extracted code string: {code_str[:50]}...")
                    
                    # Parse the string literal
                    try:
                        actual_code = ast.literal_eval(code_str)
                        print("Successfully extracted code!")
                        result_lines.append(actual_code)
                    except:
                        # print("ast.literal_eval failed, trying manual parsing...")
                        # Manual parsing
                        if code_str.startswith('"') and code_str.endswith('"'):
                            actual_code = code_str[1:-1]
                            actual_code = actual_code.replace('\\n', '\n')
                            actual_code = actual_code.replace('\\"', '"')
                            actual_code = actual_code.replace('\\\\', '\\')
                            result_lines.append(actual_code)
                        else:
                            result_lines.append(line)  # Keep original if can't parse
                else:
                    print("Could not match pattern")
                    result_lines.append(line)  # Keep original
                    
            except Exception as e:
                print(f"Error processing: {e}")
                result_lines.append(line)  # Keep original
            
            i = j  # Skip the lines we've processed
        else:
            result_lines.append(line)
            i += 1
    
    return '\n'.join(result_lines)


def get_ipfs_api():
    """Get IPFS service IP from Kubernetes"""
    try:
        ipfs_svc = subprocess.run(['kubectl','get','svc','oasees-ipfs','-o','jsonpath={.spec.clusterIP}'], 
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        ipfs_ip = ipfs_svc.stdout.strip()
        return f"/ip4/{ipfs_ip}/tcp/5001/http"
    except subprocess.CalledProcessError as e:
        return None


def ipfs_add():
    api_endpoint = get_ipfs_api()
    _name = "../{}".format(Path.cwd().name)
    
    # Remove existing folder first
    rm_cmd = ['ipfs', f'--api={api_endpoint}', 'files', 'rm', '-r', f'/oasees-ml-ops/projects/ml/{Path.cwd().name}']
    subprocess.run(rm_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    add_cmd = ['ipfs', f'--api={api_endpoint}', 'add', '-q', '-r', _name]
    add_result = subprocess.run(add_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    hash_lines = add_result.stdout.strip().split('\n')
    file_hash = hash_lines[-1].strip()
    
    cp_cmd = ['ipfs', f'--api={api_endpoint}', 'files', 'cp', f'/ipfs/{file_hash}', f'/oasees-ml-ops/projects/ml/{Path.cwd().name}']
    res = subprocess.run(cp_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def convert(deploy=False):
    import subprocess
    import re
    from pathlib import Path

    try:
        app = JupyterFrontEnd()
        app.commands.execute('docmanager:save')
    except:
        print("Warning: Could not save notebook automatically")

    _name = Path.cwd().name

    if deploy:
        convert_files = ["{}_deploy.ipynb".format(_name)]
    else:
        convert_files = ["{}_client.ipynb".format(_name)]

    # convert_files = ["{}_client.ipynb".format(_name), "{}_deploy.ipynb".format(_name)]

    for cf in convert_files:
        notebook_path = Path(cf)
        if not notebook_path.exists():
            print(f"Warning: {cf} not found, skipping...")
            continue
            
        # print(f"Converting {cf}")
        
        nbconvert_cmd = [
            "jupyter", "nbconvert", cf, "--to", "script",
            "--TagRemovePreprocessor.enabled=True",
            "--TagRemovePreprocessor.remove_cell_tags=['skip-execution']"
        ]
        
        result = subprocess.run(nbconvert_cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"nbconvert failed for {cf}: {result.stderr}")
            continue

        py_file = cf.replace('.ipynb', '.py')
        py_path = Path(py_file)

        if not wait_for_file(py_path, timeout=30):
            continue


        if py_path.exists():
            with open(py_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract and reconstruct magic code
            content = extract_and_reconstruct_magic_code(content)
            # Remove other unwanted lines
            lines_to_remove = [
                r'# In\[\d*\]:\s*\n', 
                r'# Out\[\d*\]:\s*\n'
            ]
            
            for pattern in lines_to_remove:
                content = re.sub(pattern, '', content)
            
            # Clean up excessive newlines
            content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)
            
            with open(py_path, 'w', encoding='utf-8') as f:
                f.write(content)



    ipfs_add()

def list_sample_data():
    cmd = ["oasees-sdk","mlops","ipfs-ls","synthetic_data"]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out = result.stdout
    parsed = out.split('\n')
    for p in parsed:
        if(".npy" in p):
            print(p)

def get_sample_data(file1, file2):
   cmd1 = ["oasees-sdk","mlops","ipfs-get", "/oasees-ml-ops/synthetic_data/{}".format(file1)]
   cmd2 = ["oasees-sdk","mlops","ipfs-get", "/oasees-ml-ops/synthetic_data/{}".format(file2)]
   
   result1 = subprocess.run(cmd1, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
   result2 = subprocess.run(cmd2, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
   
   if result1.returncode == 0:
       print(f"Downloaded {file1}")
   else:
       print(f"Failed to download {file1}: {result1.stderr}")
       
   if result2.returncode == 0:
       print(f"Downloaded {file2}")
   else:
       print(f"Failed to download {file2}: {result2.stderr}")


def run_test_fl_server():

    from pathlib import Path
    import os
    _path = Path.cwd()

    os.system("python fl_server.py --NUM_ROUNDS 1 --MIN_CLIENTS 1 --MODEL_NAME {} &".format(_path.name))

