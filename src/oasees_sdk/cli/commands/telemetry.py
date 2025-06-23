import click
import subprocess
import json
import os
from pathlib import Path
import requests
import re

@click.group(name='telemetry')
def telemetry_commands():
    '''OASEES Telemetry Management Utilities'''
    pass

def get_telemetry_api():
    try:
        tel_svc = subprocess.run(['kubectl','get','svc','oasees-telemetry-api-svc','-o','jsonpath={.spec.clusterIP}'], 
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        tel_ip = tel_svc.stdout.strip()
        return f"http://{tel_ip}:5005"
    except subprocess.CalledProcessError as e:
        click.secho(f"Error getting Telemetry api service: {e.stderr}", fg="red", err=True)
        return None


def sanitize_k8s_name(name):
    """Sanitize names for Kubernetes (lowercase, replace underscores with hyphens, remove invalid chars)"""
    sanitized = name.lower()
    sanitized = re.sub(r'_', '-', sanitized)
    sanitized = re.sub(r'[^a-z0-9-]', '', sanitized)
    sanitized = re.sub(r'^-+|-+$', '', sanitized)
    return sanitized


def get_thanos_endpoint():
    result = subprocess.run(['kubectl', 'get', 'service', 'thanos-query', '-n', 'default', '-o', 'jsonpath={.spec.clusterIP}'], 
                        capture_output=True, text=True)
    cluster_ip = result.stdout.strip()

    url = f"http://{cluster_ip}:9090/api/v1/query"

    return url


@telemetry_commands.command()
def metrics_index():
    '''List All metric names'''

    metrics = []

    url = get_thanos_endpoint()
    params = {'query': '{__name__=~"oasees_.*"}'}
    response = requests.get(url, params=params)
    data = response.json()

    metric_names = set()
    for result in data['data']['result']:
        metric_name = result['metric'].get('metric_index')
        if metric_name:
            metric_names.add(metric_name)

    for name in sorted(metric_names):
        metrics.append(name)

    for m in metrics:
        print(m)

    return metrics

@telemetry_commands.command()
@click.option('--index', '-i', help='Name of the metric to search for')
def metrics_list(index):
    '''Get All metrics by name'''

    url = get_thanos_endpoint()
    params = {'query': f'{{__name__=~"oasees_.*", metric_index="{index}"}}'}
    response = requests.get(url, params=params)
    data = response.json()

    metric_names = set()
    by_source_dict = {}
    for result in data['data']['result']:
        source = result['metric'].get('source')


        oasees_metric = result['metric'].get('__name__').replace("oasees_","")

        if oasees_metric not in by_source_dict:
            by_source_dict[oasees_metric] = set()

        by_source_dict[oasees_metric].add(source)


        if oasees_metric:
            metric_names.add(oasees_metric)

    for m in by_source_dict:
        print(m,"[",*by_source_dict[m],"]")


@telemetry_commands.command()
@click.argument('metrics', nargs=-1, required=True)
def check_metrics(metrics):
    '''Check metrics with optional thresholds'''
    
    import re, requests
    
    pattern = r'(\w+)\[(.*?)\]'
    threshold_pattern = r'(\w+)\[(\w+)\](>=|<=|>|<|=)(\d+(?:\.\d+)?)'
    
    queries = []
    thresholds = {}
    
    for arg in metrics:
        if re.match(threshold_pattern, arg):
            match = re.match(threshold_pattern, arg)
            metric, source, op, thr = match.groups()
            thresholds[f"{metric}_{source}"] = (op, float(thr))
        else:
            match = re.match(pattern, arg)
            if match:
                index, metrics_str = match.groups()
                for m in metrics_str.split(','):
                    if m.strip():
                        queries.append((index, f"oasees_{m.strip()}"))
    
    url = get_thanos_endpoint()
    
    for index, metric in queries:
        query = f'{metric}{{metric_index="{index}"}}'
        response = requests.get(url, params={"query": query}).json()
        
        for result in response.get("data", {}).get("result", []):
            source = result.get("metric", {}).get("source", "")
            value = float(result.get("value", ["", "0"])[1])
            
            key = f"{metric.replace('oasees_', '')}_{source}"
            if key in thresholds:
                op, thr = thresholds[key]
                ops = {'>': lambda x, y: x > y, '<': lambda x, y: x < y, '>=': lambda x, y: x >= y, 
                       '<=': lambda x, y: x <= y, '=': lambda x, y: x == y}
                if ops[op](value, thr):
                    print(f"{source} | {metric.replace('oasees_', '')} = {value} | {op} {thr}")
            else:
                print(f"{source} | {metric.replace('oasees_', '')} = {value}")


@telemetry_commands.command()
def gen_oasees_collector():
    """Generate the OASEES collector script"""
    
    script_content = '''import socketio
import requests
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--metric-index', required=True, help='Metric index')
parser.add_argument('-s', '--source', required=True, help='Source identifier')
parser.add_argument('-o', '--oasees-endpoint', required=True, help='OASEES endpoint URL')
parser.add_argument('-e', '--source-endpoint', required=True, help='Source endpoint URL')
parser.add_argument('-i', '--scrape-interval', type=int, default=1, help='Scrape interval in seconds (default: 1)')

args = parser.parse_args()

METRIC_INDEX = args.metric_index
SOURCE = args.source
OASEES_ENDPOINT = 'http://{}:30080'.format(args.oasees_endpoint)
SOURCE_ENDPOINT = args.source_endpoint
SCRAPE_INTERVAL = args.scrape_interval

sio = socketio.Client()

@sio.on('metric_pushed')
def on_response(data):
    print(f"Success: {data}")

@sio.on('error')
def on_error(data):
    print(f"Error: {data}")

sio.connect(OASEES_ENDPOINT)

while True:
    r = requests.get(SOURCE_ENDPOINT)
    data = r.json()
    data.update({'metric_index': METRIC_INDEX, 'source': SOURCE})
    sio.emit('push_metric', data)
    time.sleep(SCRAPE_INTERVAL)
'''
    
    click.echo(script_content)


@telemetry_commands.command()
@click.option('--metric-index','-i',required=True, help='Metric index')
@click.option('--source','-s',required=True, help='Source identifier')
@click.option('--source-endpoint','-se',required=True, help='Source endpoint URL')
@click.option('--scrape-interval','-si',type=int, default=1, help='Scrape interval in seconds (default: 1)')
def deploy_collector(metric_index, source, source_endpoint, scrape_interval):
   """Deploy OASEES metric collector"""
   
   import tempfile
   import os
   import subprocess
   import re
   
   def sanitize_k8s_name(name):
       """Sanitize names for Kubernetes (lowercase, replace underscores with hyphens, remove invalid chars)"""
       sanitized = name.lower()
       sanitized = re.sub(r'_', '-', sanitized)
       sanitized = re.sub(r'[^a-z0-9-]', '', sanitized)
       sanitized = re.sub(r'^-+|-+$', '', sanitized)
       return sanitized
   
   # Get OASEES Telemetry API endpoint
   oasees_endpoint = get_telemetry_api()
   if not oasees_endpoint:
       return
   
   # Generate unique names
#    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
   sanitized_source = sanitize_k8s_name(source)
   
   pod_name = f"collector-{sanitized_source}"
   configmap_name = f"collector-script-{sanitized_source}"
   
   # Display configuration
   click.echo("Deploying OASEES Collector with the following configuration:")
   click.echo(f"  Metric Index: {metric_index}")
   click.echo(f"  Source: {source}")
   click.echo(f"  OASEES Endpoint: {oasees_endpoint}")
   click.echo(f"  Source Endpoint: {source_endpoint}")
   click.echo(f"  Scrape Interval: {scrape_interval} seconds")
   click.echo(f"  Pod Name: {pod_name}")
   click.echo(f"  ConfigMap Name: {configmap_name}")
   click.echo()
   
   # Python collector script content
   collector_script = '''import socketio
import requests
import time
import os

# Get configuration from environment variables
METRIC_INDEX = os.environ['METRIC_INDEX']
SOURCE = os.environ['SOURCE']
OASEES_ENDPOINT = os.environ['OASEES_ENDPOINT']
SOURCE_ENDPOINT = os.environ['SOURCE_ENDPOINT']
SCRAPE_INTERVAL = int(os.environ.get('SCRAPE_INTERVAL', '1'))

print(f"Starting collector with config:")
print(f"  Metric Index: {METRIC_INDEX}")
print(f"  Source: {SOURCE}")
print(f"  OASEES Endpoint: {OASEES_ENDPOINT}")
print(f"  Source Endpoint: {SOURCE_ENDPOINT}")
print(f"  Scrape Interval: {SCRAPE_INTERVAL}s")

sio = socketio.Client()

@sio.on('metric_pushed')
def on_response(data):
   print(f"Success: {data}")

@sio.on('error')
def on_error(data):
   print(f"Error: {data}")

@sio.on('connect')
def on_connect():
   print("Connected to OASEES")

@sio.on('disconnect')
def on_disconnect():
   print("Disconnected from OASEES")

try:
   print(f"Connecting to {OASEES_ENDPOINT}...")
   sio.connect(OASEES_ENDPOINT)
   
   while True:
       try:
           print(f"Scraping data from {SOURCE_ENDPOINT}...")
           r = requests.get(SOURCE_ENDPOINT)
           data = r.json()
           data.update({'metric_index': METRIC_INDEX, 'source': SOURCE})
           print(f"Pushing data: {data}")
           sio.emit('push_metric', data)
           time.sleep(SCRAPE_INTERVAL)
       except requests.exceptions.RequestException as e:
           print(f"Error scraping data: {e}")
           time.sleep(SCRAPE_INTERVAL)
       except Exception as e:
           print(f"Error processing data: {e}")
           time.sleep(SCRAPE_INTERVAL)
           
except Exception as e:
   print(f"Failed to connect to OASEES: {e}")
   exit(1)
'''

   # Create YAML content
   yaml_content = f"""# ConfigMap with collector script
apiVersion: v1
kind: ConfigMap
metadata:
 name: {configmap_name}
 labels:
   app: oasees-collector
   source: {sanitized_source}
   tag: collector
data:
 collector.py: |
{chr(10).join('    ' + line for line in collector_script.split(chr(10)))}

---
# Collector Pod
apiVersion: v1
kind: Pod
metadata:
 name: {pod_name}
 labels:
   app: oasees-collector
   source: {sanitized_source}
   tag: collector
spec:
 restartPolicy: Never
 containers:
 - name: collector
   image: python:3.9-slim
   command: ["sh", "-c"]
   args:
   - |
     pip install python-socketio requests &&
     python /app/collector.py
   env:
     - name: METRIC_INDEX
       value: "{metric_index}"
     - name: SOURCE
       value: "{source}"
     - name: OASEES_ENDPOINT
       value: "{oasees_endpoint}"
     - name: SOURCE_ENDPOINT
       value: "{source_endpoint}"
     - name: SCRAPE_INTERVAL
       value: "{scrape_interval}"
   volumeMounts:
   - name: script-volume
     mountPath: /app
 volumes:
 - name: script-volume
   configMap:
     name: {configmap_name}
"""

   # Apply to Kubernetes
   temp_file_path = None
   try:
       with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as temp_file:
           temp_file.write(yaml_content)
           temp_file_path = temp_file.name
       
       click.echo("Applying Kubernetes configuration...")
       result = subprocess.run(['kubectl', 'apply', '-f', temp_file_path], 
                             capture_output=True, text=True, check=True)
       
       click.secho("✅ Deployment successful!", fg="green")
       click.echo()
       click.echo("Resources created:")
       click.echo(f"  Pod: {pod_name}")
       click.echo(f"  ConfigMap: {configmap_name}")
       click.echo()
       click.echo("To check the status:")
       click.echo(f"  kubectl get pod {pod_name}")
       click.echo(f"  kubectl logs {pod_name} -f")
       click.echo()
       click.echo("To delete the collector:")
       click.echo(f"  kubectl delete pod {pod_name}")
       click.echo(f"  kubectl delete configmap {configmap_name}")
       click.echo()
       click.secho("Collector deployment completed successfully!", fg="green")
       
   except subprocess.CalledProcessError as e:
       click.secho("❌ Deployment failed!", fg="red")
       click.secho(f"Error: {e.stderr}", fg="red")
   finally:
       # Clean up temporary file
       if temp_file_path and os.path.exists(temp_file_path):
           os.unlink(temp_file_path)