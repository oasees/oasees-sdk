from kubernetes import client, config, utils
from kubernetes.client import ApiException
import yaml
import os
import requests


# Read the YAML configuration
yaml_file_template = """
apiVersion: v1
kind: Namespace
metadata:
  name: EXPERIMENT
  labels:
    name: EXPERIMENT

---

apiVersion: apps/v1
kind: Deployment
metadata:
  labels:
    app: training-pod-EXPERIMENT
  name: training-pod-EXPERIMENT
  namespace: EXPERIMENT
spec:
  replicas: 1
  selector:
    matchLabels:
      app: training-pod-EXPERIMENT
  strategy: {}
  template:
    metadata:
      labels:
        app: training-pod-EXPERIMENT
    spec:
      nodeName: NODE_NAME
      containers:
      - image: andreasoikonomakis/oasees-training
        imagePullPolicy: IfNotPresent
        name: training-pod-EXPERIMENT
        env:
          - name: cid
            value: "IPFS HASH"

---
apiVersion: v1
kind: Service
metadata:
  creationTimestamp: null
  labels:
    app: training-pod-EXPERIMENT
  name: training-pod-EXPERIMENT
  namespace: EXPERIMENT
spec:
  ports:
  - name: http
    port: 80
    protocol: TCP
    targetPort: 5000
  selector:
    app: training-pod-EXPERIMENT
  type: ClusterIP
status:
  loadBalancer: {}
---
apiVersion: traefik.containo.us/v1alpha1
kind: Middleware
metadata:
  name: strip-training-pod-EXPERIMENT
  namespace: EXPERIMENT
spec:
  stripPrefix:
    prefixes:
      - /training-pod-EXPERIMENT

---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  annotations:
    traefik.ingress.kubernetes.io/router.entrypoints: web, websecure
    traefik.ingress.kubernetes.io/router.middlewares: EXPERIMENT-redirecttohttps@kubernetescrd
    traefik.ingress.kubernetes.io/router.middlewares: EXPERIMENT-strip-training-pod-EXPERIMENT@kubernetescrd
  name: training-pod-EXPERIMENT-ingress1
  namespace: EXPERIMENT
spec:
  ingressClassName: traefik
  rules:
  - http:
      paths:
      - path: /training-pod-EXPERIMENT
        pathType: Prefix
        backend:
          service:
            name: training-pod-EXPERIMENT
            port:
              number: 80
"""




def deploy_workload(filename,ipfs_endpoint,node_name):

    zip_file_path = "{}.zip".format(filename)

    if not os.path.isfile(zip_file_path):
        print(f"File {zip_file_path} not found.")
        return None

    # url = 'http://10.160.3.151:31005/api/v0/add'
    url = ipfs_endpoint
    try:
        with open(zip_file_path, 'rb') as zip_file:
            files = {
                'file': (os.path.basename(zip_file_path), zip_file),
            }
            response = requests.post(url, files=files)
        
        response.raise_for_status()
        ipfs_hash = response.json().get('Hash')
        if ipfs_hash:
            print(f"File uploaded successfully. IPFS Hash: {ipfs_hash}")
        else:
            print("Failed to retrieve IPFS hash from response.")
            return None
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return None



    yaml_file = yaml_file_template
    yaml_file = yaml_file.replace("IPFS HASH",ipfs_hash)
    yaml_file = yaml_file.replace("NODE_NAME",node_name)
    yaml_file = yaml_file.replace("EXPERIMENT", filename.replace("_","-").lower())
    # Load the YAML into a list of dictionaries
    yaml_objects = list(yaml.safe_load_all(yaml_file))
    # Create the Kubernetes API client

    config.load_kube_config("./config")
    k8s_client = client.ApiClient()
    # Create the resources
    core_resources = []
    custom_resources = []

    for obj in yaml_objects:
        if 'kind' in obj:
            if obj['apiVersion'].startswith("apps/") or obj['apiVersion'].startswith("v1") or obj['apiVersion'].startswith("networking.k8s.io"):
                core_resources.append(obj)
            else:
                custom_resources.append(obj)

    # Create core Kubernetes resources
    for obj in core_resources:
        try:
            utils.create_from_dict(k8s_client, obj)
            print(f"Created {obj['kind']} {obj['metadata']['name']}")
        except ApiException as e:
            print(f"Exception when creating {obj['kind']} {obj['metadata']['name']}: {e}")

    # Create custom resources
    custom_api = client.CustomObjectsApi()
    for obj in custom_resources:
        try:
            group, version = obj['apiVersion'].split("/")
            plural = obj['kind'].lower() + "s"
            namespace = obj['metadata'].get('namespace', 'default')
            
            # Check if the resource already exists
            try:
                custom_api.get_namespaced_custom_object(group, version, namespace, plural, obj['metadata']['name'])
                print(f"{obj['kind']} {obj['metadata']['name']} already exists. Skipping creation.")
            except ApiException as e:
                if e.status == 404:
                    # If not found, create it
                    custom_api.create_namespaced_custom_object(group, version, namespace, plural, obj)
                    print(f"Created {obj['kind']} {obj['metadata']['name']}")
                else:
                    raise e
        except ApiException as e:
            print(f"Exception when creating {obj['kind']} {obj['metadata']['name']}: {e}")

