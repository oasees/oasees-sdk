sdk_manager_manifest = '''
apiVersion: apps/v1
kind: Deployment
metadata:
  name: oasees-ml-ops-deployment
spec:
  replicas: 1
  selector:
    matchLabels:
      app: oasees-ml-ops
  template:
    metadata:
      labels:
        app: oasees-ml-ops
    spec:
      containers:
      - name: oasees-ml-ops
        image: andreasoikonomakis/oasees-ml-ops:latest             
        env:
        - name: CONFIG_CID
          value: "REPLACE_CONFIG_CID"
        - name: IPFS_API_URL
          value: "REPLACE_IPFS_API_URL"
        - name: OASEES_IPFS_API_URL
          value: "REPLACE_OASEES_IPFS_API_URL"
        - name: IPFS_ENDPOINT
          value: "REPLACE_IPFS_ENDPOINT"
        - name: OASEES_IPFS_IP
          value: "REPLACE_OASEES_IPFS_IP"
        - name: USERNAME
          value: REPLACE_USERNAME
        - name: PASS
          value: REPLACE_PASS
        - name: PWD
          value: "REPLACE_PWD"
        - name: INDEXER_ENDPOINT
          value: "REPLACE_INDEXER_ENDPOINT"
        ports:
        - containerPort: 31007
      nodeSelector:
        node-role.kubernetes.io/master: "true"

---
apiVersion: v1
kind: Service
metadata:
  name: oasees-ml-ops-service
spec:
  selector:
    app: oasees-ml-ops
  type: NodePort
  ports:
    - name: api
      port: 31007
      targetPort: 31007
      protocol: TCP
      nodePort: 31007

'''
