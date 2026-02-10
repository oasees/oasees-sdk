
import subprocess
import json
import time
import hashlib

class NCE:
    """
    Node Component Executor (NCE) - Mock Implementation
    
    Responsible for:
    1. Accepting Execution Tickets (Interface M7).
    2. Translating Tickets to K8s Job Manifests.
    3. Applying Manifests (Interface M8).
    4. Monitoring and Cleaning up resources.
    """
    
    def __init__(self, nds):
        self.nds = nds
        print("[NCE] Initialized.")

    def submit_ticket(self, ticket):
        """
        M7: Submit Execution Ticket
        Accepts a fully resolved 'Context Bundle' from NWO.
        """
        job_id = ticket.get("job_id", f"job-{int(time.time())}")
        print(f"[NCE] Received Execution Ticket: {job_id}")
        if ticket.get("target_node"):
            print(f"[NCE] Target node: {ticket.get('target_node')}")
        
        # 1. Translate to Manifest
        manifest = self.generate_manifest(ticket)
        
        # 2. Apply Manifest (M8)
        self.apply_manifest(manifest)
        
        # 3. Update Status
        self.nds.update_status(job_id, "RUNNING")
        
        return job_id

    def generate_manifest(self, ticket):
        """
        Generate a K8s Job manifest from the execution ticket.
        """
        print("[NCE] Generating Kubernetes Job Manifest...")
        # In a real implementation, this would use the templates from `training_workload.py`
        env_vars = ticket.get("env_vars") or {}
        if isinstance(env_vars, dict):
            env_list = [{"name": k, "value": str(v)} for k, v in env_vars.items()]
        else:
            env_list = env_vars

        job_id = ticket.get("job_id")
        service_id = ticket.get("service_id")
        execution_strategy = ticket.get("execution_strategy")
        target_node = ticket.get("target_node")

        if job_id:
            env_list.append({"name": "OASEES_JOB_ID", "value": job_id})
        if service_id:
            env_list.append({"name": "OASEES_SERVICE_ID", "value": service_id})
        if execution_strategy:
            env_list.append({"name": "OASEES_EXEC_STRATEGY", "value": execution_strategy})
        if target_node:
            env_list.append({"name": "OASEES_TARGET_NODE", "value": target_node})

        manifest = {
            "apiVersion": "batch/v1",
            "kind": "Job",
            "metadata": {"name": ticket["job_id"]},
            "spec": {
                "template": {
                    "spec": {
                        "containers": [{
                            "name": "workload",
                            "image": ticket.get("image", "oasees/ml-base-image:latest"),
                            "env": env_list
                        }],
                        "restartPolicy": "Never"
                    }
                }
            }
        }
        return manifest

    def apply_manifest(self, manifest):
        """
        M8: Apply Job Manifest (kubectl apply)
        """
        print(f"[NCE] Applying Manifest: {manifest['metadata']['name']}")
        
        # Verify valid manifest
        if not manifest:
            print("[NCE] Error: Empty manifest generated.")
            return False

        # In a real scenario, we would use `subprocess.run(['kubectl', 'apply', ...])`
        # For this prototype, we print the command.
        print(f"[NCE][MOCK] execute: kubectl apply -f {manifest['metadata']['name']}.yaml")
        time.sleep(1) # Simulate delay
        return True

    def generate_attestation(self, manifest, ticket):
        """
        Create a simple execution attestation based on manifest hash.
        """
        manifest_str = json.dumps(manifest, sort_keys=True)
        manifest_hash = hashlib.sha256(manifest_str.encode()).hexdigest()
        attestation = {
            "job_id": ticket.get("job_id"),
            "node_id": (ticket.get("node_profile") or {}).get("node_id"),
            "manifest_hash": manifest_hash,
            "timestamp": time.time()
        }
        print(f"[NCE] Attestation generated: {attestation['manifest_hash'][:12]}...")
        return attestation

    def cleanup(self, job_id):
        """
        Cleanup resources after completion.
        """
        print(f"[NCE] Cleaning up resources for job: {job_id}")
        self.nds.update_status(job_id, "CLEANED")
        self.nds.release_resources(job_id)

    def monitor_job(self, job_id):
        """
        Simulate basic job monitoring and status transitions.
        """
        print(f"[NCE] Monitoring job: {job_id}")
        time.sleep(0.4)
        self.nds.update_status(job_id, "RUNNING")
        time.sleep(0.6)
        self.nds.update_status(job_id, "SUCCEEDED")
        return {"job_id": job_id, "status": "SUCCEEDED"}
