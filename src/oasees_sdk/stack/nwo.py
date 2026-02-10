
import uuid
from .nce import NCE
from .nds import NDS

class NWO:
    """
    Node Workload Orchestrator (NWO) - Mock Implementation
    
    Responsible for:
    1. Listening for Swarm Instructions (Interface M6).
    2. Orchestrating local resources via NDS.
    3. Building execution tickets for NCE.
    4. Dispatching execution to NCE.
    """
    
    def __init__(self, nce: NCE, nds: NDS):
        self.nce = nce
        self.nds = nds
        print("[NWO] Initialized.")

    def handle_swarm_notification(self, payload):
        """
        M6: Swarm Execution Trigger
        Receives a CloudEvent/JSON from DSM.
        """
        print(f"[NWO][M6] Received Swarm Notification: {payload.get('type')}")
        data = payload.get("data", {})
        service_id = data.get("serviceId")
        swarm_nodes = (data.get("swarm") or {}).get("nodes", [])
        requirements = data.get("parameters", {}).get("requirements")

        # 1. Multi-node Scheduling (Swarm)
        schedule = self.schedule_nodes(swarm_nodes, requirements)
        selected_node = (schedule.get("selected") or {}).get("node_id")
        if selected_node and selected_node != self.nds.node_profile.get("node_id"):
            print(f"[NWO] Selected remote node {selected_node}. Forwarding ticket.")
            return {"status": "FORWARDED", "target_node": selected_node, "schedule": schedule}

        # 2. Check Resources (NDS)
        if not self.nds.check_resources(requirements):
            print(f"[NWO] Resource check failed for service: {service_id}")
            return {"status": "FAILED", "reason": "Insufficient Resources"}
        
        # 3. Prepare Execution Ticket
        job_id = f"job-{uuid.uuid4().hex[:8]}"
        allocation = self.nds.reserve_resources(job_id, requirements)
        execution_strategy = data.get("parameters", {}).get("execution_strategy", "local-k8s")
        execution_ticket = {
            "job_id": job_id,
            "service_id": service_id,
            "image": data.get("parameters", {}).get("image"),
            "env_vars": data.get("parameters", {}).get("env"),
            "volume_mounts": data.get("parameters", {}).get("volumes"),
            "allocation": allocation,
            "execution_strategy": execution_strategy,
            "target_node": selected_node or self.nds.node_profile.get("node_id"),
            "schedule": schedule
        }
        
        # 4. Dispatch to NCE (M7)
        print(f"[NWO][M7] Dispatching Execution Ticket to NCE")
        self.nce.submit_ticket(execution_ticket)
        
        return {
            "status": "ACCEPTED",
            "job_id": job_id,
            "execution_strategy": execution_strategy,
            "target_node": execution_ticket["target_node"],
            "schedule": schedule
        }

    def score_node(self, node, requirements):
        score = 0.0
        if requirements and requirements.get("gpu"):
            score += 50 if node.get("gpu") else -100
        score += max(0, 100 - node.get("latency_ms", 100)) / 2
        score += max(0, (1 - node.get("load", 0.5)) * 20)
        score += node.get("cpu", 0) * 2

        # Fairness: prefer nodes with fewer recent jobs
        recent_jobs = node.get("recent_jobs", 0)
        score -= recent_jobs * 5
        return round(score, 2)

    def schedule_nodes(self, candidates, requirements):
        if not candidates:
            return {"selected": {"node_id": self.nds.node_profile.get("node_id")}, "scores": []}
        scores = []
        for node in candidates:
            scored = dict(node)
            scored["score"] = self.score_node(node, requirements)
            scores.append(scored)
        scores.sort(key=lambda n: n["score"], reverse=True)
        return {"selected": scores[0], "scores": scores}
