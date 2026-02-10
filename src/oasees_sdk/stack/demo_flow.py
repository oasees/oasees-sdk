
import time
from .nwo import NWO
from .nce import NCE
from .nds import NDS
from .ndm import NDM

def main():
    print("=== OASEES SDK: NCE & NWO Showcase Demo ===\n")

    # 1. Initialize Components
    print("--- [Phase 0] Initialization ---")
    nds = NDS()
    ndm = NDM(nds)
    nce = NCE(nds)
    nwo = NWO(nce, nds)
    print("\n")

    baseline_nodes = [
        {"node_id": "edge-node-04", "cpu": 4, "gpu": True, "load": 0.45, "latency_ms": 80, "recent_jobs": 0},
    ]
    swarm_nodes = [
        {"node_id": "edge-node-01", "cpu": 2, "gpu": False, "load": 0.6, "latency_ms": 85, "recent_jobs": 3},
        {"node_id": "edge-node-02", "cpu": 4, "gpu": True, "load": 0.7, "latency_ms": 60, "recent_jobs": 6},
        {"node_id": "edge-node-03", "cpu": 8, "gpu": True, "load": 0.2, "latency_ms": 30, "recent_jobs": 9},
        {"node_id": "edge-node-04", "cpu": 4, "gpu": True, "load": 0.5, "latency_ms": 75, "recent_jobs": 1},
    ]

    def run_execution(run_label, nodes, policy_label):
        print(f"=== {run_label} ===")
        print(f"--- [{run_label}] Phase 1: Orchestration ---")
        swarm_event = {
            "specversion": "1.0",
            "type": "com.cognets.swarm.execute",
            "source": "urn:ngsi-ld:DSM:001",
            "data": {
                "serviceId": "urn:ngsi-ld:AIModel:TrafficV2",
                "swarm": {"nodes": nodes},
                "parameters": {
                    "image": "registry/model:v2",
                    "env": {
                        "MODEL_TYPE": "TRAFFIC",
                        "INPUT_VIDEO": "demo_assets/traffic_video.mp4"
                    },
                    "requirements": {"gpu": True},
                    "execution_strategy": "local-k8s"
                }
            }
        }
        
        response = nwo.handle_swarm_notification(swarm_event)
        print(f"[Result] NWO Response: {response}")
        job_id = response.get("job_id")
        schedule = response.get("schedule") or {}
        
        if not job_id:
            print("Execution failed.")
            return

        if schedule.get("selected"):
            print(f"[NWO] Selected node: {schedule['selected'].get('node_id')}")
            print(f"[NWO] Scheduling policy: {policy_label}")
            if policy_label == "fairness":
                latencies = [n.get("latency_ms") for n in (schedule.get("scores") or []) if n.get("latency_ms") is not None]
                if latencies:
                    avg_lat = sum(latencies) / len(latencies)
                    sel_lat = schedule["selected"].get("latency_ms")
                    if sel_lat:
                        reduction = round((1 - (sel_lat / avg_lat)) * 100, 1)
                        print(f"[Swarm Advantage] Estimated latency reduction: {reduction}% vs average")
            else:
                print("[Swarm Advantage] Baseline (single-node): latency 0%, throughput 1.0x, cost 0%")
        print("\n")

        # Phase 3: Simulate Autonomous Runtime
        print(f"--- [{run_label}] Phase 3: Autonomous Runtime ---")
        time.sleep(1)
        
        print("[Model] Requesting Sensor Data from NDS (M8*)...")
        sensor_data = nds.get_sensor_data("urn:ngsi-ld:Sensor:Camera01")
        print(f"[Model] Received Data: {sensor_data['data'].keys()}")
        
        print("[Model] executing inference...")
        time.sleep(0.5)
        inference_result = {"class": "car", "confidence": 0.98, "bbox": [10, 20, 100, 200]}
        
        processed_video = "demo_assets/traffic_video_processed.mp4"
        print(f"[Model] Writing processed video with detections -> {processed_video}")
        print("[Model] Pushing result to NDM (M10)...")
        ndm.process_output(inference_result, job_id=job_id)

        nce.monitor_job(job_id)
        print("\n")

        # Phase 4: Completion
        print(f"--- [{run_label}] Phase 4: Completion ---")
        time.sleep(1)
        print(f"[System] Job {job_id} completed successfully.")
        nce.cleanup(job_id)
        print("\n")

    run_execution("Run 1: Single Node (Baseline)", baseline_nodes, "single-node")
    run_execution("Run 2: Swarm Enabled (Fairness)", swarm_nodes, "fairness")

    print("=== Demo Completed ===")

if __name__ == "__main__":
    main()
