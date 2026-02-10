import time
import uuid


class NDS:
    """
    Node Device Storage (NDS) - Mock Implementation

    Responsible for:
    1. Managing local state and resources.
    2. Providing sensor data (simulated).
    3. Tracking execution status and telemetry.
    4. Basic policy, trust, and data-access gating.
    """

    def __init__(self):
        self.state = {}
        self.telemetry = {}
        self.results = {}
        self.resources = {
            "cpu": "4",
            "memory": "16Gi",
            "gpu": True
        }
        self.node_profile = {
            "node_id": "edge-node-04",
            "zone": "zone-1",
            "trust_score": 0.92,
            "capabilities": ["gpu", "camera", "ngsi-ld"]
        }
        self.policies = {
            "allowed_sources": {"urn:ngsi-ld:DSM:001"},
            "allowed_services": {
                "urn:ngsi-ld:AIModel:TrafficV2",
                "urn:ngsi-ld:AIModel:OccupancyV1"
            }
        }
        self.data_catalog = {
            "urn:ngsi-ld:Sensor:Camera01": {
                "type": "camera",
                "tags": ["traffic", "road"],
                "scopes": ["traffic"],
                "ttl_s": 5
            },
            "urn:ngsi-ld:Sensor:Weather01": {
                "type": "weather",
                "tags": ["weather", "humidity"],
                "scopes": ["weather"],
                "ttl_s": 15
            }
        }
        self.sensor_data = {
            "temperature": 25.5,
            "humidity": 60.0,
            "camera_feed": "base64_encoded_image_placeholder"
        }
        self.access_tokens = {}
        print("[NDS] Initialized.")

    def get_node_profile(self):
        """
        Return a snapshot of the node profile and capabilities.
        """
        return {
            "node_id": self.node_profile["node_id"],
            "zone": self.node_profile["zone"],
            "trust_score": self.node_profile["trust_score"],
            "capabilities": list(self.node_profile["capabilities"]),
            "resources": dict(self.resources)
        }

    def is_source_trusted(self, source):
        """
        Basic trust gate for DSM sources.
        """
        return source in self.policies.get("allowed_sources", set())

    def is_service_allowed(self, service_id):
        """
        Basic policy gate for allowed services.
        """
        return service_id in self.policies.get("allowed_services", set())

    def check_resources(self, requirements=None):
        """
        Simulate checking if the node has enough resources for a workload.
        """
        print(f"[NDS] Checking resources against requirements: {requirements}")
        # For simulation, assume resources are always sufficient unless specified.
        if requirements and requirements.get("gpu") and not self.resources["gpu"]:
            print("[NDS] Resource check failed: GPU required but not available.")
            return False

        print("[NDS] Resources OK.")
        return True

    def reserve_resources(self, job_id, requirements=None):
        """
        Simulate resource reservation for a job.
        """
        allocation = {
            "cpu": "2",
            "memory": "4Gi",
            "gpu": bool(requirements and requirements.get("gpu"))
        }
        self.state[job_id] = "RESERVED"
        print(f"[NDS] Resources reserved for {job_id}: {allocation}")
        return allocation

    def release_resources(self, job_id):
        """
        Release previously reserved resources.
        """
        print(f"[NDS] Resources released for {job_id}")
        if self.state.get(job_id) == "RESERVED":
            self.state[job_id] = "RELEASED"

    def select_data_plan(self, service_id):
        """
        Choose data sources and scopes based on the requested service.
        """
        if "Traffic" in (service_id or ""):
            primary = "urn:ngsi-ld:Sensor:Camera01"
            scopes = ["traffic"]
        else:
            primary = "urn:ngsi-ld:Sensor:Weather01"
            scopes = ["weather"]
        return {
            "primary": primary,
            "sources": [primary],
            "scopes": scopes
        }

    def issue_access_token(self, job_id, scopes):
        """
        Issue a short-lived access token for data scopes.
        """
        token = uuid.uuid4().hex
        self.access_tokens[token] = {
            "job_id": job_id,
            "scopes": set(scopes or []),
            "issued_at": time.time()
        }
        print(f"[NDS] Access token issued for {job_id}: {token[:8]}...")
        return token

    def validate_access_token(self, token, scope=None):
        meta = self.access_tokens.get(token)
        if not meta:
            return False
        if scope and scope not in meta.get("scopes", set()):
            return False
        return True

    def update_status(self, execution_id, status):
        """
        Log status updates for a specific execution ID.
        """
        self.state[execution_id] = status
        print(f"[NDS] Status updated for {execution_id}: {status}")

    def get_status(self, execution_id):
        return self.state.get(execution_id, "UNKNOWN")

    def get_sensor_data(self, query, access_token=None, scope=None):
        """
        Simulate NGSI-LD query response (Interface M8*).
        """
        print(f"[NDS] Received sensor data query: {query}")
        if access_token:
            if not self.validate_access_token(access_token, scope):
                print("[NDS] Access denied: invalid token or scope.")
                return {
                    "error": "ACCESS_DENIED",
                    "reason": "Invalid token or scope"
                }
        # Simulate data retrieval delay
        time.sleep(0.5)
        return {
            "type": "SensorReading",
            "data": self.sensor_data,
            "timestamp": time.time()
        }

    def record_telemetry(self, job_id, metrics):
        """
        Store telemetry metrics for a running job.
        """
        self.telemetry.setdefault(job_id, []).append({
            "timestamp": time.time(),
            "metrics": metrics
        })
        print(f"[NDS] Telemetry recorded for {job_id}: {metrics}")

    def store_result(self, job_id, payload):
        """
        Store the signed result payload locally.
        """
        self.results[job_id] = payload
        print(f"[NDS] Result stored for {job_id}: integrity_proof={payload.get('integrity_proof')}")
