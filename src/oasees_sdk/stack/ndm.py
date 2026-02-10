
import hashlib
import json
import time

class NDM:
    """
    Node Data Manager (NDM) - Mock Implementation
    
    Responsible for:
    1. Ensuring data integrity (signing).
    2. Publishing results to the network/storage.
    """
    
    def __init__(self, nds=None):
        self.nds = nds
        print("[NDM] Initialized.")

    def sign_data(self, data):
        """
        Simulate data signing using a simple hash.
        """
        data_str = json.dumps(data, sort_keys=True)
        signature = hashlib.sha256(data_str.encode()).hexdigest()
        return signature

    def process_output(self, data, job_id=None):
        """
        Interface M11: Publish Signed Result (NGSI-LD Upsert).
        """
        print(f"[NDM] Processing output data: {data}")
        
        # 1. Sign
        signature = self.sign_data(data)
        
        # 2. Add metadata
        payload = {
            "data": data,
            "signature": signature,
            "timestamp": time.time(),
            "integrity_proof": f"proof-{signature[:8]}"
        }
        
        # 3. "Publish" to NDS or External Network
        print(f"[NDM] Publishing signed result: {json.dumps(payload)}")
        if self.nds and job_id:
            self.nds.store_result(job_id, payload)
        return payload
