#!/usr/bin/env python3
"""Generate NCE & NWO Technical Report as .docx"""

from docx import Document
from docx.shared import Inches, Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.section import WD_ORIENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
import os

doc = Document()

# ── Styles ──────────────────────────────────────────────────────────
style = doc.styles['Normal']
style.font.name = 'Calibri'
style.font.size = Pt(11)
style.paragraph_format.space_after = Pt(6)

BLUE = RGBColor(0x1F, 0x49, 0x7D)
DARK = RGBColor(0x33, 0x33, 0x33)

for level in range(1, 4):
    h = doc.styles[f'Heading {level}']
    h.font.color.rgb = BLUE
    h.font.name = 'Calibri'

code_style = doc.styles.add_style('CodeBlock', 1)  # paragraph style
code_style.font.name = 'Courier New'
code_style.font.size = Pt(9)
code_style.paragraph_format.space_before = Pt(4)
code_style.paragraph_format.space_after = Pt(4)


def add_code(text):
    for line in text.strip().split('\n'):
        doc.add_paragraph(line, style='CodeBlock')

def add_table(headers, rows):
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = 'Light Grid Accent 1'
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, h in enumerate(headers):
        cell = t.rows[0].cells[i]
        cell.text = h
        for p in cell.paragraphs:
            for r in p.runs:
                r.bold = True
    for row_data in rows:
        row = t.add_row()
        for i, val in enumerate(row_data):
            row.cells[i].text = str(val)
    doc.add_paragraph()

def page_break():
    doc.add_page_break()


# ══════════════════════════════════════════════════════════════════════
# 1. TITLE PAGE
# ══════════════════════════════════════════════════════════════════════
for _ in range(6):
    doc.add_paragraph()

title = doc.add_paragraph()
title.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = title.add_run('OASEES SDK')
run.bold = True
run.font.size = Pt(36)
run.font.color.rgb = BLUE

subtitle = doc.add_paragraph()
subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = subtitle.add_run('Node Component Executor (NCE) &\nNode Workload Orchestrator (NWO)')
run.font.size = Pt(22)
run.font.color.rgb = BLUE

sub2 = doc.add_paragraph()
sub2.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = sub2.add_run('Technical Report')
run.font.size = Pt(18)
run.font.color.rgb = DARK

for _ in range(3):
    doc.add_paragraph()

meta = doc.add_paragraph()
meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = meta.add_run('Version 1.0  •  February 2026')
run.font.size = Pt(14)
run.font.color.rgb = DARK

meta2 = doc.add_paragraph()
meta2.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = meta2.add_run('OASEES / COGNETS Consortium')
run.font.size = Pt(12)
run.font.color.rgb = DARK

page_break()

# ══════════════════════════════════════════════════════════════════════
# TABLE OF CONTENTS
# ══════════════════════════════════════════════════════════════════════
doc.add_heading('Table of Contents', level=1)
toc_items = [
    '1. Executive Summary',
    '2. Architecture Overview',
    '   2.1 System Architecture',
    '   2.2 Component Diagram',
    '   2.3 Interface Mapping',
    '3. Phase Workflows',
    '   3.1 Phase 0: Initialization',
    '   3.2 Phase 1: Orchestration (M6 → M7 → M8)',
    '   3.3 Phase 3: Autonomous Runtime (M8* → M9 → M10/M11)',
    '   3.4 Phase 4: Integrity & Completion',
    '4. API Reference',
    '   4.1 NWO — Node Workload Orchestrator',
    '   4.2 NCE — Node Component Executor',
    '   4.3 NDS — Node Device Storage',
    '   4.4 NDM — Node Data Manager',
    '5. Data Structures',
    '6. Deployment Guide',
    '   6.1 Prerequisites',
    '   6.2 Installation',
    '   6.3 Running the Demo',
    '   6.4 Deploying NCE Independently',
    '   6.5 Deploying NWO Independently',
    '   6.6 Production Considerations',
    '7. Sequence Diagram',
    '8. Appendix',
    '   8.1 Full Source Code',
    '   8.2 Glossary',
]
for item in toc_items:
    p = doc.add_paragraph(item)
    p.paragraph_format.space_after = Pt(2)

page_break()

# ══════════════════════════════════════════════════════════════════════
# 2. EXECUTIVE SUMMARY
# ══════════════════════════════════════════════════════════════════════
doc.add_heading('1. Executive Summary', level=1)
doc.add_paragraph(
    'The OASEES SDK provides a modular, event-driven software stack for '
    'orchestrating AI/ML workloads on edge and fog nodes within the OASEES '
    '(Open AI-driven Service Enabling Edge Systems) ecosystem. This report '
    'focuses on two core components:'
)
doc.add_paragraph(
    'Node Workload Orchestrator (NWO) — the local orchestration layer that '
    'receives swarm-level execution directives via CloudEvents (Interface M6), '
    'performs resource validation through the Node Device Storage (NDS), and '
    'dispatches fully-resolved Execution Tickets to the Node Component Executor.',
    style='List Bullet'
)
doc.add_paragraph(
    'Node Component Executor (NCE) — the execution engine that translates '
    'Execution Tickets into Kubernetes Job manifests and applies them to the '
    'local cluster (Interfaces M7/M8). NCE also manages the lifecycle of '
    'running workloads, including monitoring and cleanup.',
    style='List Bullet'
)
doc.add_paragraph(
    'Together with the Node Device Storage (NDS) for state and sensor data, '
    'and the Node Data Manager (NDM) for integrity-signed output publication, '
    'these four components form the complete node-level runtime stack. The '
    'current implementation is a mock/prototype that demonstrates the full '
    'message flow from swarm notification to signed inference output, suitable '
    'for integration testing and architectural validation.'
)

page_break()

# ══════════════════════════════════════════════════════════════════════
# 3. ARCHITECTURE OVERVIEW
# ══════════════════════════════════════════════════════════════════════
doc.add_heading('2. Architecture Overview', level=1)

doc.add_heading('2.1 System Architecture', level=2)
doc.add_paragraph(
    'The OASEES node-level stack follows a layered, dependency-injected '
    'architecture. An external Decentralized Swarm Manager (DSM) sends '
    'CloudEvent notifications to the NWO, which acts as the local entry point. '
    'NWO validates resources via NDS, constructs an Execution Ticket, and '
    'forwards it to the NCE for K8s-based execution. Once the workload (an '
    'AI/ML model container) is running, it autonomously queries NDS for sensor '
    'data, performs inference, and publishes signed results through NDM.'
)

doc.add_heading('2.2 Component Diagram', level=2)
doc.add_paragraph(
    'The system consists of four primary components interconnected through '
    'well-defined interfaces:'
)
add_table(
    ['Component', 'Role', 'Key Responsibility'],
    [
        ['NWO', 'Orchestrator', 'Receives swarm directives (M6), checks resources via NDS, dispatches tickets to NCE (M7)'],
        ['NCE', 'Executor', 'Accepts tickets (M7), generates K8s manifests, applies them (M8), manages lifecycle'],
        ['NDS', 'State & Sensors', 'Resource checking, status tracking, sensor data provision (NGSI-LD simulation)'],
        ['NDM', 'Data Integrity', 'Signs inference outputs, publishes results with integrity proofs (M10/M11)'],
    ]
)

doc.add_heading('2.3 Interface Mapping', level=2)
doc.add_paragraph(
    'The M-interfaces define the communication contracts between components:'
)
add_table(
    ['Interface', 'From → To', 'Description', 'Protocol'],
    [
        ['M6', 'DSM → NWO', 'Swarm Execution Trigger (CloudEvent)', 'CloudEvents/JSON'],
        ['M7', 'NWO → NCE', 'Execution Ticket Dispatch', 'Internal method call'],
        ['M8', 'NCE → K8s', 'Manifest Application (kubectl apply)', 'Kubernetes API'],
        ['M8*', 'Model → NDS', 'Sensor Data Query (NGSI-LD)', 'NGSI-LD / Internal'],
        ['M9', 'Model (internal)', 'Inference Execution', 'In-container'],
        ['M10', 'Model → NDM', 'Result Push', 'Internal method call'],
        ['M11', 'NDM → Network', 'Signed Result Publication (NGSI-LD Upsert)', 'NGSI-LD / HTTP'],
    ]
)

page_break()

# ══════════════════════════════════════════════════════════════════════
# 4. PHASE WORKFLOWS
# ══════════════════════════════════════════════════════════════════════
doc.add_heading('3. Phase Workflows', level=1)

# Phase 0
doc.add_heading('3.1 Phase 0: Initialization', level=2)
doc.add_paragraph(
    'All four components are instantiated using a dependency injection pattern. '
    'NDS is created first (no dependencies), followed by NDM (standalone), then '
    'NCE (depends on NDS for status updates), and finally NWO (depends on NCE '
    'and NDS). This ordering ensures all downstream references are available '
    'before orchestration begins.'
)
add_code('''nds = NDS()          # Standalone – manages state & sensors
ndm = NDM()          # Standalone – handles signing & publishing
nce = NCE(nds)       # Depends on NDS for status tracking
nwo = NWO(nce, nds)  # Depends on NCE (dispatch) and NDS (resource check)''')

# Phase 1
doc.add_heading('3.2 Phase 1: Orchestration (M6 → M7 → M8)', level=2)
doc.add_paragraph('This phase covers the complete path from swarm directive to running workload:')
steps = [
    'DSM sends a CloudEvent to NWO via handle_swarm_notification() [M6].',
    'NWO extracts the service ID, image, environment variables, and resource requirements from the event payload.',
    'NWO calls NDS.check_resources() to verify the node can satisfy requirements (e.g., GPU availability).',
    'If resources are insufficient, NWO returns a FAILED status immediately.',
    'NWO constructs an Execution Ticket with a unique job ID (uuid4-based), service ID, image, env vars, and volume mounts.',
    'NWO dispatches the ticket to NCE via nce.submit_ticket() [M7].',
    'NCE generates a Kubernetes Job manifest from the ticket via generate_manifest().',
    'NCE applies the manifest via apply_manifest() [M8] — in the mock, this prints the kubectl command.',
    'NCE updates the job status to RUNNING via NDS.update_status().',
    'NWO returns an ACCEPTED response with the job ID to the caller.',
]
for i, s in enumerate(steps, 1):
    doc.add_paragraph(f'{i}. {s}')

doc.add_heading('CloudEvent Payload Structure', level=3)
add_code('''{
    "specversion": "1.0",
    "type": "com.cognets.swarm.execute",
    "source": "urn:ngsi-ld:DSM:001",
    "data": {
        "serviceId": "urn:ngsi-ld:AIModel:TrafficV2",
        "parameters": {
            "image": "registry/model:v2",
            "env": {"MODEL_TYPE": "TRAFFIC"},
            "requirements": {"gpu": true}
        }
    }
}''')

doc.add_heading('Execution Ticket Structure', level=3)
add_code('''{
    "job_id": "job-a1b2c3d4",
    "service_id": "urn:ngsi-ld:AIModel:TrafficV2",
    "image": "registry/model:v2",
    "env_vars": {"MODEL_TYPE": "TRAFFIC"},
    "volume_mounts": null
}''')

doc.add_heading('Kubernetes Job Manifest', level=3)
add_code('''{
    "apiVersion": "batch/v1",
    "kind": "Job",
    "metadata": {"name": "job-a1b2c3d4"},
    "spec": {
        "template": {
            "spec": {
                "containers": [{
                    "name": "workload",
                    "image": "registry/model:v2",
                    "env": {"MODEL_TYPE": "TRAFFIC"}
                }],
                "restartPolicy": "Never"
            }
        }
    }
}''')

# Phase 3
doc.add_heading('3.3 Phase 3: Autonomous Runtime (M8* → M9 → M10/M11)', level=2)
doc.add_paragraph(
    'Once the model container is running inside Kubernetes, it operates autonomously:'
)
doc.add_paragraph('1. The model queries NDS for sensor data via get_sensor_data() [M8*]. '
                   'NDS returns a simulated NGSI-LD response containing temperature, humidity, '
                   'and camera feed data with a timestamp.')
doc.add_paragraph('2. The model executes inference [M9] — in the demo this is simulated as an '
                   'object detection result with class, confidence, and bounding box.')
doc.add_paragraph('3. The model pushes the inference result to NDM via process_output() [M10]. '
                   'NDM signs the data using SHA-256 hashing, attaches a timestamp and integrity '
                   'proof, and publishes the signed payload [M11].')

# Phase 4
doc.add_heading('3.4 Phase 4: Integrity & Completion', level=2)
doc.add_paragraph(
    'After the workload completes, NCE performs cleanup by calling cleanup(job_id), '
    'which updates the job status to CLEANED via NDS. In a production environment, '
    'this phase would also delete the Kubernetes Job and associated pods, release '
    'GPU allocations, and archive logs.'
)
doc.add_paragraph('Status transitions: ACCEPTED → RUNNING → CLEANED')

page_break()

# ══════════════════════════════════════════════════════════════════════
# 5. API REFERENCE
# ══════════════════════════════════════════════════════════════════════
doc.add_heading('4. API Reference', level=1)

# NWO
doc.add_heading('4.1 NWO — Node Workload Orchestrator', level=2)

doc.add_heading('__init__(self, nce: NCE, nds: NDS)', level=3)
add_table(['Parameter', 'Type', 'Description'], [
    ['nce', 'NCE', 'Node Component Executor instance for ticket dispatch'],
    ['nds', 'NDS', 'Node Device Storage instance for resource checks'],
])
doc.add_paragraph('Returns: None. Interface: Constructor.')

doc.add_heading('handle_swarm_notification(self, payload) → dict', level=3)
add_table(['Parameter', 'Type', 'Description'], [
    ['payload', 'dict', 'CloudEvent/JSON with specversion, type, source, and data fields'],
])
doc.add_paragraph('Returns: dict with "status" ("ACCEPTED" or "FAILED") and optionally "job_id" or "reason".')
doc.add_paragraph('Interface: M6 (Swarm Execution Trigger).')
doc.add_paragraph('Example:')
add_code('''response = nwo.handle_swarm_notification(cloud_event)
# {"status": "ACCEPTED", "job_id": "job-a1b2c3d4"}''')

# NCE
doc.add_heading('4.2 NCE — Node Component Executor', level=2)

doc.add_heading('__init__(self, nds)', level=3)
add_table(['Parameter', 'Type', 'Description'], [
    ['nds', 'NDS', 'Node Device Storage instance for status tracking'],
])

doc.add_heading('submit_ticket(self, ticket) → str', level=3)
add_table(['Parameter', 'Type', 'Description'], [
    ['ticket', 'dict', 'Execution Ticket with job_id, service_id, image, env_vars, volume_mounts'],
])
doc.add_paragraph('Returns: str — the job ID. Interface: M7.')

doc.add_heading('generate_manifest(self, ticket) → dict', level=3)
add_table(['Parameter', 'Type', 'Description'], [
    ['ticket', 'dict', 'Execution Ticket'],
])
doc.add_paragraph('Returns: dict — Kubernetes Job manifest. Interface: Internal.')

doc.add_heading('apply_manifest(self, manifest) → bool', level=3)
add_table(['Parameter', 'Type', 'Description'], [
    ['manifest', 'dict', 'K8s Job manifest dictionary'],
])
doc.add_paragraph('Returns: bool — True if successful. Interface: M8.')

doc.add_heading('cleanup(self, job_id) → None', level=3)
add_table(['Parameter', 'Type', 'Description'], [
    ['job_id', 'str', 'The job identifier to clean up'],
])
doc.add_paragraph('Returns: None. Updates status to CLEANED via NDS.')

# NDS
doc.add_heading('4.3 NDS — Node Device Storage', level=2)

doc.add_heading('__init__(self)', level=3)
doc.add_paragraph('No parameters. Initializes default resources (4 CPU, 16Gi memory, GPU=True) and mock sensor data.')

doc.add_heading('check_resources(self, requirements=None) → bool', level=3)
add_table(['Parameter', 'Type', 'Description'], [
    ['requirements', 'dict or None', 'Resource requirements (e.g., {"gpu": True})'],
])
doc.add_paragraph('Returns: bool. Interface: Called by NWO during M6 processing.')

doc.add_heading('update_status(self, execution_id, status) → None', level=3)
add_table(['Parameter', 'Type', 'Description'], [
    ['execution_id', 'str', 'Job identifier'],
    ['status', 'str', 'New status string (e.g., RUNNING, CLEANED)'],
])

doc.add_heading('get_status(self, execution_id) → str', level=3)
doc.add_paragraph('Returns the current status for the given execution ID, or "UNKNOWN".')

doc.add_heading('get_sensor_data(self, query) → dict', level=3)
add_table(['Parameter', 'Type', 'Description'], [
    ['query', 'str', 'NGSI-LD entity identifier (e.g., "urn:ngsi-ld:Sensor:Camera01")'],
])
doc.add_paragraph('Returns: dict with type, data, and timestamp. Interface: M8*.')

# NDM
doc.add_heading('4.4 NDM — Node Data Manager', level=2)

doc.add_heading('__init__(self)', level=3)
doc.add_paragraph('No parameters.')

doc.add_heading('sign_data(self, data) → str', level=3)
add_table(['Parameter', 'Type', 'Description'], [
    ['data', 'dict', 'Data payload to sign'],
])
doc.add_paragraph('Returns: str — SHA-256 hex digest of JSON-serialized data.')

doc.add_heading('process_output(self, data) → dict', level=3)
add_table(['Parameter', 'Type', 'Description'], [
    ['data', 'dict', 'Inference result to sign and publish'],
])
doc.add_paragraph('Returns: dict with data, signature, timestamp, integrity_proof. Interface: M10/M11.')

page_break()

# ══════════════════════════════════════════════════════════════════════
# 6. DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════════
doc.add_heading('5. Data Structures', level=1)

doc.add_heading('5.1 CloudEvent (Swarm Notification)', level=2)
add_table(['Field', 'Type', 'Description'], [
    ['specversion', 'str', 'CloudEvents spec version (always "1.0")'],
    ['type', 'str', 'Event type (e.g., "com.cognets.swarm.execute")'],
    ['source', 'str', 'NGSI-LD URN of the originating DSM'],
    ['data.serviceId', 'str', 'NGSI-LD URN of the AI model/service to deploy'],
    ['data.parameters.image', 'str', 'Container image reference'],
    ['data.parameters.env', 'dict', 'Environment variables for the container'],
    ['data.parameters.requirements', 'dict', 'Resource requirements (gpu, cpu, memory)'],
])

doc.add_heading('5.2 Execution Ticket', level=2)
add_table(['Field', 'Type', 'Description'], [
    ['job_id', 'str', 'Unique job identifier (job-<uuid8>)'],
    ['service_id', 'str', 'NGSI-LD service URN'],
    ['image', 'str', 'Container image to run'],
    ['env_vars', 'dict', 'Environment variables'],
    ['volume_mounts', 'list or None', 'Volume mount specifications'],
])

doc.add_heading('5.3 Kubernetes Job Manifest', level=2)
add_table(['Field', 'Type', 'Description'], [
    ['apiVersion', 'str', '"batch/v1"'],
    ['kind', 'str', '"Job"'],
    ['metadata.name', 'str', 'Job ID from ticket'],
    ['spec.template.spec.containers[0].name', 'str', '"workload"'],
    ['spec.template.spec.containers[0].image', 'str', 'Container image from ticket'],
    ['spec.template.spec.containers[0].env', 'dict/list', 'Environment variables from ticket'],
    ['spec.template.spec.restartPolicy', 'str', '"Never"'],
])

doc.add_heading('5.4 Sensor Data Response', level=2)
add_table(['Field', 'Type', 'Description'], [
    ['type', 'str', '"SensorReading"'],
    ['data.temperature', 'float', 'Temperature in Celsius'],
    ['data.humidity', 'float', 'Relative humidity percentage'],
    ['data.camera_feed', 'str', 'Base64-encoded image data'],
    ['timestamp', 'float', 'Unix timestamp of reading'],
])

doc.add_heading('5.5 Signed Output Payload', level=2)
add_table(['Field', 'Type', 'Description'], [
    ['data', 'dict', 'Original inference result'],
    ['signature', 'str', 'SHA-256 hex digest of the data'],
    ['timestamp', 'float', 'Unix timestamp of signing'],
    ['integrity_proof', 'str', 'Proof identifier (proof-<sig8>)'],
])

page_break()

# ══════════════════════════════════════════════════════════════════════
# 7. DEPLOYMENT GUIDE
# ══════════════════════════════════════════════════════════════════════
doc.add_heading('6. Deployment Guide', level=1)

doc.add_heading('6.1 Prerequisites', level=2)
doc.add_paragraph('• Python 3.10 or higher', style='List Bullet')
doc.add_paragraph('• pip (or poetry for dependency management)', style='List Bullet')
doc.add_paragraph('• Kubernetes cluster (for production deployments — minikube or k3s for development)', style='List Bullet')
doc.add_paragraph('• kubectl configured with cluster access', style='List Bullet')

doc.add_heading('6.2 Installation', level=2)
add_code('''# Clone the repository
git clone https://github.com/oasees/oasees-sdk.git
cd oasees-sdk

# Install in development mode
pip install -e .

# Or with poetry
poetry install''')
doc.add_paragraph('Project structure:')
add_code('''oasees-sdk/
└── src/
    └── oasees_sdk/
        └── stack/
            ├── __init__.py
            ├── nwo.py         # Node Workload Orchestrator
            ├── nce.py         # Node Component Executor
            ├── nds.py         # Node Device Storage
            ├── ndm.py         # Node Data Manager
            └── demo_flow.py   # End-to-end demonstration''')

doc.add_heading('6.3 Running the Demo', level=2)
add_code('''python -m oasees_sdk.stack.demo_flow''')
doc.add_paragraph('Expected output:')
add_code('''=== OASEES SDK: NCE & NWO Showcase Demo ===

--- [Phase 0] Initialization ---
[NDS] Initialized.
[NDM] Initialized.
[NCE] Initialized.
[NWO] Initialized.

--- [Phase 1: Orchestration] DSM -> NWO (M6) ---
[NWO][M6] Received Swarm Notification: com.cognets.swarm.execute
[NDS] Checking resources against requirements: {'gpu': True}
[NDS] Resources OK.
[NWO][M7] Dispatching Execution Ticket to NCE
[NCE] Received Execution Ticket: job-xxxxxxxx
[NCE] Generating Kubernetes Job Manifest...
[NCE] Applying Manifest: job-xxxxxxxx
[NCE][MOCK] execute: kubectl apply -f job-xxxxxxxx.yaml
[NDS] Status updated for job-xxxxxxxx: RUNNING
[Result] NWO Response: {'status': 'ACCEPTED', 'job_id': 'job-xxxxxxxx'}

--- [Phase 3: Autonomous Runtime] Model Interaction ---
[Model] Requesting Sensor Data from NDS (M8*)...
[NDS] Received sensor data query: urn:ngsi-ld:Sensor:Camera01
[Model] Received Data: dict_keys(['temperature', 'humidity', 'camera_feed'])
[Model] executing inference...
[Model] Pushing result to NDM (M10)...
[NDM] Processing output data: {'class': 'car', 'confidence': 0.98, ...}
[NDM] Publishing signed result: {"data": ..., "signature": "...", ...}

--- [Phase 4: Integrity & Completion] ---
[System] Job job-xxxxxxxx completed successfully.
[NCE] Cleaning up resources for job: job-xxxxxxxx
[NDS] Status updated for job-xxxxxxxx: CLEANED

=== Demo Completed ===''')

doc.add_heading('6.4 Deploying NCE Independently', level=2)
doc.add_paragraph(
    'NCE can be used as a standalone executor component in any Python application '
    'that needs to submit workloads to a Kubernetes cluster:'
)
add_code('''from oasees_sdk.stack.nds import NDS
from oasees_sdk.stack.nce import NCE

# Initialize
nds = NDS()
nce = NCE(nds)

# Submit a ticket programmatically
ticket = {
    "job_id": "custom-job-001",
    "service_id": "my-service",
    "image": "my-registry/my-model:latest",
    "env_vars": [{"name": "BATCH_SIZE", "value": "32"}],
    "volume_mounts": None
}

job_id = nce.submit_ticket(ticket)
print(f"Submitted: {job_id}")
print(f"Status: {nds.get_status(job_id)}")

# Later: cleanup
nce.cleanup(job_id)''')
doc.add_paragraph(
    'For production, replace the mock apply_manifest() with real subprocess calls to kubectl '
    'or use the official Kubernetes Python client library.'
)

doc.add_heading('6.5 Deploying NWO Independently', level=2)
doc.add_paragraph(
    'NWO can serve as an event-driven orchestrator, accepting CloudEvents from any source:'
)
add_code('''from oasees_sdk.stack.nds import NDS
from oasees_sdk.stack.nce import NCE
from oasees_sdk.stack.nwo import NWO

# Setup
nds = NDS()
nce = NCE(nds)
nwo = NWO(nce, nds)

# Example: Flask-based CloudEvent listener
from flask import Flask, request, jsonify
app = Flask(__name__)

@app.route("/events", methods=["POST"])
def handle_event():
    event = request.get_json()
    result = nwo.handle_swarm_notification(event)
    return jsonify(result)

app.run(host="0.0.0.0", port=8080)''')

doc.add_heading('6.6 Production Considerations', level=2)
considerations = [
    ('Real Kubernetes Integration', 'Replace mock kubectl commands with the kubernetes Python client or real subprocess.run(["kubectl", "apply", "-f", ...]) calls.'),
    ('NGSI-LD Integration', 'Replace NDS mock sensor data with actual NGSI-LD context broker queries (e.g., Scorpio or Orion-LD).'),
    ('Cryptographic Signing', 'Replace SHA-256 hash with proper digital signatures (e.g., Ed25519 or ECDSA with key management).'),
    ('Monitoring & Logging', 'Add structured logging (e.g., structlog), Prometheus metrics, and health check endpoints.'),
    ('Error Handling & Retry', 'Implement exponential backoff for K8s API calls, dead-letter queues for failed events, and circuit breakers.'),
    ('Security', 'Add authentication for the CloudEvent endpoint, RBAC for K8s operations, and secret management for signing keys.'),
]
for title, desc in considerations:
    doc.add_paragraph(f'{title}: {desc}', style='List Bullet')

page_break()

# ══════════════════════════════════════════════════════════════════════
# 8. SEQUENCE DIAGRAM
# ══════════════════════════════════════════════════════════════════════
doc.add_heading('7. Sequence Diagram', level=1)
doc.add_paragraph('The following text-based sequence diagram shows the complete message flow:')
add_code('''  DSM          NWO          NDS          NCE         K8s        Model         NDM
   |            |            |            |            |            |            |
   |---M6------>|            |            |            |            |            |
   |  CloudEvent|            |            |            |            |            |
   |            |---check--->|            |            |            |            |
   |            | resources  |            |            |            |            |
   |            |<--OK-------|            |            |            |            |
   |            |            |            |            |            |            |
   |            |---M7------>|            |            |            |            |
   |            | Exec Ticket|            |            |            |            |
   |            |            |---M8------>|            |            |            |
   |            |            | kubectl    |            |            |            |
   |            |            | apply      |            |            |            |
   |            |            |            |--deploy--->|            |            |
   |            |            |            |            |--run------>|            |
   |            |            |            |            |            |            |
   |            |            |<--------M8*-------------|            |            |
   |            |            | sensor query             |            |            |
   |            |            |----------data----------->|            |            |
   |            |            |            |            |            |            |
   |            |            |            |            |   M9:infer |            |
   |            |            |            |            |            |            |
   |            |            |            |            |            |---M10/M11->|
   |            |            |            |            |            | result     |
   |            |            |            |            |            |<--signed---|
   |            |            |            |            |            |            |
   |            |            |<--status---|            |            |            |
   |            |            | CLEANED    |            |            |            |
   |            |            |            |            |            |            |''')

page_break()

# ══════════════════════════════════════════════════════════════════════
# 9. APPENDIX
# ══════════════════════════════════════════════════════════════════════
doc.add_heading('8. Appendix', level=1)

doc.add_heading('8.1 Full Source Code', level=2)

# NWO source
doc.add_heading('nwo.py — Node Workload Orchestrator', level=3)
with open('/Users/akiskourtis/.openclaw/workspace/oasees-sdk/src/oasees_sdk/stack/nwo.py') as f:
    add_code(f.read())

doc.add_heading('nce.py — Node Component Executor', level=3)
with open('/Users/akiskourtis/.openclaw/workspace/oasees-sdk/src/oasees_sdk/stack/nce.py') as f:
    add_code(f.read())

doc.add_heading('nds.py — Node Device Storage', level=3)
with open('/Users/akiskourtis/.openclaw/workspace/oasees-sdk/src/oasees_sdk/stack/nds.py') as f:
    add_code(f.read())

doc.add_heading('ndm.py — Node Data Manager', level=3)
with open('/Users/akiskourtis/.openclaw/workspace/oasees-sdk/src/oasees_sdk/stack/ndm.py') as f:
    add_code(f.read())

doc.add_heading('demo_flow.py — End-to-End Demo', level=3)
with open('/Users/akiskourtis/.openclaw/workspace/oasees-sdk/src/oasees_sdk/stack/demo_flow.py') as f:
    add_code(f.read())

doc.add_heading('8.2 Glossary', level=2)
add_table(['Term', 'Definition'], [
    ['DSM', 'Decentralized Swarm Manager — external orchestrator that sends execution directives to nodes'],
    ['NWO', 'Node Workload Orchestrator — local orchestration layer that receives and processes swarm events'],
    ['NCE', 'Node Component Executor — execution engine that translates tickets to K8s manifests and applies them'],
    ['NDS', 'Node Device Storage — local state manager providing resource info, status tracking, and sensor data'],
    ['NDM', 'Node Data Manager — ensures data integrity through signing and publishes results'],
    ['NGSI-LD', 'Next Generation Service Interface – Linked Data; a standard API for context information management'],
    ['CloudEvent', 'A specification for describing event data in a common way (CNCF standard)'],
    ['M6–M11', 'Interface identifiers defining communication contracts between OASEES stack components'],
    ['K8s', 'Kubernetes — container orchestration platform used for workload execution'],
    ['OASEES', 'Open AI-driven Service Enabling Edge Systems'],
    ['COGNETS', 'Consortium project context for the OASEES ecosystem'],
])

# ── Save ────────────────────────────────────────────────────────────
output_path = '/Users/akiskourtis/.openclaw/workspace/oasees-sdk/NCE_NWO_Report.docx'
os.makedirs(os.path.dirname(output_path), exist_ok=True)
doc.save(output_path)
print(f"Report saved to {output_path}")
