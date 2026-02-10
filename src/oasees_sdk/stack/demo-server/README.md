# COGNETS Swarm Stack: NCE & NWO Live Demo

Interactive dashboard showcasing the COGNETS Node Component Executor (NCE) and Node Workload Orchestrator (NWO) executing a swarm-coordinated AI workload.

## Quick Start

```bash
cd demo-server
npm install
npm start
# Open http://localhost:3456
```

## Architecture

- **Backend**: Express + WebSocket server running real NCE/NWO/NDS/NDM logic
- **Frontend**: Single-page dark dashboard with live WebSocket streaming
- **Components**: NWO (orchestrator), NCE (executor), NDS (storage), NDM (data manager)

## Features

- Real JS class execution (not just animations)
- Real file I/O for K8s manifests, real SHA-256 signing
- kubectl dry-run with graceful fallback
- Step-by-step or auto-play mode with adjustable speed
- Live architecture diagram with animated data flow
- Swarm topology visualization
