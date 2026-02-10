// COGNETS Swarm Stack: NCE & NWO Demo Server
// Express + WebSocket server that runs the actual demo flow

const express = require('express');
const http = require('http');
const WebSocket = require('ws');
const path = require('path');

const NDS = require('./lib/nds');
const NDM = require('./lib/ndm');
const NCE = require('./lib/nce');
const NWO = require('./lib/nwo');

const app = express();
const server = http.createServer(app);
const wss = new WebSocket.Server({ server });

app.use(express.static(path.join(__dirname, 'public')));

// Broadcast to all connected clients
function broadcast(msg) {
  const data = JSON.stringify(msg);
  wss.clients.forEach(c => { if (c.readyState === WebSocket.OPEN) c.send(data); });
}

// Delay helper
const delay = (ms) => new Promise(r => setTimeout(r, ms));

// The demo flow — mirrors demo_flow.py exactly
async function runDemo(ws, options = {}) {
  const speed = options.speed || 1;
  const stepMode = options.stepMode || false;
  let stepIndex = 0;
  let waitingForNext = false;
  let resolveNext = null;

  // If step mode, we need to wait for "next" messages
  const waitForNext = () => new Promise(resolve => {
    waitingForNext = true;
    resolveNext = resolve;
    broadcast({ type: 'waitingForNext', step: stepIndex });
  });

  // Listen for step-mode "next" commands on this ws
  const onMessage = (raw) => {
    try {
      const msg = JSON.parse(raw);
      if (msg.type === 'next' && resolveNext) {
        waitingForNext = false;
        resolveNext();
        resolveNext = null;
      }
    } catch (_) {}
  };
  ws.on('message', onMessage);

  const d = (ms) => delay(ms / speed);
  const step = async (phase, component, action, log, data, cmdOutput) => {
    stepIndex++;
    broadcast({ type: 'step', step: stepIndex, phase, component, action, log, data: data || null, cmdOutput: cmdOutput || null });
    if (stepMode) await waitForNext();
    else await d(800);
  };

  // Collect events from components
  const events = [];
  const captureEvent = (evt) => events.push(evt);

  try {
    // === Phase 0: Initialization ===
    broadcast({ type: 'phase', phase: 0, label: 'Phase 0: Initialization' });
    await d(500);

    const nds = new NDS();
    nds.on('action', captureEvent);
    await step(0, 'NDS', 'init', '[NDS] Initialized. Local storage ready.', { resources: nds.resources, sensorKeys: Object.keys(nds.sensorData) });

    const ndm = new NDM(nds);
    ndm.on('action', captureEvent);
    await step(0, 'NDM', 'init', '[NDM] Initialized. Data integrity module ready.', null);

    const nce = new NCE(nds);
    nce.on('action', captureEvent);
    await step(0, 'NCE', 'init', '[NCE] Initialized. Component executor ready.', null);

    const nwo = new NWO(nce, nds);
    nwo.on('action', captureEvent);
    await step(0, 'NWO', 'init', '[NWO] Initialized. Swarm-coordinated orchestrator ready.', null);

    const baselineNodes = [
      { node_id: 'edge-node-04', cpu: 4, gpu: true, load: 0.45, latency_ms: 80, recent_jobs: 0 }
    ];
    const swarmNodes = [
      { node_id: 'edge-node-01', cpu: 2, gpu: false, load: 0.6, latency_ms: 85, recent_jobs: 3 },
      { node_id: 'edge-node-02', cpu: 4, gpu: true, load: 0.7, latency_ms: 60, recent_jobs: 6 },
      { node_id: 'edge-node-03', cpu: 8, gpu: true, load: 0.2, latency_ms: 30, recent_jobs: 9 },
      { node_id: 'edge-node-04', cpu: 4, gpu: true, load: 0.5, latency_ms: 75, recent_jobs: 1 }
    ];
    const baseAdvantage = { latency_reduction_pct: 0, throughput_gain_x: 1.0, cost_reduction_pct: 0 };
    const swarmAdvantage = { latency_reduction_pct: 20, throughput_gain_x: 1.9, cost_reduction_pct: 22 };

    const executeRun = async (opts) => {
      const runLabel = opts.runLabel;
      const nodes = opts.nodes;
      const policy = opts.policy;
      const advantage = opts.advantage;
      const isSwarm = policy === 'fairness';

      broadcast({ type: 'run', label: runLabel, resetPhases: !!opts.resetPhases });
      await d(400);

      // === Phase 1: Orchestration — DSM → NWO → NDS → NCE → K8s ===
      broadcast({ type: 'phase', phase: 1, label: `${runLabel} — Phase 1: Orchestration` });
      await d(600);

      const swarmEvent = {
        specversion: '1.0',
        type: 'com.cognets.swarm.execute',
        source: 'urn:ngsi-ld:DSM:001',
        data: {
          serviceId: 'urn:ngsi-ld:AIModel:TrafficV2',
          swarm: { nodes },
          parameters: {
            image: 'registry/model:v2',
            env: { MODEL_TYPE: 'TRAFFIC', INPUT_VIDEO: 'assets/traffic_video.mp4' },
            requirements: { gpu: true },
            execution_strategy: 'local-k8s'
          }
        }
      };

      // M6: DSM → NWO
      await step(1, 'NWO', 'receiveSwarm', isSwarm
        ? '[NWO][M6] ⚡ Swarm execution request received from DSM — com.cognets.swarm.execute'
        : '[NWO][M6] Baseline execution request received (single-node).', swarmEvent);
      await step(1, 'NWO', 'swarmContext', isSwarm
        ? '[NWO] Swarm Intelligence: parsing workload requirements...'
        : '[NWO] Baseline path: local execution plan (no swarm optimization).', null);

      const schedule = nwo.scheduleNodes(swarmEvent.data.swarm.nodes, swarmEvent.data.parameters.requirements);
      schedule.policy = policy;
      schedule.advantage = advantage;
      await step(1, 'NWO', 'schedule', isSwarm
        ? `[NWO] Multi-node scheduling complete — ${schedule.selected.node_id} selected (fairness-optimized).`
        : `[NWO] Single-node execution selected — ${schedule.selected.node_id} (baseline).`, schedule);

      await step(1, 'NWO', 'planExecution', '[NWO] Execution strategy selected: local-k8s', {
        service_id: swarmEvent.data.serviceId,
        target_node: schedule.selected.node_id,
        strategy: swarmEvent.data.parameters.execution_strategy,
        policy,
        input_video: swarmEvent.data.parameters.env.INPUT_VIDEO
      });

      // NWO checks resources via NDS
      const requirements = swarmEvent.data.parameters.requirements;
      await step(1, 'NDS', 'checkResources', `[NDS] Checking resources against requirements: ${JSON.stringify(requirements)}`, { requirements, available: nds.resources });

      nds.checkResources(requirements);
      await step(1, 'NDS', 'checkResourcesOK', '[NDS] Resources OK — all requirements satisfied.', { status: 'OK', available: nds.resources });

      // Create ticket
      const crypto = require('crypto');
      const jobId = `job-${crypto.randomBytes(4).toString('hex')}`;
      const allocation = nds.reserveResources(jobId, requirements);
      const ticket = {
        job_id: jobId,
        service_id: swarmEvent.data.serviceId,
        image: swarmEvent.data.parameters.image,
        env_vars: swarmEvent.data.parameters.env,
        volume_mounts: null,
        allocation,
        execution_strategy: swarmEvent.data.parameters.execution_strategy,
        target_node: schedule.selected.node_id,
        schedule
      };
      await step(1, 'NWO', 'ticketCreated', `[NWO] Execution ticket created: ${jobId}`, ticket);

      // M7: NWO → NCE
      await step(1, 'NWO', 'dispatch', `[NWO][M7] Dispatching execution ticket to NCE (target: ${schedule.selected.node_id})`, { job_id: jobId });

      // NCE generates manifest
      const manifest = nce.generateManifest(ticket);
      await step(1, 'NCE', 'generateManifest', '[NCE] Generating Kubernetes Job Manifest...', manifest);

      // M8: NCE applies manifest (real file I/O + subprocess)
      const applyResult = nce.applyManifest(manifest);
      await step(1, 'NCE', 'applyManifest', `[NCE][M8] kubectl apply -f ${jobId}.yaml`, { command: applyResult.command }, applyResult.output);

      // Update status
      nds.updateStatus(jobId, 'RUNNING');
      await step(1, 'NDS', 'updateStatus', `[NDS] Status updated for ${jobId}: RUNNING`, { jobId, status: 'RUNNING' });

      // === Phase 3: Autonomous Runtime ===
      broadcast({ type: 'phase', phase: 3, label: `${runLabel} — Phase 3: Autonomous Runtime` });
      await d(800);

      // NCE monitors job
      await step(3, 'NCE', 'monitor', `[NCE] Monitoring job ${jobId} on ${schedule.selected.node_id}...`, { job_id: jobId, target_node: schedule.selected.node_id, status: 'RUNNING' });

      // M8*: Model requests sensor data from NDS
      await step(3, 'Model', 'requestData', '[Model] Requesting Sensor Data from NDS (M8*)...', { query: 'urn:ngsi-ld:Sensor:Camera01', interface: 'NGSI-LD' });

      const sensorData = nds.getSensorData('urn:ngsi-ld:Sensor:Camera01');
      await step(3, 'NDS', 'sensorDataReturned', `[NDS] Returning sensor data to Model — ${Object.keys(sensorData.data).length} fields.`, sensorData);

      // M9: Model inference
      await step(3, 'Model', 'inference', '[Model] Executing inference (M9)...', null);
      await d(1200);
      const inferenceResult = { class: 'car', confidence: 0.98, bbox: [10, 20, 100, 200] };
      await step(3, 'Model', 'inferenceResult', '[Model] Inference complete — detected: car (98% confidence)', inferenceResult);

      await step(3, 'Model', 'processedVideo', '[Model] Video processed (YOLOv8) → assets/traffic_video_processed.mp4', {
        output_video: 'assets/traffic_video_processed.mp4',
        model: 'yolov8n.pt'
      });

      await step(3, 'Model', 'pushResult', '[Model] Pushing result to NDM (M10)...', inferenceResult);

      const signedPayload = ndm.processOutput(inferenceResult, jobId);
      const publishPayload = {
        data: signedPayload.data || inferenceResult,
        timestamp: signedPayload.timestamp,
        channel: 'ngsi-ld'
      };
      await step(3, 'NDM', 'publish', '[NDM] Publishing result payload (M11).', publishPayload);

      // === Phase 4: Completion ===
      broadcast({ type: 'phase', phase: 4, label: `${runLabel} — Phase 4: Completion` });
      await d(600);

      await step(4, 'NCE', 'complete', `[NCE] Job ${jobId} completed successfully.`, { job_id: jobId, status: 'COMPLETED' });

      nce.cleanup(jobId);
      await step(4, 'NCE', 'cleanup', `[NCE] Cleaning up resources for job: ${jobId}`, { jobId });
      await step(4, 'NDS', 'updateStatus', `[NDS] Status updated for ${jobId}: CLEANED`, { jobId, status: 'CLEANED' });
    };

    await executeRun({
      runLabel: 'Run 1: Single Node (Baseline)',
      nodes: baselineNodes,
      policy: 'single-node',
      advantage: baseAdvantage,
      resetPhases: false
    });

    await executeRun({
      runLabel: 'Run 2: Swarm Enabled (Fairness)',
      nodes: swarmNodes,
      policy: 'fairness',
      advantage: swarmAdvantage,
      resetPhases: true
    });

    broadcast({ type: 'done', message: 'Demo completed successfully.' });
  } catch (err) {
    broadcast({ type: 'error', message: err.message });
  }

  ws.removeListener('message', onMessage);
}

// WebSocket connection handling
wss.on('connection', (ws) => {
  console.log('[WS] Client connected');
  ws.send(JSON.stringify({ type: 'connected', message: 'COGNETS Swarm Stack Demo — connected' }));

  ws.on('message', (raw) => {
    try {
      const msg = JSON.parse(raw);
      if (msg.type === 'start') {
        console.log('[WS] Starting demo flow...');
        runDemo(ws, { speed: msg.speed || 1, stepMode: msg.stepMode || false });
      }
    } catch (_) {}
  });

  ws.on('close', () => console.log('[WS] Client disconnected'));
});

const PORT = 3456;
server.listen(PORT, () => {
  console.log(`\n  ╔══════════════════════════════════════════╗`);
  console.log(`  ║  COGNETS Swarm Stack: NCE & NWO Demo     ║`);
  console.log(`  ║  http://localhost:${PORT}                   ║`);
  console.log(`  ╚══════════════════════════════════════════╝\n`);
});
