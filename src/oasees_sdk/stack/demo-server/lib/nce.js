// NCE - Node Component Executor
// Accepts execution tickets, generates K8s manifests, applies them

const EventEmitter = require('events');
const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');
const os = require('os');
const crypto = require('crypto');

class NCE extends EventEmitter {
  constructor(nds) {
    super();
    this.nds = nds;
    this.emit('action', { component: 'NCE', action: 'init', log: '[NCE] Initialized. Component executor ready.' });
  }

  submitTicket(ticket) {
    const jobId = ticket.job_id || `job-${Date.now()}`;
    this.emit('action', {
      component: 'NCE', action: 'submitTicket',
      log: `[NCE] Received Execution Ticket: ${jobId}`,
      data: ticket
    });

    // 1. Generate manifest
    const manifest = this.generateManifest(ticket);

    // 2. Apply manifest (M8)
    const applyResult = this.applyManifest(manifest);

    // 3. Update status
    this.nds.updateStatus(jobId, 'RUNNING');

    return { jobId, manifest, applyResult };
  }

  generateManifest(ticket) {
    this.emit('action', {
      component: 'NCE', action: 'generateManifest',
      log: '[NCE] Generating Kubernetes Job Manifest...'
    });

    const envVars = ticket.env_vars || {};
    const envList = Array.isArray(envVars)
      ? envVars
      : Object.entries(envVars).map(([k, v]) => ({ name: k, value: String(v) }));

    const jobId = ticket.job_id;
    const serviceId = ticket.service_id;
    const execStrategy = ticket.execution_strategy;
    const targetNode = ticket.target_node;

    if (jobId) envList.push({ name: 'OASEES_JOB_ID', value: jobId });
    if (serviceId) envList.push({ name: 'OASEES_SERVICE_ID', value: serviceId });
    if (execStrategy) envList.push({ name: 'OASEES_EXEC_STRATEGY', value: execStrategy });
    if (targetNode) envList.push({ name: 'OASEES_TARGET_NODE', value: targetNode });

    const manifest = {
      apiVersion: 'batch/v1',
      kind: 'Job',
      metadata: { name: ticket.job_id },
      spec: {
        template: {
          spec: {
            containers: [{
              name: 'workload',
              image: ticket.image || 'cognets/ml-base-image:latest',
              env: envList
            }],
            restartPolicy: 'Never'
          }
        }
      }
    };

    this.emit('action', {
      component: 'NCE', action: 'manifestGenerated',
      log: `[NCE] Manifest generated for job: ${ticket.job_id}`,
      data: manifest
    });

    return manifest;
  }

  applyManifest(manifest) {
    const jobName = manifest.metadata.name;
    this.emit('action', {
      component: 'NCE', action: 'applyManifest',
      log: `[NCE][M8] Applying manifest: ${jobName}`
    });

    // Write actual YAML file
    const yaml = this._toYaml(manifest);
    const tmpFile = path.join(os.tmpdir(), `${jobName}.yaml`);
    fs.writeFileSync(tmpFile, yaml);

    this.emit('action', {
      component: 'NCE', action: 'manifestWritten',
      log: `[NCE] Wrote manifest to ${tmpFile}`,
      data: { file: tmpFile, yaml }
    });

    // Try kubectl, fall back gracefully
    let cmdOutput = '';
    let command = '';
    try {
      command = `kubectl apply --dry-run=client -f ${tmpFile}`;
      cmdOutput = execSync(command, { encoding: 'utf-8', timeout: 5000 });
    } catch (e) {
      // kubectl not available — fall back to cat
      command = `cat ${tmpFile}`;
      try {
        cmdOutput = execSync(command, { encoding: 'utf-8', timeout: 3000 });
      } catch (e2) {
        cmdOutput = yaml; // ultimate fallback
      }
      cmdOutput = `[kubectl not available — showing manifest]\n${cmdOutput}`;
    }

    this.emit('action', {
      component: 'NCE', action: 'manifestApplied',
      log: `[NCE] Command executed: ${command}`,
      data: { command, output: cmdOutput }
    });

    // Cleanup temp file
    try { fs.unlinkSync(tmpFile); } catch (_) {}

    return { command, output: cmdOutput };
  }

  cleanup(jobId) {
    this.emit('action', {
      component: 'NCE', action: 'cleanup',
      log: `[NCE] Cleaning up resources for job: ${jobId}`,
      data: { jobId }
    });
    this.nds.updateStatus(jobId, 'CLEANED');
    this.nds.releaseResources(jobId);
  }

  generateAttestation(manifest, ticket) {
    const manifestStr = JSON.stringify(manifest);
    const manifestHash = crypto.createHash('sha256').update(manifestStr).digest('hex');
    const attestation = {
      job_id: ticket.job_id,
      node_id: (ticket.node_profile || {}).node_id,
      manifest_hash: manifestHash,
      timestamp: Date.now() / 1000
    };
    this.emit('action', {
      component: 'NCE', action: 'attestation',
      log: `[NCE] Attestation generated: ${manifestHash.slice(0, 12)}...`,
      data: attestation
    });
    return attestation;
  }

  _toYaml(obj, indent = 0) {
    // Simple JSON-to-YAML converter for K8s manifests
    const pad = '  '.repeat(indent);
    let out = '';
    for (const [key, val] of Object.entries(obj)) {
      if (val === null || val === undefined) continue;
      if (Array.isArray(val)) {
        out += `${pad}${key}:\n`;
        for (const item of val) {
          if (typeof item === 'object') {
            const lines = this._toYaml(item, indent + 2).split('\n').filter(Boolean);
            out += `${pad}- ${lines[0].trim()}\n`;
            for (let i = 1; i < lines.length; i++) out += `${pad}  ${lines[i].trim()}\n`;
          } else {
            out += `${pad}- ${item}\n`;
          }
        }
      } else if (typeof val === 'object') {
        out += `${pad}${key}:\n${this._toYaml(val, indent + 1)}`;
      } else {
        out += `${pad}${key}: ${val}\n`;
      }
    }
    return out;
  }
}

module.exports = NCE;
