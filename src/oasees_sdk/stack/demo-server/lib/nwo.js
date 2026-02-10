// NWO - Node Workload Orchestrator
// Receives swarm instructions from DSM, checks resources, dispatches to NCE

const EventEmitter = require('events');
const crypto = require('crypto');

class NWO extends EventEmitter {
  constructor(nce, nds) {
    super();
    this.nce = nce;
    this.nds = nds;
    this.emit('action', { component: 'NWO', action: 'init', log: '[NWO] Initialized. Swarm-coordinated orchestrator ready.' });
  }

  handleSwarmNotification(payload) {
    this.emit('action', {
      component: 'NWO', action: 'receiveSwarm',
      log: `[NWO][M6] ⚡ Swarm Instruction received from DSM — type: ${payload.type}`,
      data: payload
    });

    this.emit('action', {
      component: 'NWO', action: 'swarmContext',
      log: '[NWO] Swarm Intelligence: validating instruction against local node capabilities...',
    });

    const data = payload.data || {};
    const serviceId = data.serviceId;
    const requirements = (data.parameters || {}).requirements;
    const swarmNodes = (data.swarm || {}).nodes || [];

    // 0. Multi-node scheduling
    const schedule = this.scheduleNodes(swarmNodes, requirements);
    const selectedNode = (schedule.selected || {}).node_id;
    if (selectedNode && selectedNode !== this.nds.nodeProfile.node_id) {
      this.emit('action', {
        component: 'NWO', action: 'schedule',
        log: `[NWO] Selected remote node ${selectedNode} — forwarding ticket.`,
        data: schedule
      });
      return { status: 'FORWARDED', target_node: selectedNode, schedule };
    }

    // 1. Check resources via NDS
    const resourcesOk = this.nds.checkResources(requirements);
    if (!resourcesOk) {
      this.emit('action', {
        component: 'NWO', action: 'resourceFailed',
        log: `[NWO] Swarm-coordinated execution ABORTED — insufficient resources for: ${serviceId}`,
        data: { status: 'FAILED', reason: 'Insufficient Resources' }
      });
      return { status: 'FAILED', reason: 'Insufficient Resources' };
    }

    // 2. Create execution ticket
    const jobId = `job-${crypto.randomBytes(4).toString('hex')}`;
    const allocation = this.nds.reserveResources(jobId, requirements);
    const executionStrategy = (data.parameters || {}).execution_strategy || 'local-k8s';

    const ticket = {
      job_id: jobId,
      service_id: serviceId,
      image: (data.parameters || {}).image,
      env_vars: (data.parameters || {}).env,
      volume_mounts: (data.parameters || {}).volumes || null,
      allocation,
      execution_strategy: executionStrategy,
      target_node: selectedNode || this.nds.nodeProfile.node_id,
      schedule
    };

    this.emit('action', {
      component: 'NWO', action: 'ticketCreated',
      log: `[NWO] Swarm execution ticket created: ${jobId}`,
      data: ticket
    });

    // 3. Dispatch to NCE (M7)
    this.emit('action', {
      component: 'NWO', action: 'dispatch',
      log: `[NWO][M7] Dispatching execution ticket to NCE — Swarm-coordinated execution underway`,
      data: { job_id: jobId, service_id: serviceId }
    });

    const result = this.nce.submitTicket(ticket);

    return { status: 'ACCEPTED', job_id: jobId, execution_strategy: executionStrategy, target_node: ticket.target_node, schedule, result };
  }

  scoreNode(node, requirements) {
    let score = 0;
    if (requirements && requirements.gpu) score += node.gpu ? 50 : -100;
    score += Math.max(0, 100 - (node.latency_ms || 100)) / 2;
    score += Math.max(0, (1 - (node.load || 0.5)) * 20);
    score += (node.cpu || 0) * 2;
    const recentJobs = node.recent_jobs || 0;
    score -= recentJobs * 5;
    return Math.round(score * 100) / 100;
  }

  scheduleNodes(candidates, requirements) {
    if (!candidates || !candidates.length) {
      return { selected: { node_id: this.nds.nodeProfile.node_id }, scores: [] };
    }
    const scores = candidates.map(n => ({ ...n, score: this.scoreNode(n, requirements) }));
    scores.sort((a, b) => b.score - a.score);
    return { selected: scores[0], scores };
  }
}

module.exports = NWO;
