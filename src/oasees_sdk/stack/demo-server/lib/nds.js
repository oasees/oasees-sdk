// NDS - Node Device Storage
// Manages local state, resources, and simulated sensor data

const EventEmitter = require('events');

class NDS extends EventEmitter {
  constructor() {
    super();
    this.state = {};
    this.telemetry = {};
    this.results = {};
    this.resources = { cpu: '4', memory: '16Gi', gpu: true };
    this.nodeProfile = {
      node_id: 'edge-node-04',
      zone: 'zone-1',
      trust_score: 0.92,
      capabilities: ['gpu', 'camera', 'ngsi-ld']
    };
    this.policies = {
      allowed_sources: new Set(['urn:ngsi-ld:DSM:001']),
      allowed_services: new Set(['urn:ngsi-ld:AIModel:TrafficV2', 'urn:ngsi-ld:AIModel:OccupancyV1'])
    };
    this.dataCatalog = {
      'urn:ngsi-ld:Sensor:Camera01': { type: 'camera', tags: ['traffic', 'road'], scopes: ['traffic'], ttl_s: 5 },
      'urn:ngsi-ld:Sensor:Weather01': { type: 'weather', tags: ['weather', 'humidity'], scopes: ['weather'], ttl_s: 15 }
    };
    this.sensorData = {
      temperature: 25.5,
      humidity: 60.0,
      camera_feed: 'base64_encoded_image_placeholder'
    };
    this.accessTokens = new Map();
    this.emit('action', { component: 'NDS', action: 'init', log: '[NDS] Initialized. Local storage ready.' });
  }

  getNodeProfile() {
    return {
      node_id: this.nodeProfile.node_id,
      zone: this.nodeProfile.zone,
      trust_score: this.nodeProfile.trust_score,
      capabilities: [...this.nodeProfile.capabilities],
      resources: { ...this.resources }
    };
  }

  isSourceTrusted(source) {
    return this.policies.allowed_sources.has(source);
  }

  isServiceAllowed(serviceId) {
    return this.policies.allowed_services.has(serviceId);
  }

  checkResources(requirements) {
    this.emit('action', {
      component: 'NDS', action: 'checkResources',
      log: `[NDS] Checking resources against requirements: ${JSON.stringify(requirements)}`,
      data: { requirements, available: this.resources }
    });

    if (requirements && requirements.gpu && !this.resources.gpu) {
      this.emit('action', {
        component: 'NDS', action: 'checkResourcesFailed',
        log: '[NDS] Resource check FAILED: GPU required but not available.'
      });
      return false;
    }

    this.emit('action', {
      component: 'NDS', action: 'checkResourcesOK',
      log: '[NDS] Resources OK — all requirements satisfied.',
      data: { status: 'OK', available: this.resources }
    });
    return true;
  }

  reserveResources(jobId, requirements) {
    const allocation = {
      cpu: '2',
      memory: '4Gi',
      gpu: Boolean(requirements && requirements.gpu)
    };
    this.state[jobId] = 'RESERVED';
    this.emit('action', {
      component: 'NDS', action: 'reserveResources',
      log: `[NDS] Resources reserved for ${jobId}`,
      data: { allocation }
    });
    return allocation;
  }

  releaseResources(jobId) {
    this.emit('action', {
      component: 'NDS', action: 'releaseResources',
      log: `[NDS] Resources released for ${jobId}`,
      data: { jobId }
    });
    if (this.state[jobId] === 'RESERVED') this.state[jobId] = 'RELEASED';
  }

  selectDataPlan(serviceId) {
    if ((serviceId || '').includes('Traffic')) {
      return { primary: 'urn:ngsi-ld:Sensor:Camera01', sources: ['urn:ngsi-ld:Sensor:Camera01'], scopes: ['traffic'] };
    }
    return { primary: 'urn:ngsi-ld:Sensor:Weather01', sources: ['urn:ngsi-ld:Sensor:Weather01'], scopes: ['weather'] };
  }

  issueAccessToken(jobId, scopes) {
    const token = Math.random().toString(16).slice(2) + Math.random().toString(16).slice(2);
    this.accessTokens.set(token, { jobId, scopes: new Set(scopes || []), issuedAt: Date.now() / 1000 });
    this.emit('action', {
      component: 'NDS', action: 'issueAccessToken',
      log: `[NDS] Access token issued for ${jobId}: ${token.slice(0, 8)}...`,
      data: { jobId, token, scopes }
    });
    return token;
  }

  validateAccessToken(token, scope) {
    const meta = this.accessTokens.get(token);
    if (!meta) return false;
    if (scope && !meta.scopes.has(scope)) return false;
    return true;
  }

  getSensorData(query, accessToken, scope) {
    this.emit('action', {
      component: 'NDS', action: 'getSensorData',
      log: `[NDS] NGSI-LD sensor query received: ${query}`,
      data: { query, interface: 'NGSI-LD', access_token: accessToken, scope }
    });

    if (accessToken && !this.validateAccessToken(accessToken, scope)) {
      this.emit('action', {
        component: 'NDS', action: 'accessDenied',
        log: '[NDS] Access denied: invalid token or scope.',
        data: { query, scope }
      });
      return { error: 'ACCESS_DENIED', reason: 'Invalid token or scope' };
    }

    const result = {
      type: 'SensorReading',
      data: this.sensorData,
      timestamp: Date.now() / 1000
    };

    this.emit('action', {
      component: 'NDS', action: 'sensorDataReturned',
      log: `[NDS] Returning sensor data (${Object.keys(this.sensorData).length} fields).`,
      data: result
    });
    return result;
  }

  updateStatus(executionId, status) {
    this.state[executionId] = status;
    this.emit('action', {
      component: 'NDS', action: 'updateStatus',
      log: `[NDS] Status updated for ${executionId}: ${status}`,
      data: { executionId, status }
    });
  }

  getStatus(executionId) {
    return this.state[executionId] || 'UNKNOWN';
  }

  recordTelemetry(jobId, metrics) {
    if (!this.telemetry[jobId]) this.telemetry[jobId] = [];
    this.telemetry[jobId].push({ ts: Date.now() / 1000, metrics });
    this.emit('action', {
      component: 'NDS', action: 'telemetry',
      log: `[NDS] Telemetry recorded for ${jobId}`,
      data: { jobId, metrics }
    });
  }

  storeResult(jobId, payload) {
    this.results[jobId] = payload;
    this.emit('action', {
      component: 'NDS', action: 'storeResult',
      log: `[NDS] Result stored for ${jobId}`,
      data: { jobId, integrity_proof: payload.integrity_proof }
    });
  }
}

module.exports = NDS;
