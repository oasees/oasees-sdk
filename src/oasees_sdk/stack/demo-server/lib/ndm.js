// NDM - Node Data Manager
// Ensures data integrity via SHA-256 signing and publishes results

const EventEmitter = require('events');
const crypto = require('crypto');

class NDM extends EventEmitter {
  constructor(nds) {
    super();
    this.nds = nds || null;
    this.emit('action', { component: 'NDM', action: 'init', log: '[NDM] Initialized. Data integrity module ready.' });
  }

  signData(data) {
    const dataStr = JSON.stringify(data, Object.keys(data).sort());
    const signature = crypto.createHash('sha256').update(dataStr).digest('hex');
    this.emit('action', {
      component: 'NDM', action: 'signData',
      log: `[NDM] Data signed — SHA-256: ${signature.slice(0, 16)}...`,
      data: { inputSize: dataStr.length, signature }
    });
    return signature;
  }

  processOutput(data, jobId) {
    this.emit('action', {
      component: 'NDM', action: 'processOutput',
      log: `[NDM] Processing output data for integrity wrapping...`,
      data
    });

    // 1. Sign
    const signature = this.signData(data);

    // 2. Add metadata
    const payload = {
      data,
      signature,
      timestamp: Date.now() / 1000,
      integrity_proof: `proof-${signature.slice(0, 8)}`
    };

    // 3. Publish
    this.emit('action', {
      component: 'NDM', action: 'publish',
      log: `[NDM] Publishing signed result (M11) — integrity_proof: ${payload.integrity_proof}`,
      data: payload
    });

    if (this.nds && jobId) {
      this.nds.storeResult(jobId, payload);
    }

    return payload;
  }
}

module.exports = NDM;
