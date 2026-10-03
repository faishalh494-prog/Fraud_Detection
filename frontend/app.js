/**
 * SyndicAI — Institutional Transaction Investigation Workstation
 * Dedicated frontend presentation layer consuming the real FastAPI backend.
 */

(function () {
  'use strict';

  // --- CONFIGURATION & STATE ---
  const DEFAULT_KEY = 'test-benchmark-key-01234567890123456789';
  const urlParams = new URLSearchParams(window.location.search);
  let apiKey = urlParams.get('api_key') || localStorage.getItem('syndicai_api_key') || DEFAULT_KEY;
  localStorage.setItem('syndicai_api_key', apiKey);

  const state = {
    currentView: 'live',
    isStreaming: true,
    pollIntervalMs: 2000,
    pollTimer: null,
    selectedEventKey: null,
    selectedRowIndex: null,
    isHistoricalSelection: false,
    referenceMaxStep: 743,
    events: [],
    status: {},
    modelsData: null,
    operatingPoints: null,
    latencySamples: [],
    lastPingMs: 0,
  };

  // --- API HELPER ---
  async function apiRequest(endpoint, options = {}) {
    const url = endpoint.startsWith('http') ? endpoint : endpoint;
    const headers = {
      'Content-Type': 'application/json',
      'X-API-Key': apiKey,
      ...(options.headers || {}),
    };

    const t0 = performance.now();
    try {
      const response = await fetch(url, { ...options, headers });
      const durationMs = Math.round(performance.now() - t0);

      if (response.status === 401) {
        updateKeyStatus(false, 'API KEY INVALID');
        throw new Error('HTTP 401: Unauthorized. Please configure a valid 32+ character SYNDICAI_API_KEY.');
      }
      if (!response.ok) {
        let errorDetail = `HTTP ${response.status}`;
        try {
          const body = await response.json();
          errorDetail = body.detail || JSON.stringify(body);
        } catch (_) {}
        throw new Error(errorDetail);
      }

      const data = await response.json();
      return { data, durationMs };
    } catch (err) {
      throw err;
    }
  }

  // --- UTILITY FORMATTERS ---
  function formatCurrency(amount) {
    if (typeof amount !== 'number') amount = parseFloat(amount) || 0;
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
    }).format(amount);
  }

  function formatTime(isoString) {
    if (!isoString) return '--:--:--';
    try {
      const date = new Date(isoString);
      return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    } catch (_) {
      return isoString;
    }
  }

  function formatScore(score) {
    if (score === null || score === undefined) return '--';
    return Number(score).toFixed(1);
  }

  function riskClass(score, priority) {
    const prio = (priority || '').toLowerCase();
    if (prio.includes('high') || score >= 90) return 'high';
    if (prio.includes('elevated') || prio.includes('review') || score >= 50) return 'review';
    return 'low';
  }

  // --- SYSTEM HEALTH & STATUS PING ---
  async function pingHealth() {
    try {
      const t0 = performance.now();
      const res = await fetch('/health');
      const latency = Math.round(performance.now() - t0);
      state.lastPingMs = latency;

      if (res.ok) {
        const health = await res.json();
        updateServerStatus(true, latency, health.status === 'ready' ? 'ONLINE' : 'ARTIFACTS PENDING');
        updateHealthChecklist(health);
      } else {
        updateServerStatus(false, latency, `STATUS ${res.status}`);
      }
    } catch (e) {
      updateServerStatus(false, 0, 'OFFLINE');
    }
  }

  function updateServerStatus(online, latencyMs, label) {
    const dot = document.getElementById('server-status-dot');
    const text = document.getElementById('server-status-text');
    const lat = document.getElementById('server-latency-text');

    if (online) {
      dot.className = 'indicator-dot active';
      text.textContent = label || 'ONLINE';
      text.style.color = 'var(--teal-accent)';
      lat.textContent = `${latencyMs} ms`;
    } else {
      dot.className = 'indicator-dot';
      text.textContent = label || 'DISCONNECTED';
      text.style.color = 'var(--red-accent)';
      lat.textContent = '-- ms';
    }
  }

  function updateKeyStatus(valid, text) {
    const dot = document.getElementById('key-configured-dot');
    const label = document.getElementById('key-status-label');
    if (valid) {
      dot.style.backgroundColor = 'var(--teal-accent)';
      label.textContent = text || 'KEY CONFIGURED';
    } else {
      dot.style.backgroundColor = 'var(--red-accent)';
      label.textContent = text || 'KEY REQUIRED';
    }
  }

  function updateHealthChecklist(health) {
    const setStatus = (id, ready) => {
      const el = document.getElementById(id);
      if (!el) return;
      el.className = ready ? 'check-icon ready' : 'check-icon missing';
      el.textContent = ready ? '✓' : '✗';
    };
    setStatus('chk-api', true);
    setStatus('chk-parquet', health.missing_artifact_count === 0);
    setStatus('chk-index', health.missing_artifact_count === 0);
    setStatus('chk-models', health.missing_artifact_count === 0);
    setStatus('chk-state', health.api_key_configured);
  }

  // --- NAVIGATION & ROUTING ---
  function setupNavigation() {
    const navItems = document.querySelectorAll('.nav-item');
    navItems.forEach(item => {
      item.addEventListener('click', () => {
        const view = item.getAttribute('data-view');
        switchView(view);
      });
    });

    document.getElementById('btn-open-key-modal').addEventListener('click', () => {
      switchView('system');
    });
  }

  function switchView(viewName) {
    state.currentView = viewName;

    // Update nav active
    document.querySelectorAll('.nav-item').forEach(btn => {
      if (btn.getAttribute('data-view') === viewName) {
        btn.classList.add('active');
      } else {
        btn.classList.remove('active');
      }
    });

    // Update panels
    document.querySelectorAll('.view-section').forEach(sec => {
      if (sec.id === `view-${viewName}`) {
        sec.classList.add('active');
      } else {
        sec.classList.remove('active');
      }
    });

    // Load view data
    if (viewName === 'alerts') {
      loadAlerts();
    } else if (viewName === 'investigations') {
      loadCases();
    } else if (viewName === 'evidence') {
      loadEvidence();
    } else if (viewName === 'score') {
      initScoreForm();
    }
  }

  // --- LIVE MONITOR & RIVER STREAM ---
  async function pollLiveStream() {
    if (!state.isStreaming && state.events.length > 0) return;

    try {
      const [statusRes, eventsRes] = await Promise.all([
        apiRequest('/live/status'),
        apiRequest('/live/events?limit=50'),
      ]);

      state.status = statusRes.data;
      const newEvents = eventsRes.data.events || [];

      // Update statistics
      document.getElementById('stat-total-events').textContent = (state.status.event_count || 0).toLocaleString();
      document.getElementById('stat-flagged-events').textContent = (state.status.flagged_event_count || 0).toLocaleString();
      document.getElementById('stat-latest-step').textContent = state.status.latest_step || '--';
      document.getElementById('nav-live-count').textContent = (state.status.event_count || 0).toLocaleString();

      if (state.status.reference_max_step) {
        state.referenceMaxStep = state.status.reference_max_step;
        document.getElementById('ref-max-step-badge').textContent = `1 — ${state.referenceMaxStep}`;
        const inputStep = document.getElementById('input-step');
        if (inputStep && (!inputStep.value || parseInt(inputStep.value) <= state.referenceMaxStep)) {
          inputStep.value = (state.status.latest_step || state.referenceMaxStep) + 1;
        }
        const formRef = document.getElementById('form-ref-max');
        if (formRef) formRef.textContent = state.referenceMaxStep;
      }

      state.events = newEvents;
      renderRiverTable(newEvents);

      // Auto-select latest event if none selected
      if (!state.selectedEventKey && newEvents.length > 0 && !state.isHistoricalSelection) {
        selectLiveEvent(newEvents[0].event_key);
      }
    } catch (err) {
      console.warn('Poll error:', err);
      if (err.message.includes('401')) {
        updateKeyStatus(false, 'KEY INVALID');
      }
    }
  }

  function renderRiverTable(events) {
    const tbody = document.getElementById('river-table-body');
    const typeFilter = document.getElementById('river-filter-type').value;
    const prioFilter = document.getElementById('river-filter-priority').value;

    const filtered = events.filter(ev => {
      const tx = ev.transaction || {};
      const risk = ev.risk || {};
      if (typeFilter !== 'ALL' && tx.type !== typeFilter) return false;
      if (prioFilter === 'FLAGGED' && !risk.flagged_for_review) return false;
      if (prioFilter === 'HIGH' && risk.review_priority !== 'High review priority') return false;
      if (prioFilter === 'NORMAL' && risk.review_priority === 'High review priority') return false;
      return true;
    });

    if (filtered.length === 0) {
      tbody.innerHTML = `
        <tr class="empty-row">
          <td colspan="8">
            <div class="empty-state-box">
              <p>No transactions match the selected filter (${events.length} total events loaded).</p>
            </div>
          </td>
        </tr>`;
      return;
    }

    // Compute average latency
    const validTimings = events.map(e => e.timings?.processing_ms).filter(Boolean);
    if (validTimings.length > 0) {
      const avg = (validTimings.reduce((a, b) => a + b, 0) / validTimings.length).toFixed(1);
      document.getElementById('stat-avg-latency').textContent = `${avg} ms`;
    }

    tbody.innerHTML = filtered.map(ev => {
      const tx = ev.transaction || {};
      const risk = ev.risk || {};
      const timings = ev.timings || {};
      const isSelected = ev.event_key === state.selectedEventKey;
      const flagged = risk.flagged_for_review ? 'flagged' : 'normal';
      const rClass = riskClass(risk.score, risk.review_priority);
      const invStatus = (ev.investigation?.status || 'Open').toLowerCase();

      return `
        <tr class="river-row ${flagged} ${isSelected ? 'selected' : ''}" data-key="${ev.event_key}">
          <td>
            <div class="mono" style="font-size:11px;font-weight:600;color:var(--text-main);">${ev.event_key}</div>
            <div style="font-size:10px;color:var(--text-muted);">${formatTime(ev.processed_at)}</div>
          </td>
          <td><span class="mono">${tx.step || '--'}</span></td>
          <td><span class="type-tag ${(tx.type || '').toLowerCase()}">${tx.type || '--'}</span></td>
          <td class="text-right mono" style="font-weight:600;color:#FFFFFF;">${formatCurrency(tx.amount)}</td>
          <td>
            <div class="mono" style="font-size:11px;">${tx.sender || '--'}</div>
            <div class="mono" style="font-size:10px;color:var(--text-muted);">↳ ${tx.receiver || '--'}</div>
          </td>
          <td>
            <span class="score-badge ${rClass}">${formatScore(risk.score)}</span>
          </td>
          <td>
            <span class="status-badge ${invStatus}">${ev.investigation?.status || 'Open'}</span>
          </td>
          <td class="text-right mono" style="font-size:11px;color:var(--text-muted);">
            ${timings.processing_ms ? `${Number(timings.processing_ms).toFixed(1)} ms` : '--'}
          </td>
        </tr>`;
    }).join('');

    // Attach click events
    tbody.querySelectorAll('.river-row').forEach(row => {
      row.addEventListener('click', () => {
        const key = row.getAttribute('data-key');
        selectLiveEvent(key);
      });
    });
  }

  // --- SELECT & INSPECT LIVE EVENT ---
  async function selectLiveEvent(eventKey) {
    if (!eventKey) return;
    state.selectedEventKey = eventKey;
    state.isHistoricalSelection = false;

    // Highlight row in table
    document.querySelectorAll('.river-row').forEach(r => {
      if (r.getAttribute('data-key') === eventKey) {
        r.classList.add('selected');
      } else {
        r.classList.remove('selected');
      }
    });

    try {
      const res = await apiRequest(`/live/events/${encodeURIComponent(eventKey)}`);
      renderInvestigationWorkspace(res.data, false);
    } catch (err) {
      console.error('Failed to load event details:', err);
    }
  }

  // --- RENDER INVESTIGATION WORKSPACE ---
  function renderInvestigationWorkspace(eventData, isHistorical = false) {
    const tx = eventData.transaction || {};
    const risk = eventData.risk || {};
    const expl = eventData.explanation || {};
    const beh = eventData.behavioural_evidence || eventData.behaviour || {};
    const evStrength = eventData.evidence_strength || {};
    const timings = eventData.timings || {};
    const inv = eventData.investigation || {};
    const net = eventData.network_context || {};
    const onlineNet = eventData.online_network_context || {};
    const related = eventData.related_activity || [];

    // Header
    const eventId = isHistorical ? `ROW-${eventData.row_index}` : (eventData.event_key || 'UNKNOWN');
    document.getElementById('ws-event-id').textContent = eventId;
    document.getElementById('ws-timestamp').textContent = isHistorical
      ? `Historical split: ${eventData.evaluation_period || 'test'} · ${eventData.score_context || ''}`
      : `Processed at ${eventData.processed_at ? new Date(eventData.processed_at).toLocaleString() : '--'} · Step ${tx.step}`;

    const statusBadge = document.getElementById('ws-status-badge');
    const currentStatus = inv.status || 'Open';
    statusBadge.textContent = currentStatus.toUpperCase();
    statusBadge.className = `workspace-status-badge ${currentStatus.toLowerCase()}`;

    // Risk card
    const scoreVal = document.getElementById('ws-score-value');
    scoreVal.textContent = formatScore(risk.score);
    const rClass = riskClass(risk.score, risk.review_priority);
    scoreVal.style.color = rClass === 'high' ? 'var(--red-accent)' : (rClass === 'review' ? 'var(--amber-accent)' : 'var(--teal-accent)');

    document.getElementById('ws-review-priority').textContent = risk.review_priority || '--';
    document.getElementById('ws-review-priority').style.color = rClass === 'high' ? 'var(--red-accent)' : (rClass === 'review' ? 'var(--amber-accent)' : 'var(--teal-accent)');
    document.getElementById('ws-threshold').textContent = `${formatScore(risk.review_threshold)} / 100`;
    document.getElementById('ws-flag-status').textContent = risk.flagged_for_review ? 'FLAGGED FOR HUMAN REVIEW' : 'STANDARD ROUTING';
    document.getElementById('ws-flag-status').style.color = risk.flagged_for_review ? 'var(--red-accent)' : 'var(--text-secondary)';

    // Particulars
    document.getElementById('ws-tx-amount').textContent = formatCurrency(tx.amount);
    document.getElementById('ws-tx-type').textContent = tx.type || tx.transaction_type || '--';
    document.getElementById('ws-tx-step').textContent = tx.step || '--';
    document.getElementById('ws-tx-sender').textContent = tx.sender || '--';
    document.getElementById('ws-tx-receiver').textContent = tx.receiver || '--';

    // TreeSHAP evidence
    const shapContainer = document.getElementById('ws-shap-factors');
    const reasons = expl.reasons || [];
    if (reasons.length === 0) {
      shapContainer.innerHTML = '<div class="no-data-hint">No SHAP factor contributions available for this model.</div>';
    } else {
      // Find maximum absolute contribution for scaling
      const maxAbs = Math.max(...reasons.map(r => Math.abs(r.contribution || 0)), 0.001);
      shapContainer.innerHTML = reasons.slice(0, 7).map(r => {
        const contrib = Number(r.contribution || 0);
        const isPos = contrib > 0;
        const pct = Math.min(Math.round((Math.abs(contrib) / maxAbs) * 100), 100);
        const signed = contrib > 0 ? `+${contrib.toFixed(3)}` : contrib.toFixed(3);

        return `
          <div class="shap-row">
            <div class="shap-meta">
              <span class="shap-name">${r.label || r.feature}</span>
              <div>
                <span class="shap-val">${r.value !== undefined ? r.value : ''}</span>
                <span class="shap-score ${isPos ? 'positive' : 'negative'}" style="margin-left:8px;">${signed}</span>
              </div>
            </div>
            <div class="shap-bar-bg">
              <div class="shap-bar-fill ${isPos ? 'positive' : 'negative'}" style="width: ${pct}%;"></div>
            </div>
          </div>`;
      }).join('');
    }

    // Behavioural history
    const histBadge = document.getElementById('ws-history-badge');
    histBadge.textContent = evStrength.status || (beh.receiver_is_new === 1 ? 'New Account' : 'History Available');
    histBadge.style.color = (evStrength.status === 'New' || beh.receiver_is_new === 1) ? 'var(--amber-accent)' : 'var(--teal-accent)';

    document.getElementById('ws-b-recv-count').textContent = (beh.receiver_txn_count_before !== undefined ? beh.receiver_txn_count_before : '--');
    document.getElementById('ws-b-recv-total').textContent = beh.receiver_total_amount_before ? formatCurrency(beh.receiver_total_amount_before) : '$0.00';
    document.getElementById('ws-b-recv-avg').textContent = beh.receiver_avg_amount_before ? formatCurrency(beh.receiver_avg_amount_before) : '$0.00';
    document.getElementById('ws-b-recv-vel').textContent = (beh.receiver_txn_count_last24_before !== undefined ? beh.receiver_txn_count_last24_before : '--');
    document.getElementById('ws-b-recv-recency').textContent = beh.receiver_steps_since_last !== undefined && beh.receiver_steps_since_last >= 0 ? `${beh.receiver_steps_since_last} steps ago` : 'None';
    document.getElementById('ws-b-sender-count').textContent = (beh.sender_txn_count_before !== undefined ? beh.sender_txn_count_before : '--');

    // Related Activity & Network
    const netContainer = document.getElementById('ws-network-summary');
    let netHtml = '';

    const priorRelSeen = onlineNet.prior_relationship_seen || net.prior_relationship_seen;
    netHtml += `
      <div style="margin-bottom:8px;font-size:11px;">
        <span style="color:var(--text-muted);">PRIOR COUNTERPARTY TRANSFER:</span>
        <strong style="color:${priorRelSeen ? 'var(--amber-accent)' : 'var(--text-secondary)'}; margin-left:6px;">
          ${priorRelSeen ? 'YES — Prior direct edge observed' : 'NO — First observed transfer between these accounts'}
        </strong>
      </div>`;

    if (related.length > 0) {
      netHtml += `
        <div style="margin-top:10px;">
          <div style="font-size:10px;font-weight:700;color:var(--text-muted);margin-bottom:6px;">RECENT RELATED ONLINE TRANSACTIONS (${related.length}):</div>
          <table style="width:100%;font-size:10px;border-collapse:collapse;">
            ${related.slice(0, 4).map(r => `
              <tr style="border-bottom:1px solid rgba(255,255,255,0.05);">
                <td class="mono" style="padding:4px 0;">Step ${r.step}</td>
                <td style="padding:4px 0;">${r.type}</td>
                <td class="mono text-right" style="padding:4px 0;">${formatCurrency(r.amount)}</td>
                <td style="padding:4px 0;text-align:right;">
                  <span class="status-badge ${(r.investigation_status || 'open').toLowerCase()}" style="font-size:8px;">${r.investigation_status || 'Open'}</span>
                </td>
              </tr>
            `).join('')}
          </table>
        </div>`;
    }

    const counterparties = onlineNet.sender_prior_receivers || [];
    if (counterparties.length > 0) {
      netHtml += `
        <div style="margin-top:10px;">
          <div style="font-size:10px;font-weight:700;color:var(--text-muted);margin-bottom:4px;">OBSERVED PRIOR SENDER COUNTERPARTIES:</div>
          <div class="mono" style="font-size:10px;color:var(--text-secondary);">
            ${counterparties.slice(0, 3).map(c => `${c.account} (${c.transactions} txns, ${formatCurrency(c.amount)})`).join(' · ')}
          </div>
        </div>`;
    }

    if (!priorRelSeen && related.length === 0 && counterparties.length === 0) {
      netHtml += '<div class="no-data-hint">No prior online counterparties recorded in causal history.</div>';
    }

    netContainer.innerHTML = netHtml;

    // Action Card & Notes
    document.getElementById('ws-action-current-status').textContent = `STATUS: ${currentStatus.toUpperCase()}`;
    const noteArea = document.getElementById('ws-investigation-note');
    noteArea.value = inv.note || '';
    document.getElementById('ws-note-saved-time').textContent = inv.updated_at ? `Updated: ${new Date(inv.updated_at).toLocaleTimeString()}` : 'Not saved';

    // Update bottom latency bar
    document.getElementById('lat-feat').textContent = `${Number(timings.feature_ms || 0).toFixed(2)} ms`;
    document.getElementById('lat-inf').textContent = `${Number(timings.inference_ms || 0).toFixed(2)} ms`;
    document.getElementById('lat-exp').textContent = `${Number(timings.explanation_ms || 0).toFixed(2)} ms`;
    document.getElementById('lat-state').textContent = `${Number(timings.state_update_ms || 0).toFixed(2)} ms`;
    document.getElementById('lat-total').textContent = `${Number(timings.processing_ms || 0).toFixed(2)} ms`;
  }

  // --- ACTIONS (INVESTIGATING, ESCALATING, CLOSING, NOTES) ---
  function setupActions() {
    const actionBtns = document.querySelectorAll('.action-btn');
    actionBtns.forEach(btn => {
      btn.addEventListener('click', async () => {
        const newStatus = btn.getAttribute('data-status');
        const note = document.getElementById('ws-investigation-note').value;
        await submitInvestigationUpdate(newStatus, note);
      });
    });

    document.getElementById('btn-save-note').addEventListener('click', async () => {
      const currentStatusBadge = document.getElementById('ws-status-badge').textContent || 'OPEN';
      const status = currentStatusBadge.charAt(0).toUpperCase() + currentStatusBadge.slice(1).toLowerCase();
      const note = document.getElementById('ws-investigation-note').value;
      await submitInvestigationUpdate(status, note);
    });
  }

  async function submitInvestigationUpdate(status, note) {
    if (!state.selectedEventKey && state.selectedRowIndex === null) {
      alert('Please select a transaction to investigate first.');
      return;
    }

    try {
      if (state.isHistoricalSelection && state.selectedRowIndex !== null) {
        // Historical investigation update
        await apiRequest(`/investigations/${state.selectedRowIndex}`, {
          method: 'PUT',
          body: JSON.stringify({ status, note }),
        });
      } else {
        // Live event investigation update
        await apiRequest(`/live/events/${encodeURIComponent(state.selectedEventKey)}/investigation`, {
          method: 'PUT',
          body: JSON.stringify({ status, note }),
        });
      }

      // Update UI state
      document.getElementById('ws-status-badge').textContent = status.toUpperCase();
      document.getElementById('ws-status-badge').className = `workspace-status-badge ${status.toLowerCase()}`;
      document.getElementById('ws-action-current-status').textContent = `STATUS: ${status.toUpperCase()}`;
      document.getElementById('ws-note-saved-time').textContent = `Updated: ${new Date().toLocaleTimeString()}`;

      // Refresh river feed to reflect updated status
      pollLiveStream();
    } catch (err) {
      alert(`Failed to update investigation: ${err.message}`);
    }
  }

  // --- ALERTS VIEW (HISTORICAL TEST ALERTS) ---
  async function loadAlerts() {
    const model = document.getElementById('alert-model-select').value || 'B';
    const tbody = document.getElementById('alerts-table-body');
    tbody.innerHTML = '<tr><td colspan="8" style="text-align:center;padding:30px;"><div class="empty-spinner"></div>Loading test alert queue...</td></tr>';

    try {
      const res = await apiRequest(`/alerts?model=${model}&limit=100`);
      const alerts = res.data || [];
      document.getElementById('nav-alert-count').textContent = alerts.length.toLocaleString();

      if (alerts.length === 0) {
        tbody.innerHTML = '<tr><td colspan="8" style="text-align:center;padding:30px;">No alerts in queue for selected model.</td></tr>';
        return;
      }

      tbody.innerHTML = alerts.map(a => `
        <tr class="alert-row" data-row="${a.row_index}">
          <td class="mono"><strong>#${a.row_index}</strong></td>
          <td class="mono">${a.step}</td>
          <td><span class="type-tag ${(a.transaction_type || '').toLowerCase()}">${a.transaction_type}</span></td>
          <td class="text-right mono" style="font-weight:600;">${formatCurrency(a.amount)}</td>
          <td><span class="score-badge high">${formatScore(a.risk_score)}</span></td>
          <td>${a.review_priority}</td>
          <td><span class="status-badge ${(a.status || 'open').toLowerCase()}">${a.status || 'Open'}</span></td>
          <td><button class="secondary-btn" style="padding:2px 8px;font-size:10px;" onclick="window.selectHistoricalTransaction(${a.row_index}, '${model}')">Inspect</button></td>
        </tr>
      `).join('');

      tbody.querySelectorAll('.alert-row').forEach(row => {
        row.addEventListener('click', (e) => {
          if (e.target.tagName !== 'BUTTON') {
            const rowIndex = parseInt(row.getAttribute('data-row'), 10);
            selectHistoricalTransaction(rowIndex, model);
          }
        });
      });
    } catch (err) {
      tbody.innerHTML = `<tr><td colspan="8" style="color:var(--red-accent);text-align:center;padding:20px;">Failed to load alerts: ${err.message}</td></tr>`;
    }
  }

  window.selectHistoricalTransaction = async function (rowIndex, model = 'B') {
    state.selectedRowIndex = rowIndex;
    state.selectedEventKey = null;
    state.isHistoricalSelection = true;

    try {
      const res = await apiRequest(`/transactions/${rowIndex}?model=${model}`);
      renderInvestigationWorkspace(res.data, true);
    } catch (err) {
      alert(`Failed to load historical transaction: ${err.message}`);
    }
  };

  // --- CASES / INVESTIGATIONS LIST ---
  async function loadCases() {
    const tbody = document.getElementById('cases-table-body');
    // Filter active live events that have status other than Open or have a note
    const activeCases = state.events.filter(e => {
      const inv = e.investigation || {};
      return (inv.status && inv.status !== 'Open') || (inv.note && inv.note.trim().length > 0);
    });

    document.getElementById('nav-cases-count').textContent = activeCases.length;

    if (activeCases.length === 0) {
      tbody.innerHTML = `
        <tr>
          <td colspan="8" style="text-align:center;padding:40px;color:var(--text-muted);">
            No active cases yet. Select an event in the Live Monitor or Alert Queue and click <em>Mark Investigating</em> or add notes.
          </td>
        </tr>`;
      return;
    }

    tbody.innerHTML = activeCases.map(c => `
      <tr class="case-row" data-key="${c.event_key}">
        <td class="mono"><strong>${c.event_key}</strong></td>
        <td class="mono">${c.transaction?.step || '--'}</td>
        <td class="mono">${formatCurrency(c.transaction?.amount)}</td>
        <td class="mono">${c.transaction?.sender} → ${c.transaction?.receiver}</td>
        <td><span class="score-badge ${riskClass(c.risk?.score, c.risk?.review_priority)}">${formatScore(c.risk?.score)}</span></td>
        <td><span class="status-badge ${(c.investigation?.status || 'open').toLowerCase()}">${c.investigation?.status}</span></td>
        <td style="max-width:200px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">${c.investigation?.note || '<em>No notes</em>'}</td>
        <td style="font-size:10px;color:var(--text-muted);">${c.investigation?.updated_at ? new Date(c.investigation.updated_at).toLocaleString() : '--'}</td>
      </tr>
    `).join('');

    tbody.querySelectorAll('.case-row').forEach(row => {
      row.addEventListener('click', () => {
        const key = row.getAttribute('data-key');
        selectLiveEvent(key);
      });
    });
  }

  // --- MODEL EVIDENCE VIEW ---
  async function loadEvidence() {
    try {
      const [modelsRes, opRes] = await Promise.all([
        apiRequest('/models'),
        apiRequest('/operating_points'),
      ]);

      const models = modelsRes.data.models || {};
      const tbody = document.getElementById('model-metrics-tbody');

      tbody.innerHTML = ['B', 'A', 'C'].map(name => {
        const m = models[name];
        if (!m) return '';
        const isDefault = name === 'B';
        const v = m.validation || {};
        const t = m.test || {};

        return `
          <tr style="${isDefault ? 'background-color: var(--blue-dim); font-weight:600;' : ''}">
            <td><strong>Model ${name}</strong> ${isDefault ? '<span class="type-tag transfer" style="font-size:9px;">PRODUCTION DEFAULT</span>' : ''}</td>
            <td>${m.features?.length || 0} features</td>
            <td>${formatScore(v.threshold * 100)} / 100</td>
            <td>${t.f1 ? t.f1.toFixed(4) : '--'}</td>
            <td>${t.precision ? (t.precision * 100).toFixed(1) + '%' : '--'}</td>
            <td>${t.recall ? (t.recall * 100).toFixed(1) + '%' : '--'}</td>
            <td>${t.pr_auc ? t.pr_auc.toFixed(4) : '--'}</td>
            <td>${t.alerts ? t.alerts.toLocaleString() : '--'}</td>
          </tr>`;
      }).join('');

      // Operating points list
      const opList = document.getElementById('operating-points-list');
      const points = opRes.data.operating_points || [];
      opList.innerHTML = points.map(p => `
        <div style="background-color:var(--bg-canvas);border:1px solid var(--border-subtle);border-radius:4px;padding:8px 12px;margin-bottom:6px;display:flex;justify-content:space-between;align-items:center;">
          <div>
            <strong style="color:var(--text-main);font-size:12px;">${p.label}</strong>
            <span class="mono" style="font-size:11px;color:var(--text-muted);margin-left:8px;">Threshold: ${formatScore(p.threshold * 100)} / 100</span>
          </div>
          <div class="mono" style="font-size:11px;">
            <span style="color:var(--teal-accent);">Test F1: ${p.test?.f1?.toFixed(4)}</span> ·
            <span style="color:var(--amber-accent);">Alerts: ${p.test?.alerts?.toLocaleString()}</span>
          </div>
        </div>
      `).join('');
    } catch (err) {
      console.warn('Failed to load evidence:', err);
    }
  }

  // --- MANUAL SCORING FORM ---
  function initScoreForm() {
    const inputStep = document.getElementById('input-step');
    if (inputStep && (!inputStep.value || parseInt(inputStep.value) <= state.referenceMaxStep)) {
      inputStep.value = (state.status.latest_step || state.referenceMaxStep) + 1;
    }

    document.getElementById('btn-fill-mule-example').onclick = () => {
      document.getElementById('input-type').value = 'CASH_OUT';
      document.getElementById('input-amount').value = '985000.00';
      document.getElementById('input-orig').value = 'C228192041';
      document.getElementById('input-dest').value = 'C449102831';
    };

    document.getElementById('btn-fill-routine-example').onclick = () => {
      document.getElementById('input-type').value = 'PAYMENT';
      document.getElementById('input-amount').value = '18.50';
      document.getElementById('input-orig').value = 'C104928190';
      document.getElementById('input-dest').value = 'M882194012';
    };

    document.getElementById('manual-scoring-form').onsubmit = async (e) => {
      e.preventDefault();
      const feedback = document.getElementById('score-feedback');
      const spinner = document.getElementById('scoring-spinner');
      feedback.style.display = 'none';
      spinner.style.display = 'inline-block';

      const payload = {
        step: parseInt(document.getElementById('input-step').value, 10),
        type: document.getElementById('input-type').value,
        amount: parseFloat(document.getElementById('input-amount').value),
        nameOrig: document.getElementById('input-orig').value.trim(),
        nameDest: document.getElementById('input-dest').value.trim(),
        model: 'B',
        operating_point: document.getElementById('input-policy').value,
        event_id: `manual-${Date.now().toString(36)}`,
      };

      try {
        const res = await apiRequest('/score_transaction', {
          method: 'POST',
          body: JSON.stringify(payload),
        });

        spinner.style.display = 'none';
        feedback.className = 'score-submission-feedback success';
        feedback.style.display = 'block';
        feedback.innerHTML = `
          <strong>Transaction Successfully Scored</strong><br>
          Risk Score: <strong>${res.data.risk.score}/100</strong> (${res.data.risk.review_priority})<br>
          Latency: <strong>${res.data.timings.processing_ms} ms</strong> (Feature lookup: ${res.data.timings.feature_ms} ms · Inference: ${res.data.timings.inference_ms} ms)<br>
          ${res.data.explanation.summary}
        `;

        // Switch back to live view and inspect result
        setTimeout(() => {
          switchView('live');
          pollLiveStream();
          selectLiveEvent(payload.event_id);
        }, 1200);
      } catch (err) {
        spinner.style.display = 'none';
        feedback.className = 'score-submission-feedback error';
        feedback.style.display = 'block';
        feedback.textContent = `Scoring Failed: ${err.message}`;
      }
    };
  }

  // --- KEY SETTINGS ---
  function setupSystemKeyConfig() {
    const keyInput = document.getElementById('input-api-key');
    keyInput.value = apiKey;

    document.getElementById('btn-save-key').addEventListener('click', () => {
      const val = keyInput.value.trim();
      if (val.length < 32) {
        alert('SYNDICAI_API_KEY must be at least 32 characters long.');
        return;
      }
      apiKey = val;
      localStorage.setItem('syndicai_api_key', apiKey);
      document.getElementById('key-feedback-msg').textContent = 'Key saved to browser session.';
      updateKeyStatus(true, 'KEY CONFIGURED');
      pingHealth();
      pollLiveStream();
    });

    document.getElementById('btn-toggle-key-visibility').addEventListener('click', () => {
      const btn = document.getElementById('btn-toggle-key-visibility');
      if (keyInput.type === 'password') {
        keyInput.type = 'text';
        btn.textContent = 'Hide';
      } else {
        keyInput.type = 'password';
        btn.textContent = 'Show';
      }
    });
  }

  // --- CONTROLS: STREAM TOGGLE & REFRESH ---
  function setupStreamControls() {
    const toggleBtn = document.getElementById('btn-stream-toggle');
    const toggleIcon = document.getElementById('stream-toggle-icon');
    const toggleLabel = document.getElementById('stream-toggle-label');

    toggleBtn.addEventListener('click', () => {
      state.isStreaming = !state.isStreaming;
      if (state.isStreaming) {
        toggleBtn.classList.add('active');
        toggleIcon.textContent = '⏸';
        toggleLabel.textContent = 'LIVE FEED';
      } else {
        toggleBtn.classList.remove('active');
        toggleIcon.textContent = '▶';
        toggleLabel.textContent = 'PAUSED';
      }
    });

    document.getElementById('btn-manual-refresh').addEventListener('click', () => {
      pollLiveStream();
      pingHealth();
    });

    document.getElementById('river-filter-type').addEventListener('change', () => {
      renderRiverTable(state.events);
    });

    document.getElementById('river-filter-priority').addEventListener('change', () => {
      renderRiverTable(state.events);
    });

    document.getElementById('alert-model-select').addEventListener('change', () => {
      loadAlerts();
    });
  }

  // --- INITIALIZATION ---
  async function init() {
    setupNavigation();
    setupActions();
    setupSystemKeyConfig();
    setupStreamControls();

    // Check server health
    await pingHealth();
    // Load initial events
    await pollLiveStream();

    // Set recurring timer
    state.pollTimer = setInterval(() => {
      pollLiveStream();
      pingHealth();
    }, state.pollIntervalMs);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

})();
