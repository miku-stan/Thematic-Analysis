/* Thematic Analysis Studio: local, browser-side orchestration and rendering. */
'use strict';

const $ = (id) => document.getElementById(id);
const state = { parsed: null, analysis: null, themes: [], lastRequest: null, busy: false };
const fields = [
  ['full_name', 'Identifier / Record Name'],
  ['headline', 'Headline / Short Summary'],
  ['current_position', 'Position / Primary Role'],
  ['current_company', 'Company / Workplace / Source'],
  ['degree_program', 'Degree Program / Discipline'],
  ['skills', 'Skills / Competencies / Keywords'],
  ['about', 'Narrative / Response / Open-Ended Text'],
];
const esc = (value) => String(value ?? '').replace(/[&<>"']/g, c => ({
  '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
}[c]));
const number = (value, digits = 1) => Number(value || 0).toFixed(digits);

async function api(path, options = {}) {
  const response = await fetch(path, options);
  let body;
  try { body = await response.json(); } catch { body = {}; }
  if (!response.ok) throw new Error(body.detail || body.error || `Request failed (${response.status})`);
  return body;
}
function jsonOptions(body) { return { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) }; }
function notify(message, error = false) {
  const banner = $('toastBanner');
  banner.textContent = message;
  banner.style.borderColor = error ? 'var(--rose)' : 'var(--accent)';
  banner.classList.add('show');
  clearTimeout(notify.timer);
  notify.timer = setTimeout(() => banner.classList.remove('show'), 4500);
}
function setBusy(value) {
  state.busy = value;
  document.querySelectorAll('button').forEach(button => {
    if (button.id === 'themeToggleBtn') return;
    button.disabled = value;
  });
}

window.switchTab = function (name) {
  document.querySelectorAll('.nav-tab').forEach(el => el.classList.toggle('active', el.dataset.tab === name));
  document.querySelectorAll('.tab-panel').forEach(el => el.classList.toggle('active', el.id === `tab-${name}`));
};
window.toggleThemeMode = function () {
  const light = document.documentElement.dataset.theme !== 'light';
  document.documentElement.dataset.theme = light ? 'light' : 'dark';
  $('themeToggleBtn').textContent = light ? 'Dark Mode' : 'Light Mode';
  localStorage.setItem('thematic-theme', light ? 'light' : 'dark');
};
window.triggerFileInput = function () {
  const input = $('fileInput');
  if (input) input.click();
};

function showParsed(data) {
  state.parsed = data;
  $('uploadPreviewBox').style.display = '';
  $('uploadedFileName').textContent = data.filename;
  $('uploadedFileMeta').textContent = `${data.total_rows} records · ${data.file_type}`;
  const mapping = data.suggested_mapping || {};
  $('columnMapperGrid').innerHTML = fields.map(([field, label]) => `<label class="form-label">${esc(label)}<select class="select mapping-select" data-field="${field}"><option value="">Do not map</option>${data.columns.map(col => `<option value="${esc(col)}" ${mapping[field] === col ? 'selected' : ''}>${esc(col)}</option>`).join('')}</select></label>`).join('');
  const table = $('uploadPreviewTable');
  table.querySelector('thead').innerHTML = `<tr>${data.columns.map(col => `<th>${esc(col)}</th>`).join('')}</tr>`;
  table.querySelector('tbody').innerHTML = data.preview_rows.slice(0, 5).map(row => `<tr>${data.columns.map(col => `<td>${esc(row[col])}</td>`).join('')}</tr>`).join('');
  notify(`Loaded ${data.total_rows} records from ${data.filename}`);
}
window.clearUploadedFile = function () {
  state.parsed = null;
  $('fileInput').value = '';
  $('uploadPreviewBox').style.display = 'none';
};
async function upload(files) {
  if (!files) return;
  const fileList = files instanceof FileList || Array.isArray(files) ? Array.from(files) : [files];
  if (!fileList.length || !fileList[0]) return;
  try {
    setBusy(true);
    const form = new FormData();
    for (const f of fileList) {
      form.append('file', f);
    }
    const body = await api('/api/upload', { method: 'POST', body: form });
    showParsed(body.data);
  } catch (error) { notify(error.message, true); }
  finally { setBusy(false); }
}
window.handleFileSelect = event => upload(event.target.files);

function params() { return { use_hybrid_rules: $('useHybridRulesCheckbox').checked, n_clusters: Number($('clusterCountSelect').value) }; }
async function analyze(payload) {
  try {
    setBusy(true);
    const body = await api('/api/analyze', jsonOptions({ ...payload, ...params() }));
    state.lastRequest = payload;
    state.analysis = body.data;
    renderAll();
    switchTab('dashboard');
    notify(`Analyzed ${body.data.summary.total_records} records`);
  } catch (error) { notify(error.message, true); }
  finally { setBusy(false); }
}
window.runUploadedFileAnalysis = function () {
  if (!state.parsed?.records?.length) return notify('Import a nonempty file first.', true);
  const column_mapping = Object.fromEntries([...document.querySelectorAll('.mapping-select')].filter(el => el.value).map(el => [el.dataset.field, el.value]));
  analyze({ records: state.parsed.records, column_mapping });
};

function renderAll() {
  const analysis = state.analysis;
  if (!analysis) return;
  const summary = analysis.summary;
  $('dashboardBadge').textContent = summary.total_records;
  $('recordsBadge').textContent = summary.total_records;
  $('kpiTotalRecords').textContent = summary.total_records;
  $('kpiTotalLatency').textContent = `${number(summary.total_latency_ms)} ms total inference time`;
  $('kpiDominantTheme').textContent = summary.dominant_theme;
  $('kpiThemeCountSub').textContent = `Across ${state.themes.length} codebook themes`;
  const sustained = summary.sustained_count !== undefined ? summary.sustained_count : summary.employed_count;
  const constrained = summary.constrained_count !== undefined ? summary.constrained_count : summary.underemployed_count;
  const deficient = summary.deficient_count !== undefined ? summary.deficient_count : summary.unemployed_count;
  const rate = summary.institutionalization_rate !== undefined ? summary.institutionalization_rate : summary.employment_alignment_rate;

  $('kpiEmploymentRate').textContent = `${number(rate)}%`;
  $('kpiOutcomeCounts').textContent = `${sustained} Sustained · ${constrained} Constrained · ${deficient} Deficient`;
  $('kpiMeanConfidence').textContent = `${number(summary.mean_confidence_pct)}%`;
  $('kpiAvgLatency').textContent = `${number(summary.avg_latency_ms, 2)} ms`;
  $('insightsContainer').innerHTML = analysis.qualitative_insights.map(item => `<div class="insight-item">${esc(item)}</div>`).join('');
  if ($('crossCaseMatrixBody')) {
    const rawMatrix = analysis.cross_case_matrix || [];
    const cases = rawMatrix.cases || [
      { id: 'lgu_01', name: 'LGU-01' },
      { id: 'lgu_02', name: 'LGU-02' },
      { id: 'lgu_03', name: 'LGU-03' }
    ];
    const rows = rawMatrix.rows || (Array.isArray(rawMatrix) ? rawMatrix : []);
    if ($('crossCaseMatrixHead')) {
      $('crossCaseMatrixHead').innerHTML = `<tr>
        <th style="min-width:180px;">Operational Issue</th>
        ${cases.map(c => `<th style="min-width:200px;">${esc(c.name)}</th>`).join('')}
        <th style="min-width:240px;">Cross-Case Interpretation</th>
      </tr>`;
    }
    $('crossCaseMatrixBody').innerHTML = rows.map(m => `<tr>
      <td><strong style="color:var(--text-primary);">${esc(m.issue)}</strong></td>
      ${cases.map(c => `<td style="font-size:0.83rem; color:var(--text-secondary);">${esc(m[c.id] || m[c.name] || '—')}</td>`).join('')}
      <td style="font-size:0.83rem; color:var(--accent);">${esc(m.interpretation)}</td>
    </tr>`).join('') || '<tr><td colspan="5">No cross-case matrix available.</td></tr>';
  }
  if ($('respondentGroupTableBody')) {
    const rgb = analysis.respondent_group_breakdown;
    if (rgb && rgb.themes && rgb.groups) {
      const themes = rgb.themes;
      const counts = rgb.counts || {};
      const totals = rgb.group_totals || {};
      const rowsHtml = themes.map(themeName => {
        const mayorCount = counts['Mayor / Admin']?.[themeName] || 0;
        const focalCount = counts['LCE / Focal']?.[themeName] || 0;
        const dilgCount = counts['DILG / PLGU']?.[themeName] || 0;
        const total = mayorCount + focalCount + dilgCount;
        let concentration = '<span style="color:var(--text-muted);">None</span>';
        if (total > 0) {
          const maxVal = Math.max(mayorCount, focalCount, dilgCount);
          const topGroups = [];
          if (mayorCount === maxVal) topGroups.push('Mayor/Admin');
          if (focalCount === maxVal) topGroups.push('LCE/Focal');
          if (dilgCount === maxVal) topGroups.push('DILG/PLGU');
          const pct = Math.round((maxVal / total) * 100);
          concentration = `<strong style="color:var(--text-primary);">${esc(topGroups.join(' & '))}</strong> <small style="color:var(--text-muted);">(${pct}%)</small>`;
        }
        return `<tr>
          <td><strong style="color:var(--text-primary);">${esc(themeName)}</strong></td>
          <td style="text-align:center;"><span class="badge-group badge-group-mayor">${mayorCount}</span></td>
          <td style="text-align:center;"><span class="badge-group badge-group-focal">${focalCount}</span></td>
          <td style="text-align:center;"><span class="badge-group badge-group-dilg">${dilgCount}</span></td>
          <td style="text-align:center;"><strong>${total}</strong></td>
          <td>${concentration}</td>
        </tr>`;
      }).join('');
      const totalsRow = `<tr style="background:var(--bg-tertiary); font-weight:800; border-top:2px solid var(--border-color);">
        <td>Total Responses by Group</td>
        <td style="text-align:center;"><span class="badge-group badge-group-mayor">${totals['Mayor / Admin'] || 0}</span></td>
        <td style="text-align:center;"><span class="badge-group badge-group-focal">${totals['LCE / Focal'] || 0}</span></td>
        <td style="text-align:center;"><span class="badge-group badge-group-dilg">${totals['DILG / PLGU'] || 0}</span></td>
        <td style="text-align:center;"><strong>${analysis.summary.total_records}</strong></td>
        <td style="color:var(--accent);">Active Stakeholder Corpus</td>
      </tr>`;
      $('respondentGroupTableBody').innerHTML = rowsHtml + totalsRow;
    } else {
      $('respondentGroupTableBody').innerHTML = '<tr><td colspan="6">No respondent group data available.</td></tr>';
    }
  }
  $('themeDistributionChart').innerHTML = analysis.theme_distribution.map(theme => `<div class="theme-bar-row"><div class="theme-bar-top"><span>${esc(theme.theme)}</span><span>${theme.count} · ${number(theme.percentage)}%</span></div><div class="theme-bar-track"><div class="theme-bar-fill" style="width:${Math.min(100, theme.percentage)}%;background:${esc(theme.color)}"></div></div><div class="theme-kw-pills">${(theme.top_keywords || []).slice(0, 5).map(k => `<span class="kw-pill">${esc(k.term)}</span>`).join('')}</div></div>`).join('') || '<p>No themes detected.</p>';
  $('crossTabTable').querySelector('tbody').innerHTML = analysis.theme_distribution.map(t => {
    const sust = t.sustained !== undefined ? t.sustained : t.employed;
    const constr = t.constrained !== undefined ? t.constrained : t.underemployed;
    const defic = t.deficient !== undefined ? t.deficient : t.unemployed;
    const pct = t.count ? number(sust / t.count * 100, 0) : 0;
    return `<tr>
      <td><strong style="color:var(--text-primary);">${esc(t.theme)}</strong></td>
      <td>${t.count}</td>
      <td><span class="status-badge status-sustained">${sust}</span></td>
      <td><span class="status-badge status-constrained">${constr}</span></td>
      <td><span class="status-badge status-deficient">${defic}</span></td>
      <td>${pct}% institutionalized</td>
    </tr>`;
  }).join('');
  $('emergentClustersContainer').innerHTML = (analysis.emergent_clusters || []).map(c => `<div class="cluster-item"><strong>${esc(c.label)}</strong><span>${c.size} records · ${esc(c.dominant_outcome)}</span><small>${esc((c.top_terms || []).join(' · '))}</small></div>`).join('') || '<p>At least three substantive records are needed for clustering.</p>';
  $('globalKeyphrasesContainer').innerHTML = `<div class="theme-kw-pills">${(analysis.global_tfidf_keyphrases || []).map(k => `<span class="kw-pill">${esc(k.term)} <b>${number(k.weight, 2)}</b></span>`).join('')}</div><h4>Matched codebook phrases</h4><div class="theme-kw-pills">${(analysis.top_matched_codes || []).map(k => `<span class="kw-pill">${esc(k.code)} (${k.count})</span>`).join('')}</div>`;
  updateFilters(); renderRecordsTable();
}
function statusBadge(status) {
  if (status === 'employed' || status === 'sustained') {
    return `<span class="status-badge status-sustained">Sustained</span>`;
  }
  if (status === 'underemployed' || status === 'constrained') {
    return `<span class="status-badge status-constrained">Constrained</span>`;
  }
  return `<span class="status-badge status-deficient">Deficient</span>`;
}
function groupBadge(group) {
  if (!group) return '<span style="color:var(--text-muted);">—</span>';
  let cls = 'badge-group-focal';
  if (group.includes('Mayor') || group.includes('Admin')) cls = 'badge-group-mayor';
  else if (group.includes('DILG') || group.includes('PLGU')) cls = 'badge-group-dilg';
  return `<span class="badge-group ${cls}">${esc(group)}</span>`;
}
function updateFilters() {
  const themes = [...new Set(state.analysis.records.map(row => row.primary_theme))].sort();
  const clusters = [...new Set(state.analysis.records.map(row => row.emergent_cluster))].sort();
  for (const [id, values, label] of [['filterThemeSelect', themes, 'All Primary Themes'], ['filterClusterSelect', clusters, 'All Emergent Clusters']]) {
    const select = $(id), previous = select.value;
    select.innerHTML = `<option value="ALL">${label}</option>${values.map(value => `<option value="${esc(value)}">${esc(value)}</option>`).join('')}`;
    if (values.includes(previous)) select.value = previous;
  }
}
window.renderRecordsTable = function () {
  if (!state.analysis) return;
  const query = $('recordSearchInput').value.trim().toLowerCase();
  const theme = $('filterThemeSelect').value;
  const group = $('filterGroupSelect') ? $('filterGroupSelect').value : 'ALL';
  const status = $('filterStatusSelect').value;
  const cluster = $('filterClusterSelect').value;
  const minConfidence = Number($('filterConfidenceSelect').value);
  const rows = state.analysis.records.filter(row =>
    (theme === 'ALL' || row.primary_theme === theme) &&
    (group === 'ALL' || row.respondent_group === group) &&
    (status === 'ALL' || row.final_status === status || row.governance_status_key === status) &&
    (cluster === 'ALL' || row.emergent_cluster === cluster) &&
    Number(row.confidence) >= minConfidence &&
    (!query || [row.identifier, row.input_text, row.primary_theme, row.secondary_theme, row.respondent_group, row.position, row.skills, row.governance_status, ...(row.matched_codes || [])].join(' ').toLowerCase().includes(query))
  );
  $('codedRecordsTable').querySelector('tbody').innerHTML = rows.map(row => `<tr>
    <td>${row.row_index}</td>
    <td><strong>${esc(row.identifier)}</strong><br><small>${esc(row.position || '—')}</small></td>
    <td>${groupBadge(row.respondent_group)}</td>
    <td><span class="theme-chip">${esc(row.primary_theme)}</span><br><small>${esc(row.secondary_theme || '')}</small></td>
    <td>${esc((row.matched_codes || []).join(', ') || '—')}</td>
    <td>${esc(row.emergent_cluster || '—')}</td>
    <td>${statusBadge(row.final_status)}</td>
    <td>${number(row.confidence * 100)}%</td>
    <td>${row.token_count} / ${number(row.latency_ms, 2)} ms</td>
    <td><button class="btn btn-secondary btn-sm" onclick="openInspector(${row.row_index})">Inspect</button></td>
  </tr>`).join('') || '<tr><td colspan="10">No records match the filters.</td></tr>';
};
window.openInspector = function (rowIndex) {
  const row = state.analysis?.records.find(item => item.row_index === rowIndex);
  if (!row) return;
  $('modalRecordTitle').textContent = `${row.identifier} · Deep Inspector`;
  const probabilities = Object.entries(row.probabilities || {}).map(([label, probability]) => {
    let govLabel = label;
    if (label === 'employed') govLabel = 'Sustained / Institutionalized';
    else if (label === 'underemployed') govLabel = 'Constrained / In-Progress';
    else if (label === 'unemployed') govLabel = 'Deficient / Unaligned';
    return `<div class="probability-row"><span>${esc(govLabel)}</span><div class="theme-bar-track"><div class="theme-bar-fill" style="width:${probability * 100}%;background:var(--accent)"></div></div><b>${number(probability * 100)}%</b></div>`;
  }).join('');
  const tokens = (row.tokens_preview || []).map((token, i) => `<span class="token-chip ${token.startsWith('[') ? 'special' : ''}">${esc(token)}<span class="token-id">${esc(row.token_ids_preview?.[i])}</span></span>`).join('');
  const govStatus = row.governance_status || (row.final_status === 'employed' ? 'Sustained / Institutionalized' : (row.final_status === 'underemployed' ? 'Constrained / In-Progress' : 'Deficient / Unaligned'));
  $('modalInspectorBody').innerHTML = `<div class="inspector-grid"><div><h4>Standardized qualitative input</h4><p class="inspector-text">${esc(row.input_text)}</p><h4>DistilBERT sequence output</h4><p>Governance Trajectory: <strong>${esc(govStatus)}</strong>${row.hybrid_override ? ' · Hybrid rule active' : ''}</p><p>Raw logits: ${esc((row.logits || []).join(' · '))}</p>${probabilities}<p>${row.token_count} tokens · ${number(row.latency_ms, 2)} ms</p></div><div><h4>Thematic coding</h4><p>Primary: <strong>${esc(row.primary_theme)}</strong></p><p>Secondary: ${esc(row.secondary_theme || 'None')}</p><p>Respondent group: <strong>${groupBadge(row.respondent_group)}</strong></p><p>Matched codes: ${esc((row.matched_codes || []).join(', ') || 'None')}</p><p>Governance trajectory tone: <strong>${esc(row.trajectory_tone || 'Standard')}</strong></p><label class="form-label" for="manualThemeSelect">Manual theme override</label><select id="manualThemeSelect" class="select">${state.themes.map(t => `<option value="${esc(t.id)}" ${row.primary_theme_id === t.id ? 'selected' : ''}>${esc(t.name)}</option>`).join('')}</select><button class="btn btn-primary btn-sm" onclick="applyThemeOverride(${row.row_index})">Apply override</button></div></div><h4>WordPiece subword token preview</h4><div class="token-list">${tokens}</div>`;
  $('inspectorModal').classList.add('open');
};
window.closeInspectorModal = function () { $('inspectorModal').classList.remove('open'); };
window.applyThemeOverride = function (rowIndex) {
  const row = state.analysis?.records.find(item => item.row_index === rowIndex);
  const theme = state.themes.find(item => item.id === $('manualThemeSelect').value);
  if (!row || !theme) return;
  row.primary_theme_id = theme.id; row.primary_theme = theme.name; row.primary_theme_color = theme.color;
  row.manual_theme_override = true;
  rebuildThemeSummary(); renderAll(); openInspector(rowIndex);
  notify('Theme override applied to this analysis. Export to preserve it.');
};
function rebuildThemeSummary() {
  const analysis = state.analysis, rows = analysis.records;
  const counts = new Map();
  for (const row of rows) {
    if (!counts.has(row.primary_theme)) counts.set(row.primary_theme, { theme: row.primary_theme, color: row.primary_theme_color, description: state.themes.find(t => t.name === row.primary_theme)?.description || '', count: 0, employed: 0, underemployed: 0, unemployed: 0, top_keywords: [] });
    const theme = counts.get(row.primary_theme); theme.count++;
    if (Object.hasOwn(theme, row.final_status)) theme[row.final_status]++;
  }
  analysis.theme_distribution = [...counts.values()].sort((a, b) => b.count - a.count).map(t => ({ ...t, percentage: Number((t.count / rows.length * 100).toFixed(1)) }));
  analysis.summary.dominant_theme = analysis.theme_distribution[0]?.theme || 'None';
  analysis.qualitative_insights[0] = `Dominant pattern: ${analysis.summary.dominant_theme} (${analysis.theme_distribution[0]?.count || 0} of ${rows.length} records). Manual overrides included.`;

  if (analysis.respondent_group_breakdown) {
    const rgb = analysis.respondent_group_breakdown;
    const newCounts = {
      'Mayor / Admin': {},
      'LCE / Focal': {},
      'DILG / PLGU': {}
    };
    for (const t of (rgb.themes || [])) {
      for (const g of (rgb.groups || [])) {
        if (!newCounts[g]) newCounts[g] = {};
        newCounts[g][t] = 0;
      }
    }
    for (const r of rows) {
      const g = r.respondent_group;
      const t = r.primary_theme;
      if (newCounts[g] && newCounts[g][t] !== undefined) {
        newCounts[g][t]++;
      }
    }
    rgb.counts = newCounts;
  }
}

window.exportAnalysis = async function (format) {
  if (!state.analysis) return notify('Analyze records before exporting.', true);
  try {
    const response = await fetch('/api/export', jsonOptions({ format, analysis: state.analysis }));
    if (!response.ok) { const body = await response.json(); throw new Error(body.detail || 'Export failed'); }
    const blob = await response.blob();
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob); link.download = `sglg_thematic_analysis.${format}`;
    link.click(); setTimeout(() => URL.revokeObjectURL(link.href), 1000);
  } catch (error) { notify(error.message, true); }
};

window.loadSampleDataset = async function (format = 'csv') {
  try {
    setBusy(true);
    const body = await api(`/api/samples/${format}`);
    showParsed(body.data);
    await runUploadedFileAnalysis();
    notify("Loaded and analyzed Teacher's SGLG Dataset (18 records across 3 LGU cases).");
  } catch (error) { notify(error.message, true); }
  finally { setBusy(false); }
};

window.handlePresetSelect = async function (presetId) {
  try {
    setBusy(true);
    const body = await api(`/api/presets/${presetId}`, { method: 'POST' });
    state.themes = body.themes;
    renderThemes();
    notify(`Switched codebook preset to '${presetId === 'sglg_governance' ? "Teacher's SGLG Governance Study" : 'Alumni Career Tracer'}'`);
    if (state.lastRequest) await analyze(state.lastRequest);
  } catch (error) { notify(error.message, true); }
  finally { setBusy(false); }
};

function renderThemes() {
  $('codebookEditorContainer').innerHTML = state.themes.map((theme, index) => {
    const subthemeTags = (theme.subthemes || []).map(s => `<span class="subtheme-chip">${esc(s)}</span>`).join('');
    const sequenceHtml = theme.writing_sequence ? `<div class="theme-meta-note"><strong>Sequence:</strong> ${esc(theme.writing_sequence)}</div>` : '';
    const quoteHtml = theme.sample_quote ? `<div class="theme-meta-quote"><strong>Sample Quote:</strong> ${esc(theme.sample_quote)}</div>` : '';
    const benchHtml = theme.suggested_analytic_statement ? `<div class="theme-meta-bench"><strong>Benchmark:</strong> “${esc(theme.suggested_analytic_statement)}”</div>` : '';
    const conclHtml = theme.suggested_theme_conclusion ? `<div class="theme-meta-bench"><strong>Conclusion:</strong> “${esc(theme.suggested_theme_conclusion)}”</div>` : '';

    return `
      <div class="theme-editor" data-index="${index}">
        <div class="theme-editor-head">
          <input class="input theme-name" aria-label="Theme name" value="${esc(theme.name)}">
          <input class="theme-color" aria-label="Theme color" type="color" value="${esc(theme.color)}">
          <button class="btn btn-secondary btn-sm" onclick="removeThemeCard(${index})">Remove</button>
        </div>
        <label class="form-label">Theme Description
          <textarea class="textarea theme-description" style="min-height:55px;">${esc(theme.description)}</textarea>
        </label>
        ${subthemeTags ? `<div class="subtheme-wrapper"><span class="subtheme-label">Subthemes &amp; Key Dimensions:</span><div class="subtheme-pills">${subthemeTags}</div></div>` : ''}
        ${sequenceHtml}
        ${quoteHtml}
        ${benchHtml}
        ${conclHtml}
        <label class="form-label" style="margin-top:8px;">Seed Keywords (comma-separated for matching)
          <textarea class="textarea theme-keywords">${esc((theme.keywords || []).join(', '))}</textarea>
        </label>
      </div>
    `;
  }).join('');
}

function collectThemes() {
  return [...document.querySelectorAll('.theme-editor')].map((el, index) => {
    const prev = state.themes[index] || {};
    return {
      id: prev.id,
      name: el.querySelector('.theme-name').value.trim(),
      color: el.querySelector('.theme-color').value,
      description: el.querySelector('.theme-description').value.trim(),
      subthemes: prev.subthemes || [],
      writing_sequence: prev.writing_sequence || '',
      sample_quote: prev.sample_quote || '',
      suggested_analytic_statement: prev.suggested_analytic_statement || '',
      suggested_theme_conclusion: prev.suggested_theme_conclusion || '',
      keywords: el.querySelector('.theme-keywords').value.split(',').map(s => s.trim()).filter(Boolean),
    };
  }).filter(t => t.name);
}
window.addNewThemeCard = function () {
  state.themes = collectThemes();
  state.themes.push({ id: `custom_${Date.now()}`, name: 'New Custom Theme', color: '#14b8a6', description: '', subthemes: [], keywords: [] });
  renderThemes();
};
window.removeThemeCard = function (index) {
  state.themes = collectThemes().filter((_, i) => i !== index); renderThemes();
};
window.resetCodebookThemes = async function () {
  try {
    state.themes = (await api('/api/themes/reset', { method: 'POST' })).themes;
    renderThemes();
    if ($('codebookPresetSelect')) $('codebookPresetSelect').value = 'sglg_governance';
    notify("Default Teacher's SGLG themes restored.");
  } catch (error) { notify(error.message, true); }
};
window.saveAndReanalyzeThemes = async function () {
  try {
    const themes = collectThemes();
    if (!themes.length) throw new Error('Keep at least one theme.');
    state.themes = (await api('/api/themes', jsonOptions({ themes }))).themes;
    renderThemes();
    if (state.lastRequest) await analyze(state.lastRequest);
    else notify('Codebook saved. Analyze a dataset to apply it.');
  } catch (error) { notify(error.message, true); }
};
function renderModel(model) {
  const entries = [
    ['Runtime', model.runtime], ['Model directory', model.model_dir],
    ['Architecture', model.architecture], ['Layers', model.num_layers],
    ['Attention heads', model.num_attention_heads], ['Hidden dimensions', model.hidden_dim],
    ['Vocabulary', model.vocab_size], ['Max input tokens', model.max_sequence_length],
  ];
  $('modelSpecsContainer').innerHTML = `<div class="spec-grid">${entries.map(([label, value]) => `<div><small>${esc(label)}</small><strong>${esc(value)}</strong></div>`).join('')}</div>${model.load_error ? `<p class="model-warning">${esc(model.load_error)}</p>` : ''}`;
  $('modelStatusText').textContent = model.loaded ? (model.runtime.startsWith('ONNX') ? 'ONNX Engine Ready' : 'PyTorch Engine Ready') : 'Heuristic Fallback Active';
  $('modelStatusPill').classList.toggle('offline', !model.loaded);
  $('modelArchSubtitle').textContent = model.loaded ? `DistilBERT Transformer · ${model.model_size_mb} MB · Study 3 SGLG Qualitative Inference` : 'Model unavailable · heuristic estimates only';
}

async function initialize() {
  if (localStorage.getItem('thematic-theme') === 'light') toggleThemeMode();
  const dropzone = $('dropzone');
  for (const event of ['dragenter', 'dragover']) dropzone.addEventListener(event, e => { e.preventDefault(); dropzone.classList.add('dragover'); });
  for (const event of ['dragleave', 'drop']) dropzone.addEventListener(event, e => { e.preventDefault(); dropzone.classList.remove('dragover'); });
  dropzone.addEventListener('drop', e => upload(e.dataTransfer.files));
  try {
    const status = await api('/api/status');
    state.themes = status.themes;
    renderThemes();
    renderModel(status.model);
    if ($('codebookPresetSelect')) $('codebookPresetSelect').value = 'sglg_governance';
  } catch (error) {
    $('modelStatusText').textContent = 'Server unavailable';
    $('modelStatusPill').classList.add('offline');
    notify(error.message, true);
  }
}
initialize();
