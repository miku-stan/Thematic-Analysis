/* Thematic Analysis Studio: local, browser-side orchestration and rendering. */
'use strict';

const $ = (id) => document.getElementById(id);
const state = { parsed: null, analysis: null, themes: [], lastRequest: null, busy: false, narrative: null, narrativeCase: 'all' };
const fields = [
  ['full_name', 'Identifier / Participant ID'],
  ['headline', 'Subtheme / Interview Topic'],
  ['current_position', 'Office / Role / Designation'],
  ['current_company', 'LGU / City / Municipality'],
  ['degree_program', 'Governance Area / SGLG Pillar'],
  ['skills', 'Indicators / Criteria / Operational Codes'],
  ['about', 'Narrative Quote / Transcript Excerpt'],
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
  if (name === 'narrative' && state.analysis && !state.narrative) {
    fetchNarrative(state.narrativeCase || 'all');
  }
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
  state.analysis = null;
  state.narrative = null;
  $('fileInput').value = '';
  $('uploadPreviewBox').style.display = 'none';
  if ($('narrativeEmptyPlaceholder')) $('narrativeEmptyPlaceholder').style.display = '';
  if ($('narrativeManuscriptBox')) $('narrativeManuscriptBox').style.display = 'none';
  if ($('narrativeBadge')) $('narrativeBadge').textContent = '0';
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
  const sustained = summary.sustained_count || 0;
  const constrained = summary.constrained_count || 0;
  const deficient = summary.deficient_count || 0;
  const rate = summary.institutionalization_rate || 0;

  if ($('kpiEmploymentRate')) $('kpiEmploymentRate').textContent = `${number(rate)}%`;
  if ($('kpiInstitutionalizationRate')) $('kpiInstitutionalizationRate').textContent = `${number(rate)}%`;
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
    const sust = t.sustained || 0;
    const constr = t.constrained || 0;
    const defic = t.deficient || 0;
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
  if ($('narrativeBadge')) $('narrativeBadge').textContent = summary.total_records;
  if ($('narrativeCaseFilterSelect') && analysis.records) {
    const lguSet = new Set();
    analysis.records.forEach(r => {
      const v = r.company || r.lgu || (r.raw_record && (r.raw_record.current_company || r.raw_record.lgu));
      if (v) lguSet.add(v);
    });
    const lgus = Array.from(lguSet).sort();
    const curVal = $('narrativeCaseFilterSelect').value;
    $('narrativeCaseFilterSelect').innerHTML = `<option value="all">All Participating LGUs (${lgus.length} Cases)</option>` + lgus.map(c => `<option value="${esc(c)}">${esc(c)}</option>`).join('');
    if (lgus.includes(curVal)) $('narrativeCaseFilterSelect').value = curVal;
  }
  fetchNarrative(state.narrativeCase || 'all');
}
function statusBadge(status) {
  if (status === 'sustained') {
    return `<span class="status-badge status-sustained">Sustained</span>`;
  }
  if (status === 'constrained') {
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
    <td><strong>${esc(row.identifier)}</strong>${row.is_taglish ? '<span class="badge" style="background:#0ea5e920;color:#0284c7;border:1px solid #0284c7;font-size:10px;margin-left:5px;padding:1px 5px;" title="Taglish Detected">Taglish</span>' : ''}<br><small>${esc(row.position || '—')}</small></td>
    <td>${groupBadge(row.respondent_group)}</td>
    <td><span class="theme-chip" style="background:${row.primary_theme_color ? row.primary_theme_color + '22' : 'var(--accent-glow)'}; color:${row.primary_theme_color || 'var(--accent)'}; border-color:${row.primary_theme_color || 'var(--accent)'};">${esc(row.primary_theme)}</span><br><small>${esc(row.secondary_theme || '')}</small></td>
    <td>${(row.matched_codes || []).map(c => c.startsWith('[Emergent]') ? `<span style="display:inline-block; margin:2px; padding:1px 6px; border-radius:4px; background:#fef3c7; color:#92400e; font-size:11px; font-weight:600;">${esc(c)}</span>` : `<span style="display:inline-block; margin:2px; padding:1px 6px; border-radius:4px; background:#f1f5f9; color:#334155; font-size:11px;">${esc(c)}</span>`).join('') || '—'}</td>
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
    if (label === 'sustained') govLabel = 'Sustained / Institutionalized';
    else if (label === 'constrained') govLabel = 'Constrained / In-Progress';
    else if (label === 'deficient') govLabel = 'Deficient / Unaligned';
    return `<div class="probability-row"><span>${esc(govLabel)}</span><div class="theme-bar-track"><div class="theme-bar-fill" style="width:${probability * 100}%;background:var(--accent)"></div></div><b>${number(probability * 100)}%</b></div>`;
  }).join('');
  const tokens = (row.tokens_preview || []).map((token, i) => `<span class="token-chip ${token.startsWith('[') ? 'special' : ''}">${esc(token)}<span class="token-id">${esc(row.token_ids_preview?.[i])}</span></span>`).join('');
  const govStatus = row.governance_status || (row.final_status === 'sustained' ? 'Sustained / Institutionalized' : (row.final_status === 'constrained' ? 'Constrained / In-Progress' : 'Deficient / Unaligned'));
  const matchedCodesList = (row.matched_codes || []).map(c => c.startsWith('[Emergent]') ? `<span style="display:inline-block; margin:2px; padding:2px 8px; border-radius:12px; background:#fef3c7; color:#92400e; font-size:11px; font-weight:700;">${esc(c)}</span>` : `<span style="display:inline-block; margin:2px; padding:2px 8px; border-radius:12px; background:#e0f2fe; color:#0369a1; font-size:11px; font-weight:600;">${esc(c)}</span>`).join(' ') || 'None';
  $('modalInspectorBody').innerHTML = `<div class="inspector-grid"><div><h4>Standardized qualitative input</h4><p class="inspector-text">${esc(row.input_text)}</p><h4>DistilBERT sequence output</h4><p>Governance Trajectory: <strong>${esc(govStatus)}</strong>${row.hybrid_override ? ' · Hybrid rule active' : ''}</p><p>Language: <strong>${row.is_taglish ? '<span style="color:#0284c7;">Taglish (Morphological Normalizer Active)</span>' : '<span style="color:var(--text-secondary);">English</span>'}</strong></p><p>Raw logits: ${esc((row.logits || []).join(' · '))}</p>${probabilities}<p>${row.token_count} tokens · ${number(row.latency_ms, 2)} ms</p></div><div><h4>Thematic coding</h4><p>Primary: <strong>${esc(row.primary_theme)}</strong></p><p>Secondary: ${esc(row.secondary_theme || 'None')}</p><p>Respondent group: <strong>${groupBadge(row.respondent_group)}</strong></p><p>Matched codes: ${matchedCodesList}</p><p>Governance trajectory tone: <strong>${esc(row.trajectory_tone || 'Standard')}</strong></p><label class="form-label" for="manualThemeSelect">Manual theme override</label><select id="manualThemeSelect" class="select">${state.themes.map(t => `<option value="${esc(t.id)}" ${row.primary_theme_id === t.id ? 'selected' : ''}>${esc(t.name)}</option>`).join('')}</select><button class="btn btn-primary btn-sm" onclick="applyThemeOverride(${row.row_index})">Apply override</button></div></div><h4>WordPiece subword token preview</h4><div class="token-list">${tokens}</div>`;
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
    if (!counts.has(row.primary_theme)) counts.set(row.primary_theme, { theme: row.primary_theme, color: row.primary_theme_color, description: state.themes.find(t => t.name === row.primary_theme)?.description || '', count: 0, sustained: 0, constrained: 0, deficient: 0, top_keywords: [] });
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
    setBusy(true);
    const response = await fetch('/api/export', jsonOptions({ format, analysis: state.analysis }));
    if (!response.ok) {
      const body = await response.json().catch(() => ({}));
      throw new Error(body.detail || 'Export failed');
    }
    const blob = await response.blob();
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    const ext = (format === 'doc' || format === 'docx') ? 'doc' : format;
    link.download = `sglg_thematic_analysis_report.${ext}`;
    document.body.appendChild(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(link.href), 1000);
    notify(`Exported ${format.toUpperCase()} report successfully.`);
  } catch (error) {
    notify(error.message, true);
  } finally {
    setBusy(false);
  }
};

window.printCurrentTab = function () {
  if (!state.analysis) {
    return notify('Analyze records first before printing report.', true);
  }
  window.print();
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
    notify(`Switched codebook preset to '${body.name || "Teacher's SGLG Governance Study"}'`);
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
    ['Vocabulary', `${model.vocab_size} (WordPiece uncased)`], ['Max input tokens', model.max_sequence_length],
    ['Taglish Linguistic Adapter', model.taglish_support?.active ? 'Active (Morphological Normalizer)' : 'Standard English'],
    ['Affix Handling', model.taglish_support?.affix_handling ? 'nag-, mag-, pag-, na-, i-, -in-, -an' : 'None'],
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

/* ==========================================================================
   QUALITATIVE THEMATIC ANALYSIS NARRATIVE SYNTHESIS ENGINE
   ========================================================================== */
window.fetchNarrative = async function (caseFilter = 'all') {
  if (!state.analysis) {
    if ($('narrativeEmptyPlaceholder')) $('narrativeEmptyPlaceholder').style.display = '';
    if ($('narrativeManuscriptBox')) $('narrativeManuscriptBox').style.display = 'none';
    return;
  }
  try {
    const res = await api('/api/narrative', jsonOptions({
      analysis: state.analysis,
      case_filter: caseFilter === 'all' ? null : caseFilter,
    }));
    state.narrative = res.data;
    renderNarrative(res.data);
  } catch (err) {
    notify('Failed to generate narrative draft: ' + err.message, true);
  }
};

window.renderNarrative = function (data) {
  if (!data || !data.sections || !data.sections.length) {
    if ($('narrativeEmptyPlaceholder')) $('narrativeEmptyPlaceholder').style.display = '';
    if ($('narrativeManuscriptBox')) $('narrativeManuscriptBox').style.display = 'none';
    return;
  }
  if ($('narrativeEmptyPlaceholder')) $('narrativeEmptyPlaceholder').style.display = 'none';
  if ($('narrativeManuscriptBox')) $('narrativeManuscriptBox').style.display = '';

  const meta = data.metadata || {};
  const casesText = (meta.cases_analyzed || []).join(', ');

  let html = `
    <div class="manuscript-header">
      <div class="manuscript-title">${esc(data.title)}</div>
      <div class="manuscript-subtitle">${esc(data.subtitle)}</div>
      <div class="manuscript-meta-box">
        <div class="manuscript-meta-item">
          <strong>Corpus Scope:</strong> <span>${meta.total_records || 0} Records</span>
        </div>
        <div class="manuscript-meta-item">
          <strong>Cases Analyzed:</strong> <span>${esc(casesText || 'Multi-LGU')}</span>
        </div>
        <div class="manuscript-meta-item">
          <strong>Dominant Theme:</strong> <span>${esc(meta.dominant_theme || '—')}</span>
        </div>
        <div class="manuscript-meta-item">
          <strong>Institutionalization Rate:</strong> <span>${meta.institutionalization_rate || 0}% Sustained</span>
        </div>
      </div>
    </div>

    <!-- Table of Analytical Sections -->
    <div class="manuscript-toc">
      <div class="manuscript-toc-title">Table of Analytical Sections</div>
      <ul class="manuscript-toc-list">
        ${data.sections.map(sec => `
          <li>
            <a class="manuscript-toc-link" href="#sec_${esc(sec.id)}">
              <span style="font-weight:700;">${esc(sec.section_num)}</span>
              <span>${esc(sec.title)}</span>
            </a>
          </li>
        `).join('')}
      </ul>
    </div>
  `;

  // Render each section
  data.sections.forEach(sec => {
    const secColor = sec.color || '#6366f1';
    html += `
      <section class="manuscript-section" id="sec_${esc(sec.id)}">
        <div class="manuscript-heading">
          <span class="manuscript-num-badge" style="background:${secColor}20; color:${secColor};">${esc(sec.section_num)}</span>
          <span>${esc(sec.title)}</span>
        </div>
    `;

    if (sec.benchmark) {
      html += `
        <div class="manuscript-benchmark-callout" style="border-left-color:${secColor};">
          <span class="manuscript-benchmark-tag">Core Analytic Proposition / Benchmark</span>
          “${esc(sec.benchmark)}”
        </div>
      `;
    }

    if (sec.paragraphs && sec.paragraphs.length) {
      sec.paragraphs.forEach(p => {
        const cleanP = (p || '').replace(/\*\*/g, '').replace(/\*/g, '');
        html += `<p class="manuscript-p">${esc(cleanP)}</p>`;
      });
    }

    if (sec.quotes && sec.quotes.length) {
      html += `<div class="manuscript-quotes-wrapper">`;
      sec.quotes.forEach(q => {
        html += `
          <div class="manuscript-quote-card" style="border-left-color:${secColor};">
            <div class="manuscript-quote-text">“${esc(q.quote)}”</div>
            <div class="manuscript-quote-footer">
              <div class="manuscript-participant-tag">
                ${esc(q.participant_id)} <span class="manuscript-participant-meta">· ${esc(q.office_role)}, ${esc(q.lgu)}</span>
              </div>
              <span class="status-badge status-${esc(q.status.toLowerCase())}">${esc(q.status)}</span>
            </div>
          </div>
        `;
      });
      html += `</div>`;
    }

    html += `</section>`;
  });

  $('narrativeContentContainer').innerHTML = html;
};

window.copyNarrativeDraft = async function () {
  if (!state.narrative) {
    return notify('Analyze a dataset first to generate narrative draft.', true);
  }
  try {
    const rawText = state.narrative.plain_text || state.narrative.markdown || '';
    const cleanText = rawText.replace(/\*\*/g, '').replace(/\*/g, '');
    await navigator.clipboard.writeText(cleanText);
    notify('Thematic Analysis Narrative Draft copied to clipboard! Ready to paste into Word or Google Docs.');
  } catch (err) {
    notify('Clipboard copy failed. Please use Download options.', true);
  }
};

window.downloadNarrativeFile = async function (format) {
  if (!state.analysis) {
    return notify('Analyze a dataset first before exporting narrative.', true);
  }
  try {
    setBusy(true);
    const caseFilter = state.narrativeCase === 'all' ? null : state.narrativeCase;
    const response = await fetch('/api/narrative/export', jsonOptions({
      analysis: state.analysis,
      format,
      case_filter: caseFilter,
    }));
    if (!response.ok) {
      const errBody = await response.json().catch(() => ({}));
      throw new Error(errBody.detail || 'Narrative export failed');
    }
    const blob = await response.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    const ext = format === 'doc' ? 'doc' : format;
    a.download = `Thematic_Analysis_Narrative_Synthesis.${ext}`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
    notify(`Downloaded Thematic Analysis Narrative (${format.toUpperCase()}) successfully.`);
  } catch (error) {
    notify(error.message, true);
  } finally {
    setBusy(false);
  }
};

window.printNarrativeManuscript = function () {
  if (!state.narrative) {
    return notify('Generate a narrative draft first before printing.', true);
  }
  window.print();
};

window.refreshNarrativeDraft = function () {
  if (!state.analysis) {
    return notify('Analyze a dataset first before regenerating narrative.', true);
  }
  fetchNarrative(state.narrativeCase || 'all');
  notify('Regenerating Thematic Analysis narrative synthesis draft...');
};

window.onNarrativeCaseFilterChange = function () {
  const sel = $('narrativeCaseFilterSelect');
  state.narrativeCase = sel ? sel.value : 'all';
  fetchNarrative(state.narrativeCase);
};
