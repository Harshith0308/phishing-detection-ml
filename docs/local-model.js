/*
 * Static-demo backend shim for GitHub Pages.
 * Re-implements the app.py endpoints (/metrics, /explain, /predict) fully
 * client-side, using the SAME Random Forest trained by app.py (trees exported
 * to model-data.json). Feature extraction is a faithful port of
 * app.py::extract_features. The original application code is unchanged.
 */
(function () {
  'use strict';

  let _dataPromise = null;
  function loadData() {
    if (!_dataPromise) {
      _dataPromise = window.__ORIG_FETCH__('model-data.json').then(r => r.json());
    }
    return _dataPromise;
  }

  // ---- faithful port of Python urlparse().hostname for our purposes ----
  function pyHostname(url) {
    let rest = String(url || '');
    const m = rest.match(/^[a-zA-Z][a-zA-Z0-9+.\-]*:/);
    if (m) rest = rest.slice(m[0].length);
    if (!rest.startsWith('//')) return '';
    let netloc = rest.slice(2).split(/[\/?#]/)[0];
    const at = netloc.lastIndexOf('@');
    if (at !== -1) netloc = netloc.slice(at + 1);
    if (netloc.startsWith('[')) {
      const j = netloc.indexOf(']');
      return (j >= 0 ? netloc.slice(1, j) : netloc.slice(1)).toLowerCase();
    }
    const c = netloc.indexOf(':');
    if (c !== -1) netloc = netloc.slice(0, c);
    return netloc.toLowerCase();
  }

  const IPV4_RE = /^(\d{1,3}\.){3}\d{1,3}$/;
  const IPV6_RE = /^[0-9a-f:]+$/i;

  // ---- port of app.py::extract_features ----
  function extractFeatures(url) {
    url = String(url || '');
    const host = pyHostname(url);
    return {
      'URL_Length': url.length < 54 ? 1 : -1,
      'having_At_Symbol': url.includes('@') ? -1 : 1,
      'having_IP_Address': (host && (IPV4_RE.test(host) || IPV6_RE.test(host))) ? -1 : 1,
      'Prefix_Suffix': host.includes('-') ? -1 : 1,
      'web_traffic': host.length < 20 ? 1 : 0
    };
  }

  // ---- port of app.py::build_full_feature_row ----
  function buildRow(feats, featureNames, colMap) {
    const row = new Array(featureNames.length).fill(0);
    const idx = {};
    featureNames.forEach((n, i) => { idx[n] = i; });
    for (const std of Object.keys(colMap)) {
      const actual = colMap[std];
      if (actual !== null && std in feats && actual in idx) {
        row[idx[actual]] = feats[std];
      }
    }
    return row;
  }

  // ---- RandomForestClassifier.predict (soft voting, like sklearn) ----
  function rfPredict(model, x) {
    const nc = model.classes.length;
    const acc = new Array(nc).fill(0);
    for (const t of model.trees) {
      let i = 0;
      while (t.feature[i] !== -2) {
        i = (x[t.feature[i]] <= t.threshold[i]) ? t.children_left[i] : t.children_right[i];
      }
      const counts = t.value[i];
      const s = counts.reduce((a, b) => a + b, 0) || 1;
      for (let k = 0; k < nc; k++) acc[k] += counts[k] / s;
    }
    let best = 0;
    for (let k = 1; k < nc; k++) if (acc[k] > acc[best]) best = k;
    return model.classes[best];
  }

  function jsonResponse(obj) {
    return new Response(JSON.stringify(obj), { status: 200, headers: { 'Content-Type': 'application/json' } });
  }

  async function handlePredict(url) {
    const D = await loadData();
    const feats = extractFeatures(url);
    const row = buildRow(feats, D.model.feature_names, D.col_map);
    const pred = rfPredict(D.model, row);
    return {
      label: pred >= 0 ? 'Safe' : 'Phishing',
      pred: pred,
      features: feats,
      model_input_columns: D.model.feature_names,
      model_input_row: row.map(Number)
    };
  }

  // ---- port of the /explain fallback explainer in app.py (importance × value) ----
  async function handleExplain(url) {
    const D = await loadData();
    const base = await handlePredict(url);
    const impMap = {};
    D.explain_global.rf_importances.forEach(it => { impMap[it.feature] = it.importance; });
    const contribs = D.model.feature_names.map((col, i) => {
      const importance = impMap[col] || 0;
      const value = base.model_input_row[i];
      return { column: col, value: value, importance: importance, weighted: value * importance };
    });
    contribs.sort((a, b) => Math.abs(b.weighted) - Math.abs(a.weighted));
    return {
      label: base.label,
      pred: base.pred,
      explainer: 'rf (feature importance × value — static demo)',
      contributions: contribs.slice(0, 10),
      rf_importances: D.explain_global.rf_importances.slice(0, 10)
    };
  }

  window.__ORIG_FETCH__ = window.fetch.bind(window);
  window.fetch = async function (input, init) {
    const u = typeof input === 'string' ? input : (input && input.url) || '';
    const method = ((init && init.method) || 'GET').toUpperCase();
    if (u === '/metrics') {
      const D = await loadData();
      return jsonResponse(D.metrics);
    }
    if (u === '/explain' && method === 'GET') {
      const D = await loadData();
      return jsonResponse(D.explain_global);
    }
    if (u === '/explain' && method === 'POST') {
      const body = JSON.parse((init && init.body) || '{}');
      return jsonResponse(await handleExplain(body.url || ''));
    }
    if (u === '/predict' && method === 'POST') {
      const body = JSON.parse((init && init.body) || '{}');
      return jsonResponse(await handlePredict(body.url || ''));
    }
    return window.__ORIG_FETCH__(input, init);
  };
})();
