/* KTBench gh-pages front-end.
 *
 * Vanilla JS. Loads /runs.json, applies the filter dropdowns, runs the
 * requested sort, and renders into one or more tables. The leaderboard
 * (index.html) calls renderLeaderboard(); the run index (runs.html)
 * calls renderRunsIndex(). Filter state is shared across pages within
 * the same tab so navigating between the two does not silently reset.
 *
 * Schema KTBench-specific: the leaderboard reports final_score and
 * sol_score (Speed-of-Light fraction) instead of speedup, because
 * speedup is gameable by slowing the baseline and KTBench scores
 * candidates against the hardware ceiling. final_score is the
 * primary ranking column.
 */

const KTBenchSite = (() => {
  let DATA = { runs: [], generated_at: null };
  let FILTER_STATE = { tgt_dsl: "", src_dsl: "", persona: "", model: "", scenario: "" };
  const SORT_STATE = new Map();

  const EM_DASH = "—";

  function fmtNumber(n, digits = 2) {
    if (n === null || n === undefined || !isFinite(n)) return EM_DASH;
    return Number(n).toFixed(digits);
  }
  function fmtPct(n, digits = 0) {
    if (n === null || n === undefined || !isFinite(n)) return EM_DASH;
    return (n * 100).toFixed(digits) + "%";
  }
  function fmtInt(n) {
    if (n === null || n === undefined) return EM_DASH;
    return String(n);
  }
  function fmtStr(s) {
    if (s === null || s === undefined || s === "") return EM_DASH;
    return s;
  }

  async function loadData() {
    if (DATA.runs.length) return DATA;
    const r = await fetch("runs.json", { cache: "no-store" });
    if (!r.ok) throw new Error("runs.json fetch failed: " + r.status);
    DATA = await r.json();
    return DATA;
  }

  function filteredRuns() {
    return DATA.runs.filter(r => {
      if (FILTER_STATE.tgt_dsl && r.tgt_dsl !== FILTER_STATE.tgt_dsl) return false;
      if (FILTER_STATE.src_dsl && r.src_dsl !== FILTER_STATE.src_dsl) return false;
      if (FILTER_STATE.persona && r.persona !== FILTER_STATE.persona) return false;
      if (FILTER_STATE.scenario && r.scenario !== FILTER_STATE.scenario) return false;
      if (FILTER_STATE.model && r.model !== FILTER_STATE.model) return false;
      return true;
    });
  }

  function uniqueValues(key) {
    const seen = new Set();
    for (const r of DATA.runs) {
      const v = r[key];
      if (v !== null && v !== undefined && v !== "") seen.add(String(v));
    }
    return Array.from(seen).sort((a, b) => a.localeCompare(b));
  }

  function buildFilters(container, keys) {
    for (const sel of container.querySelectorAll("select[data-filter]")) {
      const key = sel.dataset.filter;
      if (keys && !keys.includes(key)) continue;
      const current = sel.value;
      sel.innerHTML = "";
      const optAll = document.createElement("option");
      optAll.value = "";
      optAll.textContent = "all";
      sel.appendChild(optAll);
      for (const v of uniqueValues(key)) {
        const o = document.createElement("option");
        o.value = v; o.textContent = v;
        sel.appendChild(o);
      }
      sel.value = FILTER_STATE[key] || current || "";
      sel.onchange = (ev) => {
        FILTER_STATE[key] = ev.target.value;
        if (sel.closest("body").contains(document.getElementById("by-model"))) {
          renderLeaderboard();
        } else {
          renderRunsIndex();
        }
      };
    }
  }

  function aggregate(runs, groupKeys) {
    const groups = new Map();
    for (const r of runs) {
      const groupId = groupKeys.map(k => String(r[k] ?? "")).join("|");
      let g = groups.get(groupId);
      if (!g) {
        g = { runs: 0, passed: 0, final_scores: [], sol_scores: [], speedups: [], cost: 0 };
        for (const k of groupKeys) g[k] = r[k];
        groups.set(groupId, g);
      }
      g.runs += 1;
      if (r.outcome === "passed") g.passed += 1;
      if (typeof r.final_score === "number") g.final_scores.push(r.final_score);
      if (typeof r.sol_score === "number" && r.sol_score >= 0) {
        g.sol_scores.push(r.sol_score);
      }
      if (typeof r.speedup_vs_ref === "number" && r.speedup_vs_ref >= 0) {
        g.speedups.push(r.speedup_vs_ref);
      }
      if (typeof r.cost_gpu_seconds === "number") g.cost += r.cost_gpu_seconds;
    }
    const out = [];
    for (const g of groups.values()) {
      g.success_rate = g.runs ? g.passed / g.runs : 0;
      g.mean_final = g.final_scores.length
        ? g.final_scores.reduce((a, b) => a + b, 0) / g.final_scores.length
        : null;
      g.mean_sol = g.sol_scores.length
        ? g.sol_scores.reduce((a, b) => a + b, 0) / g.sol_scores.length
        : null;
      // Geometric mean for speedup so a single 100x outlier does not
      // dominate the arithmetic mean across a basket of problems.
      if (g.speedups.length) {
        const logs = g.speedups.map(s => Math.log(Math.max(s, 1e-9)));
        g.mean_speedup = Math.exp(logs.reduce((a, b) => a + b, 0) / logs.length);
      } else {
        g.mean_speedup = null;
      }
      g.cost_gpu_seconds = g.cost;
      out.push(g);
    }
    return out;
  }

  function sortRows(rows, tableId, defaultKey, defaultDir = "desc") {
    let st = SORT_STATE.get(tableId);
    if (!st) {
      st = { key: defaultKey, direction: defaultDir };
      SORT_STATE.set(tableId, st);
    }
    rows.sort((a, b) => {
      const av = a[st.key], bv = b[st.key];
      if (av === null || av === undefined) return 1;
      if (bv === null || bv === undefined) return -1;
      if (av === bv) return 0;
      const cmp = (typeof av === "number" && typeof bv === "number")
        ? av - bv
        : String(av).localeCompare(String(bv));
      return st.direction === "asc" ? cmp : -cmp;
    });
    return st;
  }

  function bindHeaderSort(table, defaultKey, defaultDir, rerender) {
    const ths = table.querySelectorAll("thead th[data-sortable='true']");
    let st = SORT_STATE.get(table.id) || { key: defaultKey, direction: defaultDir };
    SORT_STATE.set(table.id, st);
    for (const th of ths) {
      th.onclick = () => {
        const key = th.dataset.key;
        const cur = SORT_STATE.get(table.id);
        if (cur.key === key) {
          cur.direction = cur.direction === "asc" ? "desc" : "asc";
        } else {
          cur.key = key;
          cur.direction = "desc";
        }
        rerender();
      };
      const cur = SORT_STATE.get(table.id);
      th.classList.remove("sort-asc", "sort-desc");
      if (th.dataset.key === cur.key) {
        th.classList.add(cur.direction === "asc" ? "sort-asc" : "sort-desc");
      }
    }
  }

  function renderLeaderboard() {
    loadData().then(() => {
      const filters = document.getElementById("filters");
      if (filters) buildFilters(filters, ["tgt_dsl", "src_dsl", "persona", "scenario"]);

      const visibleRuns = filteredRuns();

      // By-model table.
      const byModel = aggregate(visibleRuns, ["model"]);
      const t1 = document.getElementById("by-model");
      bindHeaderSort(t1, "mean_final", "desc", renderLeaderboard);
      sortRows(byModel, "by-model", "mean_final", "desc");
      const peak = Math.max(0, ...byModel.map(g => g.mean_final || 0));
      const tbody1 = t1.querySelector("tbody");
      tbody1.innerHTML = "";
      if (!byModel.length) {
        tbody1.innerHTML = '<tr><td colspan="7" class="empty">No runs match the current filter.</td></tr>';
      }
      for (const g of byModel) {
        const tr = document.createElement("tr");
        if (g.mean_final && g.mean_final === peak) tr.classList.add("peak");
        const speedupCell = (g.mean_speedup != null)
          ? `${fmtNumber(g.mean_speedup, 2)}x`
          : "";
        tr.innerHTML = `
          <td>${fmtStr(g.model)}</td>
          <td class="numeric">${fmtInt(g.runs)}</td>
          <td class="numeric">${fmtPct(g.success_rate, 0)}</td>
          <td class="numeric">${fmtNumber(g.mean_final, 3)}</td>
          <td class="numeric">${fmtNumber(g.mean_sol, 3)}</td>
          <td class="numeric">${speedupCell}</td>
          <td class="numeric">${fmtNumber(g.cost_gpu_seconds, 1)}</td>
        `;
        tbody1.appendChild(tr);
      }

      // By-model-and-translation-axis table.
      const byModelAxis = aggregate(visibleRuns, ["model", "src_dsl", "tgt_dsl"]);
      const t2 = document.getElementById("by-model-axis");
      bindHeaderSort(t2, "mean_final", "desc", renderLeaderboard);
      sortRows(byModelAxis, "by-model-axis", "mean_final", "desc");
      const tbody2 = t2.querySelector("tbody");
      tbody2.innerHTML = "";
      if (!byModelAxis.length) {
        tbody2.innerHTML = '<tr><td colspan="8" class="empty">No runs match the current filter.</td></tr>';
      }
      for (const g of byModelAxis) {
        const tr = document.createElement("tr");
        const axis = `${fmtStr(g.src_dsl)} → ${fmtStr(g.tgt_dsl)}`;
        const speedupCell = (g.mean_speedup != null)
          ? `${fmtNumber(g.mean_speedup, 2)}x`
          : "";
        tr.innerHTML = `
          <td>${fmtStr(g.model)}</td>
          <td>${axis}</td>
          <td class="numeric">${fmtInt(g.runs)}</td>
          <td class="numeric">${fmtPct(g.success_rate, 0)}</td>
          <td class="numeric">${fmtNumber(g.mean_final, 3)}</td>
          <td class="numeric">${fmtNumber(g.mean_sol, 3)}</td>
          <td class="numeric">${speedupCell}</td>
          <td class="numeric">${fmtNumber(g.cost_gpu_seconds, 1)}</td>
        `;
        tbody2.appendChild(tr);
      }

      const gen = document.getElementById("generated-at");
      if (gen && DATA.generated_at) gen.textContent = "data: " + DATA.generated_at;
    }).catch(err => {
      const tbody = document.querySelector("#by-model tbody");
      if (tbody) tbody.innerHTML = `<tr><td colspan="7" class="empty">${err.message}</td></tr>`;
    });
  }

  function renderRunsIndex() {
    loadData().then(() => {
      const filters = document.getElementById("filters");
      if (filters) buildFilters(filters, ["model", "tgt_dsl", "src_dsl", "persona", "scenario"]);

      const visibleRuns = filteredRuns().slice();
      const t = document.getElementById("runs");
      bindHeaderSort(t, "timestamp", "desc", renderRunsIndex);
      sortRows(visibleRuns, "runs", "timestamp", "desc");
      const tbody = t.querySelector("tbody");
      tbody.innerHTML = "";
      if (!visibleRuns.length) {
        tbody.innerHTML = '<tr><td colspan="12" class="empty">No runs match the current filter.</td></tr>';
      }
      for (const r of visibleRuns) {
        const tr = document.createElement("tr");
        const outcomeClass = "outcome-" + (r.outcome || "incomplete");
        const axis = `${fmtStr(r.src_dsl)} → ${fmtStr(r.tgt_dsl)}`;
        const speedupCell = (typeof r.speedup_vs_ref === "number" && r.speedup_vs_ref >= 0)
          ? `${fmtNumber(r.speedup_vs_ref, 2)}x`
          : "";
        tr.innerHTML = `
          <td>${fmtStr(r.timestamp)}</td>
          <td>${fmtStr(r.scenario)}</td>
          <td>${fmtStr(r.model)}</td>
          <td>${fmtStr(r.persona)}</td>
          <td>${fmtStr(r.problem_id)}</td>
          <td>${axis}</td>
          <td class="${outcomeClass}">${fmtStr(r.outcome)}</td>
          <td class="numeric">${fmtNumber(r.final_score, 3)}</td>
          <td class="numeric">${fmtNumber(r.sol_score, 3)}</td>
          <td class="numeric">${speedupCell}</td>
          <td class="numeric">${fmtNumber(r.cost_gpu_seconds, 1)}</td>
          <td><a href="${r.viewer_path}">trace</a></td>
        `;
        tbody.appendChild(tr);
      }

      const gen = document.getElementById("generated-at");
      if (gen && DATA.generated_at) gen.textContent = "data: " + DATA.generated_at;
    }).catch(err => {
      const tbody = document.querySelector("#runs tbody");
      if (tbody) tbody.innerHTML = `<tr><td colspan="12" class="empty">${err.message}</td></tr>`;
    });
  }

  return { renderLeaderboard, renderRunsIndex };
})();
