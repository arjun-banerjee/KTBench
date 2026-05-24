// Trace viewer. Loads ./trace.jsonl, parses one event per line, and
// renders two views: a three-panel timeline (actors, tools, state
// changes) and a linear chat feed. The slider scrubs through events
// for both views. No framework.
//
// While a run is in flight (the live trace writer is appending lines
// to disk) the viewer polls for new events every two seconds and
// stays pinned to the latest event so a reader sees the run grow.
// When the slider is parked away from the tail, polling continues
// but the cursor stays put.

(function () {
  const els = {
    slider:    document.getElementById('slider'),
    playBtn:   document.getElementById('playBtn'),
    prevBtn:   document.getElementById('prevBtn'),
    nextBtn:   document.getElementById('nextBtn'),
    tickLabel: document.getElementById('tickLabel'),
    messages:  document.getElementById('messagesPanel'),
    tools:     document.getElementById('toolsPanel'),
    diff:      document.getElementById('diffPanel'),
    timeline:  document.getElementById('timelineStage'),
    chat:      document.getElementById('chatStage'),
    chatFeed:  document.getElementById('chatFeed'),
    tabTimeline: document.getElementById('tabTimeline'),
    tabChat:     document.getElementById('tabChat'),
    traceLabel:  document.getElementById('traceLabel'),
  };

  let events = [];
  let actors = [];
  let actorKinds = {};
  let position = 0;
  let playTimer = null;
  let pollTimer = null;
  let stuckToTail = true;
  let view = 'timeline';

  async function load() {
    let text = '';
    try {
      const resp = await fetch('trace.jsonl', { cache: 'no-store' });
      if (!resp.ok) throw new Error('http ' + resp.status);
      text = await resp.text();
    } catch (err) {
      showError('could not load trace.jsonl: ' + err.message);
      return;
    }
    parseAndStore(text);
    if (els.traceLabel) {
      els.traceLabel.textContent = `${events.length} events  ·  ${actors.length} actors`;
    }
    setPosition(events.length > 0 ? events.length - 1 : 0);
    startPolling();
  }

  function parseAndStore(text) {
    events = text
      .split('\n')
      .map((s) => s.trim())
      .filter(Boolean)
      .map((s) => { try { return JSON.parse(s); } catch { return null; } })
      .filter(Boolean);
    actors = collectActors(events);
    actorKinds = inferActorKinds(events);
    els.slider.max = String(Math.max(events.length - 1, 0));
  }

  function startPolling() {
    if (pollTimer) return;
    pollTimer = setInterval(async () => {
      try {
        const resp = await fetch('trace.jsonl', { cache: 'no-store' });
        if (!resp.ok) return;
        const text = await resp.text();
        const before = events.length;
        parseAndStore(text);
        if (events.length === before) return;
        if (els.traceLabel) {
          els.traceLabel.textContent = `${events.length} events  ·  ${actors.length} actors`;
        }
        // If the user was already at the tail, follow new events.
        // Otherwise leave their cursor alone but keep the slider's
        // max in sync so they can scrub forward.
        if (stuckToTail) {
          setPosition(events.length - 1);
        } else {
          els.slider.max = String(Math.max(events.length - 1, 0));
          els.tickLabel.textContent = `${position + 1} / ${events.length}`;
        }
      } catch (_) {
        // Ignore transient fetch errors so polling survives a
        // momentary disk-write race.
      }
    }, 2000);
  }

  function collectActors(events) {
    const seen = new Set();
    const order = [];
    for (const e of events) {
      if (e.actor && !seen.has(e.actor)) {
        seen.add(e.actor);
        order.push(e.actor);
      }
    }
    return order;
  }

  // Guess whether each actor is a user or an agent by looking at what
  // kinds of payloads they produce. Agents emit tool_call or
  // agent_message; users emit user_message.
  function inferActorKinds(events) {
    const kinds = {};
    for (const e of events) {
      if (!e.actor) continue;
      const k = e.payload && e.payload.kind;
      if (k === 'agent_message' || k === 'tool_call' || k === 'tool_result') {
        kinds[e.actor] = 'agent';
      } else if (k === 'user_message' && !kinds[e.actor]) {
        kinds[e.actor] = 'user';
      }
    }
    return kinds;
  }

  function setPosition(p) {
    position = clamp(p, 0, Math.max(events.length - 1, 0));
    els.slider.value = String(position);
    stuckToTail = position === Math.max(events.length - 1, 0);
    els.tickLabel.textContent = `${position + 1} / ${events.length}`;
    render();
  }

  function clamp(v, lo, hi) {
    return v < lo ? lo : v > hi ? hi : v;
  }

  function render() {
    const upTo = events.slice(0, position + 1);
    if (view === 'timeline') {
      renderMessages(upTo);
      renderTools(upTo);
      renderDiff(upTo);
    } else {
      renderChat(upTo);
    }
  }

  function renderMessages(slice) {
    const cols = {};
    for (const a of actors) cols[a] = [];
    for (const e of slice) {
      const k = e.payload && e.payload.kind;
      if (k !== 'user_message' && k !== 'agent_message') continue;
      const who = e.actor || '?';
      (cols[who] = cols[who] || []).push({ kind: k, text: e.payload.text || '', tick: e.tick });
    }
    const html = ['<h4>actors</h4>'];
    for (const a of actors) {
      const msgs = cols[a] || [];
      html.push(`<div class="actor-col">`);
      html.push(`<div class="actor-msg" style="border-color:#bbb"><div class="who">${escape(a)}</div></div>`);
      for (const m of msgs.slice(-5)) {
        html.push(`<div class="actor-msg"><div class="who">${escape(a)} | tick ${m.tick}</div><div class="text">${escape(m.text)}</div></div>`);
      }
      html.push(`</div>`);
    }
    els.messages.innerHTML = html.join('');
  }

  function renderTools(slice) {
    const html = ['<h4>tool calls</h4>'];
    for (const e of slice) {
      const k = e.payload && e.payload.kind;
      const seed = e.payload && e.payload.seed === true;
      const seedTag = seed ? ' <span class="seed-tag">[seed]</span>' : '';
      const lineClass = seed ? 'tool-line seeded' : 'tool-line';
      if (k === 'tool_call') {
        html.push(`<div class="${lineClass}"><div><span class="name">${escape(e.payload.name)}</span>${seedTag} <span class="who">[call] ${escape(e.actor || 'system')} | tick ${e.tick}</span></div><div class="args">${renderArgs(e.payload.args)}</div></div>`);
      } else if (k === 'tool_result') {
        html.push(`<div class="${lineClass}"><div><span class="name">${escape(e.payload.name)}</span>${seedTag} <span class="who">[result]</span></div><div class="res">${renderResult(e.payload.result)}</div></div>`);
      }
    }
    if (html.length === 1) html.push(`<div class="who">no tool activity yet</div>`);
    els.tools.innerHTML = html.join('');
  }

  // Args may include long strings (kernel source, transcripts).
  // Render scalar / short values inline; promote long strings to a
  // collapsible code block so the panel stays readable.
  function renderArgs(args) {
    if (args == null || typeof args !== 'object') return escape(stringify(args));
    const parts = [];
    for (const [k, v] of Object.entries(args)) {
      if (typeof v === 'string' && v.length > 80) {
        parts.push(`<details class="arg-long"><summary><code>${escape(k)}</code> <span class="who">(${v.length} chars)</span></summary><pre><code>${escape(v)}</code></pre></details>`);
      } else {
        parts.push(`<div><code>${escape(k)}</code> = ${escape(stringify(v))}</div>`);
      }
    }
    return parts.join('');
  }

  function renderResult(result) {
    if (result && typeof result === 'object') {
      const summary = result.summary || (result.effect && result.effect.summary);
      if (typeof summary === 'string') {
        return `<div class="res-summary">${escapeMultiline(summary)}</div>`;
      }
    }
    const s = compactJson(result, 300);
    return escape(s);
  }

  function renderDiff(slice) {
    const rows = [];
    for (const e of slice) {
      const k = e.payload && e.payload.kind;
      if (k === 'state_diff' && e.payload.diff) {
        const d = e.payload.diff;
        rows.push({ who: e.actor || '?', table: d.table || (d.field || ''), field: d.field || '', old: stringify(d.old), new: stringify(d.new) });
      } else if (k === 'tool_result' && e.payload.result) {
        const r = e.payload.result;
        if (r && r.data && typeof r.data === 'object') {
          for (const [k2, v2] of Object.entries(r.data)) {
            rows.push({ who: e.actor || '?', table: e.payload.name, field: k2, old: '', new: stringify(v2) });
          }
        }
      }
    }
    const html = ['<h4>state changes</h4>'];
    if (rows.length === 0) {
      html.push(`<div class="who">no state changes yet</div>`);
    } else {
      html.push('<table class="diff">');
      html.push('<thead><tr><th>actor</th><th>table</th><th>field</th><th>old</th><th>new</th></tr></thead><tbody>');
      for (const r of rows.slice(-12)) {
        html.push(`<tr><td>${escape(r.who)}</td><td>${escape(r.table)}</td><td>${escape(r.field)}</td><td>${escape(r.old)}</td><td>${escape(r.new)}</td></tr>`);
      }
      html.push('</tbody></table>');
    }
    els.diff.innerHTML = html.join('');
  }

  // Linear chat feed: user messages right-aligned, agent messages
  // left-aligned, tool calls and tool results rendered inline as small
  // italicized rows so the flow stays readable.
  function renderChat(slice) {
    const html = [];
    for (const e of slice) {
      const p = e.payload || {};
      const k = p.kind;
      const who = e.actor || '';
      const kind = actorKinds[who] || (k === 'user_message' ? 'user' : 'agent');
      if (k === 'user_message') {
        html.push(bubble(who, p.text || '', 'user', e.tick));
      } else if (k === 'agent_message') {
        html.push(bubble(who, p.text || '', 'agent', e.tick));
      } else if (k === 'tool_call') {
        const argsHtml = renderArgs(p.args);
        html.push(`<div class="chat-tool ${kind === 'user' ? 'right' : 'left'}"><span class="chat-tool-tag">[tool]</span> <strong>${escape(who)}</strong> called <code>${escape(p.name)}</code><div class="chat-tool-args">${argsHtml}</div></div>`);
      } else if (k === 'tool_result') {
        const res = renderResult(p.result);
        html.push(`<div class="chat-tool ${kind === 'user' ? 'right' : 'left'}"><span class="chat-tool-tag">[result]</span> <code>${escape(p.name)}</code> <div class="chat-tool-res">${res}</div></div>`);
      } else if (k === 'cost') {
        const unit = escape(String(p.unit || ''));
        const amount = stringify(p.amount);
        const total = stringify(p.running_total);
        html.push(`<div class="chat-note">[cost] ${escape(who)} +${escape(amount)} ${unit} (total ${escape(total)})</div>`);
      } else if (k === 'progress') {
        const pct = Math.round(Number(p.fraction || 0) * 100);
        html.push(`<div class="chat-note">[progress] ${escape(p.tool || '')} ${pct}% ${escape(p.message || '')}</div>`);
      } else if (k === 'state_diff' && p.diff) {
        const d = p.diff;
        const summary = `${d.table || d.field || 'state'}.${d.field || ''}: ${stringify(d.old)} -> ${stringify(d.new)}`;
        html.push(`<div class="chat-note">[state] ${escape(summary)}</div>`);
      } else if (k === 'system') {
        const note = p.note || '';
        if (note.startsWith('grader: ')) {
          try {
            const payload = JSON.parse(note.slice('grader: '.length));
            const rows = Object.entries(payload.scores || {})
              .map(([k, v]) => `<tr><td><code>${escape(k)}</code></td><td>${escape(stringify(v))}</td></tr>`)
              .join('');
            html.push(
              `<div class="chat-grader"><div class="chat-grader-head">grader (${escape(payload.scenario || '')})</div>` +
              `<table class="chat-grader-table">${rows}</table></div>`
            );
            continue;
          } catch (_) { /* fall through to plain note */ }
        }
        html.push(`<div class="chat-note">${escape(note)}</div>`);
      }
    }
    if (html.length === 0) {
      html.push('<div class="chat-note">no events yet</div>');
    }
    els.chatFeed.innerHTML = html.join('');
    els.chatFeed.scrollTop = els.chatFeed.scrollHeight;
  }

  function bubble(who, text, side, tick) {
    const sideClass = side === 'user' ? 'right' : 'left';
    return `<div class="chat-row ${sideClass}">
      <div class="chat-bubble ${sideClass}">
        <div class="chat-meta">${escape(who)} | tick ${tick}</div>
        <div class="chat-text">${renderMarkdown(text)}</div>
      </div>
    </div>`;
  }

  // Lightweight markdown for chat bubbles. We support fenced ``` code
  // blocks, single-backtick inline code, and newline preservation. We
  // deliberately avoid a full markdown parser so the viewer stays
  // dependency-free and so a malformed message does not break the
  // page layout.
  function renderMarkdown(text) {
    if (text == null) return '';
    const parts = [];
    const re = /```([a-zA-Z0-9_+\-]*)\n([\s\S]*?)```/g;
    let last = 0;
    let m;
    while ((m = re.exec(text)) !== null) {
      const before = text.slice(last, m.index);
      parts.push(renderInline(before));
      const lang = m[1] ? ` data-lang="${escape(m[1])}"` : '';
      parts.push(`<pre${lang}><code>${escape(m[2])}</code></pre>`);
      last = re.lastIndex;
    }
    parts.push(renderInline(text.slice(last)));
    return parts.join('');
  }

  function renderInline(text) {
    if (!text) return '';
    // escape first, then promote inline `code` spans. Newlines become
    // <br>; everything else is left alone.
    const escaped = escape(text);
    const withCode = escaped.replace(
      /`([^`]+)`/g,
      (_m, body) => `<code>${body}</code>`,
    );
    return withCode.replaceAll('\n', '<br>');
  }

  function compactJson(v, max = 120) {
    if (v == null) return '';
    let s;
    try { s = JSON.stringify(v); } catch { s = String(v); }
    if (s && s.length > max) s = s.slice(0, max) + '...';
    return s || '';
  }

  function stringify(v) {
    if (v == null) return '';
    if (typeof v === 'string') return v;
    try { return JSON.stringify(v); } catch (e) { return String(v); }
  }

  function escape(s) {
    return String(s)
      .replaceAll('&', '&amp;')
      .replaceAll('<', '&lt;')
      .replaceAll('>', '&gt;');
  }

  function escapeMultiline(s) {
    return escape(s).replaceAll('\n', '<br>');
  }

  els.slider.addEventListener('input', () => setPosition(Number(els.slider.value)));
  els.prevBtn.addEventListener('click', () => setPosition(position - 1));
  els.nextBtn.addEventListener('click', () => setPosition(position + 1));
  els.playBtn.addEventListener('click', () => {
    if (playTimer) {
      clearInterval(playTimer);
      playTimer = null;
      els.playBtn.textContent = 'play';
      return;
    }
    els.playBtn.textContent = 'pause';
    if (position >= events.length - 1) setPosition(0);
    playTimer = setInterval(() => {
      if (position >= events.length - 1) {
        clearInterval(playTimer);
        playTimer = null;
        els.playBtn.textContent = 'play';
        return;
      }
      setPosition(position + 1);
    }, 1000);
  });

  if (els.tabTimeline && els.tabChat) {
    els.tabTimeline.addEventListener('click', () => switchView('timeline'));
    els.tabChat.addEventListener('click', () => switchView('chat'));
  }

  function switchView(next) {
    view = next;
    if (els.tabTimeline) els.tabTimeline.classList.toggle('active', next === 'timeline');
    if (els.tabChat) els.tabChat.classList.toggle('active', next === 'chat');
    if (els.timeline) els.timeline.style.display = next === 'timeline' ? '' : 'none';
    if (els.chat) els.chat.style.display = next === 'chat' ? '' : 'none';
    render();
  }

  function showError(msg) {
    if (els.messages) els.messages.innerHTML = '<h4>actors</h4><div class="who">' + escape(msg) + '</div>';
    if (els.tools) els.tools.innerHTML = '<h4>tool calls</h4>';
    if (els.diff) els.diff.innerHTML = '<h4>state changes</h4>';
    if (els.chatFeed) els.chatFeed.innerHTML = `<div class="chat-note">${escape(msg)}</div>`;
  }

  load();
})();
