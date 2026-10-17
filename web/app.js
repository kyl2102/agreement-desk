// Agreement Desk front end: plain JavaScript, no build step. Built from the Claude Design canvas.
// Every number on screen comes from api.py, which calls the same pipeline as the Streamlit app.
"use strict";

const $view = document.getElementById("view");
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[c]));
const api = async (path, body) => {
  const r = await fetch("/api/" + path, body ? {method: "POST", headers: {"Content-Type": "application/json"},
                                               body: JSON.stringify(body)} : {});
  if (!r.ok) throw new Error((await r.text()) || r.statusText);
  return r.json();
};
// Bold the words that ARE the answer. Text is escaped piece by piece, so bolding can never inject HTML.
// Filler words never get bolded: they'd light up half of every passage ("each other", "keeping", "much").
const STOP = new Set(("the a an of to for and or in on at by is are be can do does we they their our us it if what which who how " +
  "when where with from this that any there each other others otherwise them have keep keeping much many make some someone else " +
  "would could should will shall must about into than then also only just like want need deal thing things stop happens happen get").split(" "));
function questionTerms(q) {
  // search results: bold the question's own words, matching word stems ("terminate" also bolds "termination")
  return [...new Set((q || "").toLowerCase().match(/[a-z]{4,}/g) || [])].filter((w) => !STOP.has(w))
    .map((w) => w.replace(/(ing|ed|es|s)$/, "")).filter((w) => w.length >= 4).map((w) => ({stem: w}));
}
function highlight(text, terms) {
  text = String(text ?? "");
  const parts = (terms || []).filter(Boolean).map((t) => typeof t === "string"
      ? t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")
      : "\\b" + t.stem.replace(/[.*+?^${}()|[\]\\]/g, "\\$&") + "\\w*")
    .sort((a, b) => b.length - a.length);
  if (!parts.length) return esc(text);
  const re = new RegExp("(" + parts.join("|") + ")", "gi");
  return text.split(re).map((seg, k) => k % 2 ? `<b class="key">${esc(seg)}</b>` : esc(seg)).join("");
}
const store = {
  get(k) { try { return JSON.parse(sessionStorage.getItem(k)); } catch (e) { return null; } },
  set(k, v) { try { sessionStorage.setItem(k, JSON.stringify(v)); } catch (e) { /* private mode: memory only */ } },
};
const EXAMPLES = ["Which agreements have no liability cap?", "How many agreements are governed by New York law?",
  "whats the termination notice period for ingram", "Which agreements are with Bay Networks?",
  "Notice periods longer than 60 days", "Agreements executed between 2000 and 2004",
  "who pays the lawyers if we fight", "does the contract renew by itself"];
const ICON_PREV = '<svg width="14" height="14" viewBox="0 0 14 14" aria-hidden="true"><path d="M9 2L4 7l5 5" stroke="#1C2333" stroke-width="1.6" fill="none"/></svg>';
const ICON_NEXT = '<svg width="14" height="14" viewBox="0 0 14 14" aria-hidden="true"><path d="M5 2l5 5-5 5" stroke="#1C2333" stroke-width="1.6" fill="none"/></svg>';

// ---------- source pane: one PDF page with the cited text boxed in red ----------
const source = {doc: null, page: 1, pages: 1, quote: "", citedPage: 1, title: "", label: "Source", highlight: [], full: false};
function setSource(s) { Object.assign(source, {highlight: [], full: false}, s, {citedPage: s.page}); renderSource(); }
function renderSource() {
  const el = document.getElementById("source");
  if (!el) return;
  if (!source.doc) {
    el.innerHTML = `<div class="muted">Source</div><div class="empty">Select a result to see the page it came from, with the cited text boxed in red.</div>`;
    return;
  }
  const cited = source.page === source.citedPage;
  const q = cited ? source.quote.slice(0, 200) : "";
  const crop = cited && q && !source.full;   // zoom to the clause on the cited page; whole page otherwise
  el.innerHTML = `
    <div class="src-head">
      <div><div class="muted">${esc(source.label)}</div><h2 style="font-size:19px">${esc(source.title)}</h2></div>
      <div class="pager">
        <button class="iconbtn" aria-label="Previous page" data-pg="-1" ${source.page <= 1 ? "disabled" : ""}>${ICON_PREV}</button>
        <span class="muted" style="padding:0 8px">Page ${source.page} of ${source.pages}</span>
        <button class="iconbtn" aria-label="Next page" data-pg="1" ${source.page >= source.pages ? "disabled" : ""}>${ICON_NEXT}</button>
      </div>
    </div>
    ${source.quote ? `<blockquote class="quote">"${highlight(source.quote.slice(0, 500), source.highlight)}${source.quote.length > 500 ? "…" : ""}"</blockquote>` : ""}
    <div class="sheet"><img alt="Page ${source.page} of ${esc(source.title)}${crop ? ", zoomed to the cited clause" : ""}"
      src="/api/page/${esc(source.doc)}/${source.page}?q=${encodeURIComponent(q)}&crop=${crop ? 1 : 0}"></div>
    <div class="row">
      ${cited && q ? `<button class="btn" type="button" id="zoom">${source.full ? "Zoom to the clause" : "Show full page"}</button>` : ""}
      <a class="btn link" href="/api/pdf/${esc(source.doc)}" target="_blank" rel="noopener">Open full document</a>
    </div>`;
  el.querySelectorAll("[data-pg]").forEach((b) => b.onclick = () => { source.page += Number(b.dataset.pg); renderSource(); });
  const zoom = el.querySelector("#zoom");
  if (zoom) zoom.onclick = () => { source.full = !source.full; renderSource(); };
}

// ---------- Ask ----------
const askState = Object.assign({mode: "auto", level: "agreements", dropScope: false}, (store.get("ask-last") || {}).key || {});
delete askState.q;
function viewAsk(params) {
  const last = store.get("ask-last");
  if (!params.get("q") && last && last.q) { go("ask", {q: last.q}); return; }   // came back to Ask: reopen the last question
  const q = params.get("q") || "";
  $view.innerHTML = `
    <form class="card" id="askform">
      <label class="lead" for="q">Ask about the agreements</label>
      <div class="row">
        <input id="q" class="q" type="text" value="${esc(q)}" placeholder="e.g. Which agreements can be terminated on 60 days notice?">
        <button class="btn primary" type="submit">Ask</button>
      </div>
      <div class="row">
        <span class="muted">Answer as</span>
        <div class="seg" role="group" aria-label="Answer as">
          ${[["auto", "Let the desk decide"], ["all", "Complete list"], ["examples", "Examples"]].map(([k, l]) =>
            `<button type="button" data-mode="${k}" aria-pressed="${askState.mode === k}">${l}</button>`).join("")}
        </div>
        <label class="sr" for="level">Count by</label>
        <select id="level"><option value="agreements">Count agreements</option><option value="documents">Count documents</option></select>
        <label class="sr" for="ex">Try an example</label>
        <select id="ex"><option value="">Try an example…</option>${EXAMPLES.map((e) => `<option>${esc(e)}</option>`).join("")}</select>
      </div>
    </form>
    <div class="split">
      <section class="primary" id="answer" aria-label="Answer"></section>
      <aside class="aside card" id="source" aria-label="Source document"></aside>
    </div>`;
  const form = document.getElementById("askform");
  document.getElementById("level").value = askState.level;
  form.querySelectorAll("[data-mode]").forEach((b) => b.onclick = () => {
    askState.mode = b.dataset.mode;
    form.querySelectorAll("[data-mode]").forEach((x) => x.setAttribute("aria-pressed", x === b));
    if (q) runAsk(q);
  });
  document.getElementById("level").onchange = (e) => { askState.level = e.target.value; if (q) runAsk(q); };
  document.getElementById("ex").onchange = (e) => { if (e.target.value) { askState.mode = "auto"; go("ask", {q: e.target.value}); } };
  form.onsubmit = (e) => { e.preventDefault(); const v = document.getElementById("q").value.trim(); if (v) { askState.dropScope = false; go("ask", {q: v}); } };
  source.doc = null; renderSource();
  if (q) runAsk(q); else document.getElementById("answer").innerHTML =
    `<div class="card"><p class="muted">Questions about every agreement ("which", "how many", "list") get a complete answer from the extracted data. Questions about wording get the best-matching clauses. Every answer opens to the page it came from.</p></div>`;
}

async function runAsk(q) {
  const box = document.getElementById("answer");
  const key = {q, mode: askState.mode, level: askState.level, dropScope: askState.dropScope};
  const last = store.get("ask-last");
  let r = last && JSON.stringify(last.key) === JSON.stringify(key) ? last.response : null;   // same question: no re-run
  if (!r) {
    box.innerHTML = `<div class="card loading">Working out the query… (the local model plans it; this can take a few seconds)</div>`;
    try { r = await api("ask", {question: q, mode: askState.mode, level: askState.level, drop_scope: askState.dropScope}); }
    catch (e) { box.innerHTML = `<div class="card"><div class="note error">Something went wrong: ${esc(e.message)}</div></div>`; return; }
    store.set("ask-last", {q, key, response: r});
  }
  const parts = [];
  if (r.coverage) parts.push(`<div class="note warn">${esc(r.coverage)}</div>`);
  r.answers.forEach((a, i) => parts.push(a.badge === "complete" ? renderComplete(a, i) : renderExamples(a, i, r.question)));
  parts.push(renderHow(r.how));
  box.innerHTML = `<div class="card">${parts.join("")}</div>`;
  wireRows(box, r);
  const drop = box.querySelector("#dropscope");
  if (drop) drop.onclick = () => { askState.dropScope = true; runAsk(q); };
  // reopen the citation picked last time (or the first one), so the source is never more than zero clicks away
  const saved = (store.get("ask-last") || {}).cite;
  const pick = (saved && box.querySelector(`[data-cite="${saved}"]`)) || box.querySelector("[data-cite]");
  if (pick) pick.click();
}

function renderComplete(a, i) {
  if (a.error) return `<h1 class="headline">${esc(a.headline)}</h1>` +
    (a.notes || []).map((n) => `<div class="note ${esc(n.kind)}">${esc(n.text)}</div>`).join("");
  const notes = (a.notes || []).map((n) => `<div class="note ${esc(n.kind)}">${esc(n.text)}</div>`).join("");
  const chips = (a.filters || []).map((f) => `<span class="chip">${esc(f)}</span>`).join("") +
    (a.scoped ? `<button class="btn ghost" id="dropscope" type="button">Remove the date filter</button>` : "");
  let body = "";
  if (a.groups) {
    body = `<div class="tablebox"><table><thead><tr><th>${esc(a.columns[0])}</th><th>Agreements</th><th>Which</th></tr></thead><tbody>
      ${a.groups.map((g) => `<tr><td class="cell-title">${esc(g.value)}</td><td>${g.count}</td>
        <td>${g.agreements.map((x) => `<a href="#/agreements/${esc(x.id)}">${esc(x.name)}</a>`).join(", ")}</td></tr>`).join("")}
      </tbody></table></div>`;
  } else if (a.rows && a.rows.length) {
    body = `<div class="tablebox"><table><thead><tr>${a.columns.map((c) => `<th scope="col">${esc(c)}</th>`).join("")}</tr></thead><tbody>
      ${a.rows.map((r, k) => `<tr class="pick" data-cite="${i}:${k}" tabindex="0">
          <td><div class="cell-title">${esc(r.agreement)}</div><div class="cell-sub"><a href="#/agreements/${esc(r.id)}">Open agreement</a></div></td>
          ${r.cells.map((c) => `<td>${esc(c)}</td>`).join("")}</tr>`).join("")}
      </tbody></table></div>`;
  }
  return `
    <div class="row"><span class="badge complete">Complete list</span><span class="muted">Every agreement in the set was checked, not a sample</span></div>
    <h1 class="headline">${esc(a.headline || "")}</h1>
    ${a.subhead ? `<p class="muted" style="margin:0;font-size:15px">${esc(a.subhead)}</p>` : ""}
    <div class="chips"><span class="muted">Filters</span>${chips}</div>
    ${notes}${body}
    ${a.accuracy ? `<div class="note info">${esc(a.accuracy)}</div>` : ""}`;
}

function renderExamples(a, i, question) {
  a.terms = questionTerms(question);
  return `
    <div class="row"><span class="badge examples">Examples</span><span class="muted">Best-matching clauses, not a complete list</span></div>
    ${a.results.map((x, k) => `<div class="example">
        <div class="cell-title">${esc(x.agreement)}</div>
        <div class="muted">${esc(x.section)}, ${esc(x.pages)}</div>
        <blockquote class="quote">${highlight(x.text.slice(0, 320), a.terms)}${x.text.length > 320 ? "…" : ""}</blockquote>
        <div><button class="btn link" type="button" data-cite="x${i}:${k}">Show page ${x.page}</button></div>
      </div>`).join("")}`;
}

function renderHow(how) {
  const ranks = how.ranks ? `<p class="muted">Search ranks, merged with Reciprocal Rank Fusion, then reranked:</p>
    <pre>${esc(how.ranks.map((r) => `#${r.result}  ${r.agreement}  keyword rank ${r["keyword rank"] ?? "-"}  meaning rank ${r["meaning rank"] ?? "-"}`).join("\n"))}</pre>` : "";
  return `<details><summary>How this was answered</summary>
    <p><b>Routing:</b> ${esc(how.routing)}</p>
    ${how.repaired ? `<p><b>Plan repaired by code:</b> ${esc(how.repaired)}</p>` : ""}
    ${how.plan ? `<p class="muted">The AI's plan. It only fills these blanks; code builds the query:</p><pre>${esc(JSON.stringify(how.plan, null, 2))}</pre>` : ""}
    ${how.sql ? `<pre>${esc(how.sql)}</pre>` : ""}${ranks}</details>`;
}

function wireRows(box, r) {
  box.querySelectorAll("[data-cite]").forEach((el) => {
    const pick = () => {
      box.querySelectorAll("tr.sel").forEach((t) => t.classList.remove("sel"));
      const key = el.dataset.cite;
      const last = store.get("ask-last"); if (last) { last.cite = key; store.set("ask-last", last); }
      if (key.startsWith("x")) {
        const [i, k] = key.slice(1).split(":").map(Number);
        const x = r.answers[i].results[k];
        setSource({doc: x.doc, page: x.page, pages: x.doc_pages, quote: x.text, title: x.agreement, label: x.section,
                   highlight: questionTerms(r.question)});
      } else {
        const [i, k] = key.split(":").map(Number);
        const x = r.answers[i].rows[k];
        el.classList.add("sel");
        setSource({doc: x.doc, page: x.page, pages: x.pages, quote: x.quote, title: x.doc_name, label: "Source",
                   highlight: x.highlight});
      }
    };
    el.onclick = (e) => { if (e.target.tagName !== "A") pick(); };
    el.onkeydown = (e) => { if (e.key === "Enter") pick(); };
  });
}

// ---------- Agreements ----------
async function viewAgreement(id) {
  $view.innerHTML = `<div class="card loading">Loading…</div>`;
  const list = await api("agreements");
  id = id || (list.find((x) => x.name.startsWith("Netgear")) || list[0]).id;
  const a = await api("agreement/" + encodeURIComponent(id));
  $view.innerHTML = `
    <div style="display:flex;flex-direction:column;gap:12px">
      <div class="row" style="justify-content:space-between">
        <nav aria-label="Breadcrumb" class="muted"><a href="#/agreements">All agreements</a> / ${esc(a.parties || a.id)}</nav>
        <label class="sr" for="pickag">Agreement</label>
        <select id="pickag">${list.map((x) => `<option value="${esc(x.id)}" ${x.id === a.id ? "selected" : ""}>${esc(x.name)}</option>`).join("")}</select>
      </div>
      <div><h1 style="font-size:32px">${esc(a.title)}</h1><div class="muted" style="font-size:16px">${esc(a.parties)}</div></div>
      ${a.orphan ? `<div class="flag">The master agreement isn't in this set, so terms come from this amendment alone.</div>` : ""}
      <ol class="docstrip" aria-label="Documents in this agreement, in order">
        ${a.documents.map((d, k) => `
          <li class="doc">${k ? '<span aria-hidden="true" class="arrow">' + ICON_NEXT + "</span>" : ""}<span class="num" aria-hidden="true">${d.n}</span>
          <span><a href="#" data-doc="${esc(d.id)}" data-pages="${d.pages}"><b>${esc(d.label)}</b></a> <span class="muted">· ${esc(d.date)} · ${esc(d.id)}</span></span></li>`).join("")}
      </ol>
      <div class="muted">Ordered by date, then amendment number, then filing order. Later documents override earlier ones.</div>
    </div>
    <div class="split">
      <section class="primary card" aria-labelledby="terms-h" style="padding:0;gap:0;overflow:hidden">
        <div class="row" style="justify-content:space-between;padding:18px 24px;border-bottom:1px solid var(--line)">
          <h2 id="terms-h" style="font-size:20px">Terms in force</h2><span class="muted">Select a term to see its source</span>
        </div>
        <div style="overflow-x:auto"><table style="min-width:640px">
          <thead><tr><th scope="col" style="padding-left:24px">Term</th><th scope="col">In force today</th><th scope="col">History, by document</th></tr></thead>
          <tbody>${a.terms.map((t, k) => `<tr class="pick" data-term="${k}" tabindex="0">
            <th scope="row" style="background:none;text-align:left;font-size:15px;color:var(--ink);padding:14px 16px 14px 24px;border-top:1px solid var(--line);border-bottom:0;vertical-align:top">${esc(t.term)}</th>
            <td>${esc(t.value)}${t.market ? `<div class="vs">${esc(t.market)}</div>` : ""}</td>
            <td><ol class="hist">${t.history.map((h) => `<li><span class="num sm" aria-label="Document ${h.n}">${h.n}</span><span>${esc(h.what)} <span class="muted">p.${h.page}</span></span></li>`).join("")}</ol></td>
          </tr>`).join("")}</tbody></table></div>
      </section>
      <aside class="aside card" id="source" aria-label="Source document"></aside>
    </div>`;
  document.getElementById("pickag").onchange = (e) => go("agreements/" + e.target.value);
  const docPages = Object.fromEntries(a.documents.map((d) => [d.id, d.pages]));
  const pickTerm = (k) => {
    const t = a.terms[k];
    $view.querySelectorAll("tr.sel").forEach((x) => x.classList.remove("sel"));
    $view.querySelector(`[data-term="${k}"]`).classList.add("sel");
    setSource({doc: t.doc, page: t.page || 1, pages: docPages[t.doc] || 1, quote: t.quote, title: a.title,
               label: "Source for " + t.term, highlight: t.highlight});
  };
  $view.querySelectorAll("[data-term]").forEach((el) => {
    el.onclick = () => pickTerm(Number(el.dataset.term));
    el.onkeydown = (e) => { if (e.key === "Enter") pickTerm(Number(el.dataset.term)); };
  });
  $view.querySelectorAll("[data-doc]").forEach((el) => el.onclick = (e) => {
    e.preventDefault();
    setSource({doc: el.dataset.doc, page: 1, pages: Number(el.dataset.pages), quote: "", title: el.textContent, label: "Document"});
  });
  const cap = a.terms.findIndex((t) => t.field === "cap_type");
  if (a.terms.length) pickTerm(cap >= 0 ? cap : 0);
  else { source.doc = null; renderSource(); }
}

// ---------- Review ----------
async function viewReview(idx) {
  $view.innerHTML = `<div class="card loading">Loading the review queue…</div>`;
  const r = await api("review");
  setCount(r.items.length);
  if (!r.items.length) { $view.innerHTML = `<div class="card"><h1>Review queue</h1><p>Nothing waiting. Every flagged answer has been reviewed (${r.done} decisions so far).</p></div>`; return; }
  const i = Math.min(Number(idx) || 0, r.items.length - 1);
  const it = r.items[i];
  $view.innerHTML = `
    <div class="row" style="justify-content:space-between">
      <div><h1 style="font-size:28px">Review queue</h1>
        <div class="muted" style="font-size:15px">Values the system was unsure about. Your decision is saved alongside the model's answer, never over it.</div></div>
      <div class="row"><span style="font-weight:600;font-size:14px">Item ${i + 1} of ${r.items.length}</span>
        <div class="progress" aria-hidden="true"><div style="width:${Math.round(100 * (i + 1) / r.items.length)}%"></div></div>
        <a class="btn ghost" href="#/review/${(i + 1) % r.items.length}">Skip for now</a></div>
    </div>
    <div class="split">
      <section class="card" style="flex:1 1 440px;min-width:0" aria-label="Item to review">
        <div class="muted">${esc(it.agreement)} · ${esc(it.term)}</div>
        <div style="display:flex;flex-direction:column;gap:6px"><div class="task-label">Your task</div><h2 class="task">${esc(it.question)}</h2></div>
        ${it.quote ? `<blockquote class="quote">"${highlight(it.quote.slice(0, 500), it.highlight)}"</blockquote>
          ${it.page ? `<div><button class="btn link" type="button" id="jump">Show this clause on page ${it.page}</button></div>` : ""}` : ""}
        <div class="flag"><b>Why it's here:</b> ${esc(it.why)}</div>
        <fieldset style="border:0;border-top:1px solid var(--line);margin:0;padding:18px 0 0;display:flex;flex-direction:column;gap:10px">
          <legend style="float:left;width:100%;padding:0 0 4px;font-weight:600">Your answer</legend>
          <label class="check"><input type="checkbox" id="read">I read the source${it.page ? " on page " + it.page : ""}</label>
          ${it.labels.yes ? `<button class="btn primary wide" id="yes" disabled type="button">${esc(it.labels.yes)}</button>` : ""}
          <div class="row">
            <label class="sr" for="fix">Correct value</label>
            <button class="btn wide" style="flex:1 1 220px" id="diff" disabled type="button">${esc(it.labels.diff)}</button>
            <input id="fix" type="text" style="flex:1 1 140px;min-height:52px" placeholder="${esc(it.hint || "the correct value")}">
          </div>
          <button class="btn wide" id="none" disabled type="button">${esc(it.labels.none)}</button>
          <label class="sr" for="note">Note</label><input id="note" type="text" placeholder="Note (optional)">
          <div class="muted" id="msg">Answers unlock once you confirm you read the source, so nothing gets approved unread.</div>
        </fieldset>
      </section>
      <aside class="aside card" id="source" style="flex:999 1 560px" aria-label="Source document"></aside>
    </div>`;
  const open = () => setSource({doc: it.doc, page: it.page || 1, pages: it.pages, quote: it.quote || "", title: it.agreement,
                                label: "Clause at issue, boxed in red", highlight: it.highlight});
  open();
  const jump = document.getElementById("jump"); if (jump) jump.onclick = open;
  const read = document.getElementById("read");
  const btns = ["yes", "diff", "none"].map((x) => document.getElementById(x)).filter(Boolean);
  read.onchange = () => btns.forEach((b) => b.disabled = !read.checked);
  const send = async (status, value) => {
    btns.forEach((b) => b.disabled = true);
    try {
      await api("review", {doc: it.doc, field: it.field, status, value, note: document.getElementById("note").value,
                           page: it.page, quote: it.quote});
      go("review/" + i);   // the saved item leaves the queue; the next one slides into this position
    } catch (e) { document.getElementById("msg").textContent = "Couldn't save: " + e.message; btns.forEach((b) => b.disabled = false); }
  };
  const yes = document.getElementById("yes"); if (yes) yes.onclick = () => send("accepted", null);
  document.getElementById("diff").onclick = () => {
    const val = document.getElementById("fix").value.trim();
    if (!val) { document.getElementById("msg").textContent = "Type the correct value first."; return; }
    send("corrected", it.field === "liability_cap" ? JSON.stringify({cap_type: val, cap_amount: null}) : val);
  };
  document.getElementById("none").onclick = () => send("absent", null);
}

// ---------- routing ----------
function setCount(n) { document.getElementById("review-count").textContent = n || ""; }
function go(path, params) {
  const qs = params ? "?" + new URLSearchParams(params) : "";
  const target = "#/" + path + qs;
  if (location.hash === target) route(); else location.hash = target;
}
async function route() {
  const [path, qs] = location.hash.replace(/^#\/?/, "").split("?");
  const [page, arg] = path.split("/");
  const params = new URLSearchParams(qs || "");
  document.querySelectorAll("[data-nav]").forEach((a) => a.classList.toggle("on", a.dataset.nav === (page || "ask")));
  try {
    if (page === "agreements") await viewAgreement(arg);
    else if (page === "review") await viewReview(arg);
    else viewAsk(params);
  } catch (e) { $view.innerHTML = `<div class="card"><div class="note error">Couldn't load this screen: ${esc(e.message)}. Is the API server running?</div></div>`; }
}
window.addEventListener("hashchange", route);
api("meta").then((m) => { setCount(m.review_waiting); document.getElementById("meta").textContent = `${m.agreements} agreements · CUAD sample · runs on this Mac`; }).catch(() => {});
route();
