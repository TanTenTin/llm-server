"""
관측용 임시 대시보드 (self-contained HTML).

`GET /dashboard`가 이 HTML 셸을 그대로 반환한다. 셸 자체엔 비밀이 없어 무인증으로
서빙하고, 페이지가 브라우저에서 **동일 출처**로 `/health`·`/health/providers`·
`/v1/usage`를 폴링한다(그래서 CORS·CSP 문제가 없다). 게이트웨이 키가 설정돼 있으면
사용자가 입력한 키를 localStorage에 저장해 `Authorization: Bearer`로 붙인다.

인메모리 집계라 서버 재기동 시 통계가 리셋된다(usage.py의 전제와 동일). 운영 관측이
아니라 "지금 무슨 일이 벌어지는지" 눈으로 확인하는 임시 도구로 설계했다.
"""

# HTML은 f-string이 아니라 일반 문자열이라 CSS/JS의 중괄호를 그대로 쓸 수 있다.
DASHBOARD_HTML = """<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LLM Gateway Dashboard</title>
<style>
  :root {
    --bg: #f6f7f9; --panel: #ffffff; --border: #e3e6ea; --text: #1a1d21;
    --muted: #6b7280; --accent: #3b82f6; --ok: #16a34a; --warn: #d97706;
    --err: #dc2626; --off: #9ca3af; --row: #fafbfc; --mono: ui-monospace, SFMono-Regular, Menlo, monospace;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #0e1116; --panel: #161b22; --border: #272d36; --text: #e6edf3;
      --muted: #8b949e; --accent: #58a6ff; --ok: #3fb950; --warn: #d29922;
      --err: #f85149; --off: #6e7681; --row: #1b2027;
    }
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: var(--bg); color: var(--text);
    font-family: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
    font-size: 14px; line-height: 1.5;
  }
  .wrap { max-width: 1080px; margin: 0 auto; padding: 20px 16px 48px; }
  header {
    display: flex; align-items: center; gap: 12px; flex-wrap: wrap;
    margin-bottom: 20px;
  }
  h1 { font-size: 18px; margin: 0; font-weight: 650; letter-spacing: -0.01em; }
  .dot { width: 9px; height: 9px; border-radius: 50%; display: inline-block; }
  .dot.live { background: var(--ok); box-shadow: 0 0 0 0 rgba(63,185,80,.6); animation: pulse 2s infinite; }
  .dot.stale { background: var(--off); }
  @keyframes pulse {
    0% { box-shadow: 0 0 0 0 rgba(63,185,80,.5); }
    70% { box-shadow: 0 0 0 6px rgba(63,185,80,0); }
    100% { box-shadow: 0 0 0 0 rgba(63,185,80,0); }
  }
  .spacer { flex: 1 1 auto; }
  .controls { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
  .controls label { color: var(--muted); font-size: 12px; }
  select, button, input {
    font: inherit; font-size: 13px; color: var(--text); background: var(--panel);
    border: 1px solid var(--border); border-radius: 7px; padding: 5px 9px;
  }
  button { cursor: pointer; }
  button:hover { border-color: var(--accent); }
  input.key { width: 180px; font-family: var(--mono); }
  .meta { color: var(--muted); font-size: 12px; }
  .grid {
    display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
    gap: 12px; margin-bottom: 22px;
  }
  .card {
    background: var(--panel); border: 1px solid var(--border);
    border-radius: 10px; padding: 13px 15px;
  }
  .card .name { font-weight: 600; display: flex; align-items: center; gap: 8px; }
  .card .line { color: var(--muted); font-size: 12.5px; margin-top: 5px; display: flex; justify-content: space-between; }
  .card .line b { color: var(--text); font-weight: 550; }
  .grid.two { grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); }
  .card .subhead { font-weight: 600; margin-bottom: 8px; display: flex; align-items: center; justify-content: space-between; }
  .card .subhead .count { color: var(--muted); font-weight: 500; font-size: 12px; }
  .mlist { list-style: none; margin: 0; padding: 0; }
  .mlist li { display: flex; justify-content: space-between; gap: 10px; padding: 5px 0; border-top: 1px solid var(--border); font-size: 12.5px; }
  .mlist li:first-child { border-top: none; }
  .mlist .mname { font-family: var(--mono); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .mlist .mtag { color: var(--muted); font-size: 11px; white-space: nowrap; }
  .mlist .loaded-dot { color: var(--ok); }
  .badge { font-size: 11px; padding: 1px 7px; border-radius: 999px; font-weight: 600; }
  .badge.ok { color: var(--ok); background: color-mix(in srgb, var(--ok) 14%, transparent); }
  .badge.off { color: var(--off); background: color-mix(in srgb, var(--off) 16%, transparent); }
  .badge.warn { color: var(--warn); background: color-mix(in srgb, var(--warn) 16%, transparent); }
  .badge.err { color: var(--err); background: color-mix(in srgb, var(--err) 16%, transparent); }
  section h2 {
    font-size: 13px; text-transform: uppercase; letter-spacing: 0.05em;
    color: var(--muted); margin: 0 0 10px; font-weight: 600;
  }
  .seg { display: inline-flex; border: 1px solid var(--border); border-radius: 7px; overflow: hidden; margin-left: 8px; }
  .seg button { border: none; border-radius: 0; background: var(--panel); padding: 3px 11px; font-size: 12px; }
  .seg button.active { background: var(--accent); color: #fff; }
  .table-wrap { overflow-x: auto; border: 1px solid var(--border); border-radius: 10px; background: var(--panel); }
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th, td { text-align: right; padding: 9px 13px; white-space: nowrap; }
  th:first-child, td:first-child { text-align: left; }
  thead th { color: var(--muted); font-weight: 600; font-size: 11.5px; text-transform: uppercase; letter-spacing: 0.03em; border-bottom: 1px solid var(--border); }
  tbody tr:nth-child(even) { background: var(--row); }
  tbody td:first-child { font-family: var(--mono); font-size: 12.5px; }
  td.num { font-variant-numeric: tabular-nums; }
  .fallback { color: var(--warn); font-size: 11px; }
  .errcell { font-size: 11.5px; }
  .err-pill { color: var(--err); }
  .empty { color: var(--muted); padding: 24px; text-align: center; }
  .banner { padding: 10px 14px; border-radius: 9px; margin-bottom: 18px; font-size: 13px; display: none; }
  .banner.show { display: block; }
  .banner.error { background: color-mix(in srgb, var(--err) 12%, transparent); color: var(--err); border: 1px solid color-mix(in srgb, var(--err) 40%, transparent); }
  footer { margin-top: 24px; color: var(--muted); font-size: 11.5px; }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <span id="live" class="dot stale"></span>
    <h1>LLM Gateway Dashboard</h1>
    <span class="spacer"></span>
    <div class="controls">
      <input id="key" class="key" type="password" placeholder="gateway key (선택)" autocomplete="off">
      <label for="interval">새로고침</label>
      <select id="interval">
        <option value="0">수동</option>
        <option value="3000">3s</option>
        <option value="5000" selected>5s</option>
        <option value="10000">10s</option>
        <option value="30000">30s</option>
      </select>
      <button id="refresh">지금</button>
    </div>
  </header>

  <div id="banner" class="banner"></div>

  <section>
    <h2>Providers</h2>
    <div id="providers" class="grid"></div>
  </section>

  <section>
    <h2>LLM 서버 (Ollama) <span id="ollama-sub" class="meta"></span></h2>
    <div id="ollama" class="grid two"></div>
  </section>

  <section>
    <h2>모델별 사용량
      <span class="seg">
        <button id="tab-today" class="active" data-scope="today">오늘</button>
        <button id="tab-total" data-scope="total">전체(7일)</button>
      </span>
    </h2>
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>모델 (provider:model)</th>
            <th>요청</th>
            <th>입력 tok</th>
            <th>출력 tok</th>
            <th>합계 tok</th>
            <th>에러</th>
          </tr>
        </thead>
        <tbody id="usage-body"></tbody>
      </table>
    </div>
  </section>

  <footer id="footer"></footer>
</div>

<script>
(function () {
  "use strict";
  var els = {
    live: document.getElementById("live"),
    key: document.getElementById("key"),
    interval: document.getElementById("interval"),
    refresh: document.getElementById("refresh"),
    banner: document.getElementById("banner"),
    providers: document.getElementById("providers"),
    ollama: document.getElementById("ollama"),
    ollamaSub: document.getElementById("ollama-sub"),
    usageBody: document.getElementById("usage-body"),
    footer: document.getElementById("footer"),
    tabToday: document.getElementById("tab-today"),
    tabTotal: document.getElementById("tab-total"),
  };
  var state = { scope: "today", timer: null, lastData: null };

  // ── localStorage 로부터 키/설정 복원 ──
  els.key.value = localStorage.getItem("gw_key") || "";
  var savedInterval = localStorage.getItem("gw_interval");
  if (savedInterval !== null) els.interval.value = savedInterval;

  els.key.addEventListener("change", function () {
    localStorage.setItem("gw_key", els.key.value.trim());
    tick();
  });
  els.interval.addEventListener("change", function () {
    localStorage.setItem("gw_interval", els.interval.value);
    schedule();
  });
  els.refresh.addEventListener("click", tick);
  els.tabToday.addEventListener("click", function () { setScope("today"); });
  els.tabTotal.addEventListener("click", function () { setScope("total"); });

  function setScope(scope) {
    state.scope = scope;
    els.tabToday.classList.toggle("active", scope === "today");
    els.tabTotal.classList.toggle("active", scope === "total");
    if (state.lastData) renderUsage(state.lastData.usage);
  }

  // ── 유틸 ──
  function fmtNum(n) {
    if (n == null) return "0";
    if (n >= 1e6) return (n / 1e6).toFixed(1) + "M";
    if (n >= 1e3) return (n / 1e3).toFixed(1) + "k";
    return String(n);
  }
  function fmtBytes(n) {
    if (!n) return "-";
    if (n >= 1e9) return (n / 1e9).toFixed(1) + " GB";
    if (n >= 1e6) return (n / 1e6).toFixed(0) + " MB";
    if (n >= 1e3) return (n / 1e3).toFixed(0) + " KB";
    return n + " B";
  }
  function fmtErrors(errors) {
    var keys = Object.keys(errors || {});
    if (!keys.length) return '<span class="meta">-</span>';
    return keys.map(function (k) {
      return '<span class="err-pill">' + k + '&times;' + errors[k] + '</span>';
    }).join(" ");
  }
  function headers() {
    var h = {};
    var k = els.key.value.trim();
    if (k) h["Authorization"] = "Bearer " + k;
    return h;
  }
  function showBanner(msg) {
    els.banner.textContent = msg;
    els.banner.className = "banner error show";
  }
  function clearBanner() { els.banner.className = "banner"; }

  async function getJSON(path) {
    var resp = await fetch(path, { headers: headers(), cache: "no-store" });
    if (resp.status === 401) {
      var e = new Error("401");
      e.unauthorized = true;
      throw e;
    }
    if (!resp.ok) throw new Error(path + " → HTTP " + resp.status);
    return resp.json();
  }

  // ── 렌더링 ──
  var PROVIDER_ORDER = ["gemini", "anthropic", "ollama"];
  function renderProviders(health, providers) {
    var frag = document.createDocumentFragment();
    PROVIDER_ORDER.forEach(function (name) {
      var p = (providers && providers[name]) || { registered: false };
      var card = document.createElement("div");
      card.className = "card";
      var badge, badgeText;
      if (!p.registered) { badge = "off"; badgeText = "미등록"; }
      else if (p.breaker_open) { badge = "warn"; badgeText = "차단됨"; }
      else if (name === "ollama" && p.reachable === false) { badge = "err"; badgeText = "unreachable"; }
      else { badge = "ok"; badgeText = "ok"; }

      var lines = "";
      lines += line("등록", p.registered ? "<b>yes</b>" : "no");
      if (p.registered) {
        if (p.breaker_open) {
          var secs = Math.round(p.breaker_remaining_seconds || 0);
          lines += line("회로차단기", '<b class="err-pill">open · ' + secs + 's</b>');
        } else {
          lines += line("회로차단기", "closed");
        }
        if (name === "ollama") {
          lines += line("reachable", p.reachable ? "<b>yes</b>" : '<b class="err-pill">no</b>');
        }
      }
      card.innerHTML =
        '<div class="name">' + name +
        ' <span class="badge ' + badge + '">' + badgeText + '</span></div>' + lines;
      frag.appendChild(card);
    });
    els.providers.innerHTML = "";
    els.providers.appendChild(frag);
  }
  function line(label, value) {
    return '<div class="line"><span>' + label + '</span><span>' + value + '</span></div>';
  }

  function renderOllama(o) {
    // 서브헤더: base_url · version · reachable
    if (!o || !o.reachable) {
      els.ollamaSub.textContent = o && o.base_url ? "· " + o.base_url + " · unreachable" : "· unreachable";
      els.ollama.innerHTML = '<div class="card"><div class="empty">LLM 서버에 연결할 수 없습니다.</div></div>';
      return;
    }
    els.ollamaSub.textContent =
      "· " + (o.base_url || "?") + (o.version ? " · v" + o.version : "");

    // 지금 로드된 모델 이름 집합 → 설치 목록에서 로드 여부 표시.
    var loadedNames = {};
    (o.loaded || []).forEach(function (m) { loadedNames[m.name] = m; });

    // 카드 1: 현재 로드된 모델 (메모리 상주)
    var loadedHtml = '<div class="card"><div class="subhead">현재 로드됨 (메모리 상주)' +
      '<span class="count">' + (o.loaded || []).length + '개</span></div>';
    if (!(o.loaded || []).length) {
      loadedHtml += '<div class="meta">로드된 모델 없음 (콜드 상태).</div>';
    } else {
      loadedHtml += '<ul class="mlist">' + o.loaded.map(function (m) {
        var mem = m.size_vram ? fmtBytes(m.size_vram) + " VRAM" : fmtBytes(m.size);
        var exp = m.expires_at ? " · ~" + new Date(m.expires_at).toLocaleTimeString() : "";
        return '<li><span class="mname"><span class="loaded-dot">●</span> ' + m.name +
          '</span><span class="mtag">' + mem + exp + '</span></li>';
      }).join("") + "</ul>";
    }
    loadedHtml += "</div>";

    // 카드 2: 설치된 모델 전체
    var installedHtml = '<div class="card"><div class="subhead">설치된 모델' +
      '<span class="count">' + (o.installed || []).length + '개</span></div>';
    if (!(o.installed || []).length) {
      installedHtml += '<div class="meta">설치된 모델 없음.</div>';
    } else {
      installedHtml += '<ul class="mlist">' + o.installed.map(function (m) {
        var isLoaded = loadedNames[m.name] ? '<span class="loaded-dot">●</span> ' : "";
        var tag = [m.parameter_size, m.quantization, fmtBytes(m.size)]
          .filter(Boolean).join(" · ");
        return '<li><span class="mname">' + isLoaded + m.name +
          '</span><span class="mtag">' + tag + '</span></li>';
      }).join("") + "</ul>";
    }
    installedHtml += "</div>";

    els.ollama.innerHTML = loadedHtml + installedHtml;
  }

  function renderUsage(usage) {
    var rows;
    if (state.scope === "total") {
      rows = usage.totals || {};
    } else {
      var today = new Date().toISOString().slice(0, 10); // UTC 날짜 (usage.py 집계 기준)
      rows = (usage.days && usage.days[today]) || {};
    }
    var labels = Object.keys(rows).sort(function (a, b) {
      return (rows[b].requests || 0) - (rows[a].requests || 0);
    });
    if (!labels.length) {
      els.usageBody.innerHTML =
        '<tr><td colspan="6" class="empty">집계된 요청이 없습니다' +
        (state.scope === "today" ? " (오늘 UTC 기준)" : "") + '.</td></tr>';
      return;
    }
    els.usageBody.innerHTML = labels.map(function (label) {
      var r = rows[label];
      var fb = r.served_as_fallback
        ? ' <span class="fallback">(fallback ' + r.served_as_fallback + ')</span>' : "";
      return "<tr>" +
        "<td>" + label + fb + "</td>" +
        '<td class="num">' + fmtNum(r.requests) + "</td>" +
        '<td class="num">' + fmtNum(r.prompt_tokens) + "</td>" +
        '<td class="num">' + fmtNum(r.completion_tokens) + "</td>" +
        '<td class="num">' + fmtNum(r.total_tokens) + "</td>" +
        '<td class="errcell">' + fmtErrors(r.errors) + "</td>" +
        "</tr>";
    }).join("");
  }

  function renderFooter(usage) {
    var paid = usage.paid_tokens_by_day || {};
    var paidStr = Object.keys(paid).length
      ? " · paid tokens: " + Object.keys(paid).sort().map(function (d) {
          return d + "=" + fmtNum(paid[d]);
        }).join(", ")
      : "";
    els.footer.textContent =
      "tracker 시작: " + (usage.tracker_started_at || "?") +
      " · 보존 " + (usage.retention_days || "?") + "일" +
      " · 갱신 " + new Date().toLocaleTimeString() + paidStr +
      " · 인메모리 집계(재기동 시 리셋)";
  }

  function setLive(on) {
    els.live.className = "dot " + (on ? "live" : "stale");
  }

  async function tick() {
    try {
      // /health 는 무인증, 나머지는 키 필요. 병렬 호출.
      var results = await Promise.all([
        getJSON("/health").catch(function () { return { status: "down" }; }),
        getJSON("/health/providers"),
        getJSON("/v1/usage"),
        getJSON("/health/ollama").catch(function () { return null; }),
      ]);
      var health = results[0], hp = results[1], usage = results[2], ollama = results[3];
      state.lastData = { usage: usage };
      clearBanner();
      renderProviders(health, hp.providers);
      renderOllama(ollama);
      renderUsage(usage);
      renderFooter(usage);
      setLive(true);
    } catch (err) {
      setLive(false);
      if (err.unauthorized) {
        showBanner("인증 실패 (401) — 우측 상단에 게이트웨이 키를 입력하세요.");
        els.key.focus();
      } else {
        showBanner("서버에 연결할 수 없습니다 — llm-server(포트 8000)가 실행 중인지 확인하세요. (" + err.message + ")");
      }
    }
  }

  function schedule() {
    if (state.timer) { clearInterval(state.timer); state.timer = null; }
    var ms = parseInt(els.interval.value, 10);
    if (ms > 0) state.timer = setInterval(tick, ms);
  }

  // ── 시작 ──
  tick();
  schedule();
})();
</script>
</body>
</html>
"""
