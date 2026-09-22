// 04 LIBRARY: the details-pane player (spectrum + transport), the score view
// and the row-click behaviour of the works list.
//
// Gradio 6 does not replace the details pane when another work is picked: its
// HTML component morphs the DOM that is there into the new markup (same tag at
// the same position → the node is kept, its attributes synced to the new HTML,
// its children recursed).  So the very same <audio> and <button> elements live
// on with a new src — and every attribute the server HTML does not carry is
// stripped.  Hence the player state hangs off the <audio> as a property (it
// dies with the element, never with the markup) and the transport buttons are
// handled by one delegated listener — there is nothing per button to duplicate.
(function () {
  function fmt(value) {
    if (!isFinite(value) || value < 0) value = 0;
    var m = Math.floor(value / 60), s = Math.floor(value % 60);
    return m + ':' + (s < 10 ? '0' : '') + s;
  }
  // One shared AudioContext for every player: creating one per selection would
  // hit the browser's context limit. Each element gets its own source+analyser.
  var BB_AUDIO_CTX = null;
  var BB_AUDIOS = [];      // every <audio> that has a state, for reap()
  function audioCtx() {
    if (BB_AUDIO_CTX) return BB_AUDIO_CTX;
    var AC = window.AudioContext || window.webkitAudioContext;
    if (!AC) return null;
    try { BB_AUDIO_CTX = new AC(); } catch (error) { BB_AUDIO_CTX = null; }
    return BB_AUDIO_CTX;
  }
  function readInk() {
    return getComputedStyle(document.documentElement).getPropertyValue('--bb-ink').trim() || '#F1ECE2';
  }
  function playerOf(node) {
    return node && node.closest ? node.closest('[data-bb-player]') : null;
  }
  function drawViz(state) {
    var player = playerOf(state.audio);
    var canvas = player ? player.querySelector('[data-bb-viz]') : null;
    if (!canvas || !canvas.clientWidth || !canvas.clientHeight) return;
    var dpr = window.devicePixelRatio || 1;
    var w = canvas.clientWidth, h = canvas.clientHeight;
    if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
    }
    var ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    if (state.inkTick++ % 90 === 0) state.ink = readInk();
    ctx.fillStyle = state.ink;
    var mid = h / 2;
    if (!state.analyser || !state.playing) {
      ctx.globalAlpha = 0.4;
      ctx.fillRect(0, mid - 0.5, w, 1);
      ctx.globalAlpha = 1;
      return;
    }
    state.analyser.getByteFrequencyData(state.data);
    var bins = state.data.length;
    var step = 6, barW = 3;
    var bars = Math.max(1, Math.floor(w / step));
    var usable = Math.max(1, Math.floor(bins * 0.85));
    ctx.globalAlpha = 0.9;
    for (var i = 0; i < bars; i++) {
      // log-ish bin mapping: give the low end more bars, like a real analyser
      var f0 = Math.floor(Math.pow(i / bars, 1.7) * usable);
      var f1 = Math.max(f0 + 1, Math.floor(Math.pow((i + 1) / bars, 1.7) * usable));
      var sum = 0, n = 0;
      for (var j = f0; j < f1 && j < bins; j++) { sum += state.data[j] || 0; n++; }
      var level = n ? (sum / n) / 255 : 0;
      // gentle high-frequency tilt so the right half keeps moving too
      level = Math.min(1, level * (0.6 + 1.3 * Math.pow(i / bars, 0.7)));
      var bh = Math.max(1.5, level * (mid - 1) * 1.25);
      ctx.fillRect(i * step, mid - bh, barW, bh * 2);
    }
    ctx.globalAlpha = 1;
  }
  var BB_VIZ_RUNNING = false;
  function vizFrame() {
    var active = 0;
    var audios = document.querySelectorAll('[data-bb-player] audio');
    for (var i = 0; i < audios.length; i++) {
      var state = audios[i].__bbState;
      if (!state) continue;
      if (state.playing) active++;
      drawViz(state);
    }
    if (active) { requestAnimationFrame(vizFrame); }
    else { BB_VIZ_RUNNING = false; }
  }
  function startViz() {
    if (BB_VIZ_RUNNING) return;
    BB_VIZ_RUNNING = true;
    requestAnimationFrame(vizFrame);
  }
  function ensureAnalyser(state) {
    if (state.analyser) return;
    var ctx = audioCtx();
    if (!ctx) return;
    try {
      var source = ctx.createMediaElementSource(state.audio);
      var analyser = ctx.createAnalyser();
      analyser.fftSize = 128;
      analyser.smoothingTimeConstant = 0.82;
      source.connect(analyser);
      analyser.connect(ctx.destination);
      state.analyser = analyser;
      state.data = new Uint8Array(analyser.frequencyBinCount);
    } catch (error) { state.analyser = null; }
  }
  // ── the player ───────────────────────────────────────────────────────
  // Gradio 6 does not replace the details pane when another work is picked:
  // its HTML component morphs the DOM that is there into the new markup (same
  // tag at the same position → the node is kept, its attributes synced to the
  // new HTML, its children recursed).  So the very same <audio> and <button>
  // elements live on with a new src — and every attribute the server HTML
  // does not carry is stripped, which is how a "bound" marker on the wrapper
  // vanished and the same button was bound again: two click listeners,
  // play() then pause(), a dead play button on every second selection.
  // Hence: the state hangs off the <audio> as a property (it dies with the
  // element, never with the markup), and the transport buttons are handled
  // by one delegated listener — there is nothing per button to duplicate.
  function paint(audio) {
    var player = playerOf(audio);
    if (!player) return;
    var btn = player.querySelector('[data-bb-play]');
    var fill = player.querySelector('.bb-pfill');
    var time = player.querySelector('[data-bb-time]');
    if (btn) {
      var playing = !audio.paused;
      btn.classList.toggle('bb-playing', playing);
      var label = playing ? 'Pause' : 'Play';
      btn.setAttribute('aria-label', label);
      btn.title = label;
    }
    if (fill && audio.duration) {
      fill.style.width = Math.min(100, (audio.currentTime / audio.duration) * 100) + '%';
    }
    if (time) time.textContent = fmt(audio.currentTime) + ' / ' + fmt(audio.duration);
  }
  function timeOf(audio) {
    var player = playerOf(audio);
    return player ? player.querySelector('[data-bb-time]') : null;
  }
  function stateOf(audio) {
    if (audio.__bbState) return audio.__bbState;
    var state = { audio: audio, analyser: null, data: null, playing: false,
                  ink: readInk(), inkTick: 0 };
    audio.__bbState = state;
    BB_AUDIOS.push(audio);
    function repaint() { paint(audio); }
    audio.addEventListener('timeupdate', repaint);
    audio.addEventListener('loadedmetadata', repaint);
    audio.addEventListener('play', function () { state.playing = true; startViz(); paint(audio); });
    audio.addEventListener('pause', function () { state.playing = false; paint(audio); });
    audio.addEventListener('ended', function () { state.playing = false; paint(audio); });
    // a swapped src runs the load algorithm: the element is paused again
    // without a pause event, and the viz would keep drawing the old track
    audio.addEventListener('emptied', function () { state.playing = false; paint(audio); });
    audio.addEventListener('error', function () {
      var time = timeOf(audio);
      if (time) time.textContent = 'audio unavailable';
    });
    paint(audio);
    drawViz(state);   // idle baseline until playback starts
    return state;
  }
  function togglePlay(audio) {
    if (!audio.paused) { audio.pause(); return; }
    var ctx = audioCtx();
    // 'suspended' until the first gesture; iOS also reports 'interrupted'
    // after a call or another app took the output
    if (ctx && ctx.state !== 'running') { try { ctx.resume(); } catch (error) {} }
    ensureAnalyser(stateOf(audio));
    var p = audio.play();
    if (p && p.catch) p.catch(function () {
      if (audio.error) return;        // the error listener owns a broken file
      var time = timeOf(audio);
      if (time) time.textContent = 'playback blocked — click play again';
    });
  }
  function jumpTo(audio, target) {
    try {
      var dur = isFinite(audio.duration) ? audio.duration : 0;
      var next = target;
      if (next < 0) next = 0;
      if (dur && next > dur - 0.05) next = Math.max(0, dur - 0.05);
      audio.currentTime = next;
    } catch (error) {}
    paint(audio);
  }
  // transport: one listener for every player there will ever be
  document.addEventListener('click', function (event) {
    var button = event.target && event.target.closest ? event.target.closest('[data-bb-player] button') : null;
    if (!button) return;
    var player = playerOf(button);
    var audio = player ? player.querySelector('audio') : null;
    if (!audio) return;
    stateOf(audio);
    if (button.hasAttribute('data-bb-play')) {
      togglePlay(audio);
    } else if (button.hasAttribute('data-bb-back')) {
      // jump back to the top; the play/pause state is left alone, so a
      // running track simply continues from the beginning
      try { audio.currentTime = 0; } catch (error) {}
      paint(audio);
    } else if (button.hasAttribute('data-bb-jump')) {
      jumpTo(audio, audio.currentTime + (parseFloat(button.getAttribute('data-bb-jump')) || 0));
    } else if (button.hasAttribute('data-bb-end')) {
      var dur = isFinite(audio.duration) ? audio.duration : 0;
      jumpTo(audio, dur ? dur - 0.05 : audio.currentTime);
    }
  });
  // seek bar: a pointerdown starts the drag, the window finishes it
  var BB_SEEKING = null;
  function seekBar(seek, event) {
    var player = playerOf(seek);
    var audio = player ? player.querySelector('audio') : null;
    if (!audio) return;
    var rect = seek.getBoundingClientRect();
    var clientX = (event.touches && event.touches[0] ? event.touches[0].clientX : event.clientX);
    if (!audio.duration || !rect.width) return;
    var ratio = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width));
    audio.currentTime = ratio * audio.duration;
    paint(audio);
  }
  document.addEventListener('pointerdown', function (event) {
    var seek = event.target && event.target.closest ? event.target.closest('[data-bb-seek]') : null;
    if (!seek) return;
    BB_SEEKING = seek;
    seekBar(seek, event);
  });
  window.addEventListener('pointermove', function (event) { if (BB_SEEKING) seekBar(BB_SEEKING, event); });
  window.addEventListener('pointerup', function () { BB_SEEKING = null; });
  function renderScore() {
    var box = document.getElementById('bb-lib-score-inner');
    if (!box) return;
    var source = document.getElementById('bb-lib-abc-src');
    var abc = source ? (source.textContent || '') : '';
    if (!abc) {
      var wrap = document.getElementById('bb-lib-abc');
      var area = wrap ? wrap.querySelector('textarea') : null;
      abc = area ? (area.value || '') : '';
    }
    var ink = getComputedStyle(document.documentElement).getPropertyValue('--bb-ink').trim() || '#F1ECE2';
    var key = ink + '|' + abc;
    var hasSvg = !!box.querySelector('svg');
    if (box.getAttribute('data-bb-key') === key && (hasSvg || !abc.trim())) return;
    box.setAttribute('data-bb-key', key);
    if (!abc.trim()) {
      box.innerHTML = '<div class="bb-score-empty">Select a work to view its score.</div>';
      return;
    }
    box.innerHTML = '';
    try {
      if (window.ABCJS && ABCJS.renderAbc) {
        var avail = Math.max(240, Math.min(900, (box.clientWidth || 340) - 18));
        ABCJS.renderAbc(box, abc, {
          responsive: 'resize', foregroundColor: ink,
          scale: avail < 520 ? 0.95 : 1.1,
          staffwidth: avail, paddingtop: 4, paddingbottom: 4,
        });
      } else {
        box.innerHTML = '<div class="bb-score-error">Score renderer not loaded.</div>';
      }
    } catch (error) {
      box.innerHTML = '<div class="bb-score-error">Could not render this ABC: '
        + String(error && error.message ? error.message : error).slice(0, 160) + '</div>';
    }
  }
  function rowRel(label) {
    var input = label ? label.querySelector('input[type="checkbox"]') : null;
    if (!input) return null;
    return input.getAttribute('name') || input.getAttribute('title') || input.value || null;
  }
  function viewRow(rel) {
    if (!rel) return;
    var box = document.getElementById('bb-lib-active');
    var area = box ? box.querySelector('textarea') : null;
    if (!area || area.value === rel) return;
    var setter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, 'value').set;
    setter.call(area, rel);
    area.dispatchEvent(new Event('input', { bubbles: true }));
    area.dispatchEvent(new Event('change', { bubbles: true }));
  }
  function markRows() {
    var spans = document.querySelectorAll('#bb-lib-list label span');
    for (var i = 0; i < spans.length; i++) {
      var span = spans[i];
      if (span.getAttribute('data-bb-row') === '1') continue;
      span.setAttribute('data-bb-row', '1');
      span.setAttribute('role', 'button');
      span.setAttribute('tabindex', '0');
      span.setAttribute('title', 'View details');
    }
  }
  function markActive() {
    var box = document.getElementById('bb-lib-active');
    var area = box ? box.querySelector('textarea') : null;
    var rel = area ? area.value : '';
    var labels = document.querySelectorAll('#bb-lib-list label');
    for (var i = 0; i < labels.length; i++) {
      labels[i].classList.toggle('bb-active', !!rel && rowRel(labels[i]) === rel);
    }
  }
  // Clicking the row text views a work; only the checkbox itself selects it.
  document.addEventListener('click', function (event) {
    var label = event.target && event.target.closest ? event.target.closest('#bb-lib-list label') : null;
    if (!label) return;
    var input = label.querySelector('input[type="checkbox"]');
    if (!input) return;
    // direct hit on the input (mouse, keyboard or programmatic) always toggles
    var target = event.target;
    if (target === input || (target.closest && target.closest('input[type="checkbox"]'))) return;
    var box = input.getBoundingClientRect();
    var lab = label.getBoundingClientRect();
    var left = box.width ? box.left : lab.left;
    var right = box.width ? box.right : lab.left + 30;
    var top = box.height ? box.top : lab.top;
    var bottom = box.height ? box.bottom : lab.bottom;
    var pad = 6;
    var onCheckbox = event.clientX >= left - pad && event.clientX <= right + pad &&
                     event.clientY >= top - pad && event.clientY <= bottom + pad;
    if (onCheckbox) return;
    event.preventDefault();
    event.stopPropagation();
    viewRow(rowRel(label));
  }, true);
  document.addEventListener('keydown', function (event) {
    var span = event.target && event.target.closest ? event.target.closest('#bb-lib-list label span') : null;
    if (!span) return;
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      viewRow(rowRel(span.closest('label')));
    }
  });
  function reap() {
    // the morph can also recycle a wrapper into other markup and drop the
    // <audio> in it; a detached element keeps playing (and its analyser
    // graph alive) in some browsers, so the element decides, not the wrapper
    for (var i = BB_AUDIOS.length - 1; i >= 0; i--) {
      var audio = BB_AUDIOS[i];
      if (!document.contains(audio)) {
        try { audio.pause(); } catch (error) {}
        audio.removeAttribute('src');
        BB_AUDIOS.splice(i, 1);
      }
    }
  }
  function tick() {
    var audios = document.querySelectorAll('[data-bb-player] audio');
    for (var i = 0; i < audios.length; i++) stateOf(audios[i]);
    markRows();
    markActive();
    reap();
    renderScore();
  }
  // A freshly rendered player should draw its idle baseline at once rather
  // than at the next 700 ms tick; clicks never wait for either (delegated).
  var bbTickTimer = null;
  function scheduleTick() {
    if (bbTickTimer) return;
    bbTickTimer = setTimeout(function () { bbTickTimer = null; tick(); }, 50);
  }
  function observeLibrary() {
    var root = document.getElementById('bb-lib-info') || document.body;
    if (!root || !window.MutationObserver) return;
    new MutationObserver(function (records) {
      for (var i = 0; i < records.length; i++) {
        if (records[i].addedNodes && records[i].addedNodes.length) { scheduleTick(); return; }
      }
    }).observe(root, { childList: true, subtree: true });
  }
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', observeLibrary);
  } else {
    observeLibrary();
  }
  setInterval(tick, 700);
  var pending = null;
  window.addEventListener('resize', function () {
    if (pending) clearTimeout(pending);
    pending = setTimeout(function () {
      var box = document.getElementById('bb-lib-score-inner');
      if (box) box.removeAttribute('data-bb-key');
    }, 250);
  });
})();
