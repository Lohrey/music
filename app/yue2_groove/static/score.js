// abcjs score rendering.  Every score panel declares
// data-bb-abc="<label prefix of its ABC textbox>"; the script finds the matching
// textarea and renders into the panel's .bb-score-inner, re-rendering on
// rotation / resize so the staff width follows the panel.
(function () {
  // Look for the textarea whose block label starts with `prefix`, preferring the
  // panel's own tab (so identical labels in different tabs cannot cross-render).
  function findArea(prefix, panel) {
    var root = (panel && panel.closest && panel.closest('.tabitem')) || document;
    var areas = [];
    function scan(scope) {
      var labels = scope.querySelectorAll('span[data-testid="block-info"]');
      for (var i = 0; i < labels.length; i++) {
        var label = (labels[i].textContent || '').trim();
        if (label.indexOf(prefix) !== 0) continue;
        var block = labels[i].closest('.block') || labels[i].parentElement;
        var ta = block && block.querySelector('textarea');
        if (ta) areas.push(ta);
      }
    }
    scan(root);
    if (!areas.length && root !== document) scan(document);
    return areas.length ? areas[0] : null;
  }
  function renderPanel(panel) {
    var box = panel.querySelector('.bb-score-inner');
    if (!box) return;
    // read-only panels (SONG) carry the score in data-bb-abc-text and have no textarea
    var explicit = panel.getAttribute('data-bb-abc-text');
    var abc;
    if (explicit !== null) {
      abc = explicit;
    } else {
      var ta = findArea(panel.getAttribute('data-bb-abc') || 'ABC SCORE', panel);
      abc = ta ? (ta.value || '') : '';
    }
    var ink = getComputedStyle(document.documentElement).getPropertyValue('--bb-ink').trim() || '#F1ECE2';
    var key = ink + '|' + abc;
    // The key lives on the element, not in a JS cache: if Gradio re-creates the
    // tab DOM (which drops the rendered SVG), the fresh node has no key and the
    // score is drawn again instead of being skipped as "already rendered".
    var hasSvg = !!box.querySelector('svg');
    if (box.getAttribute('data-bb-key') === key && (hasSvg || !abc.trim())) return;
    box.setAttribute('data-bb-key', key);
    if (!abc.trim()) {
      var empty = panel.getAttribute('data-bb-empty') || 'No score yet.';
      box.innerHTML = '<div class="bb-score-empty">' + empty + '</div>';
      return;
    }
    box.innerHTML = '';
    try {
      if (window.ABCJS && ABCJS.renderAbc) {
        // fit the staff to the panel: a fixed 900px staffwidth overflows the
        // score box on phones and gets clipped / needs sideways scrolling
        var avail = Math.max(240, Math.min(900, (box.clientWidth || 340) - 18));
        ABCJS.renderAbc(box, abc, {
          responsive: 'resize', foregroundColor: ink,
          scale: avail < 520 ? 0.95 : 1.1,
          staffwidth: avail, paddingtop: 4, paddingbottom: 4,
        });
      } else {
        box.innerHTML = '<div class="bb-score-error">Score renderer not loaded.</div>';
      }
    } catch (e) {
      box.innerHTML = '<div class="bb-score-error">Could not render this ABC: '
        + String(e && e.message ? e.message : e).slice(0, 180) + '</div>';
    }
  }
  function render() {
    var panels = document.querySelectorAll('[data-bb-abc]');
    for (var i = 0; i < panels.length; i++) renderPanel(panels[i]);
  }
  // re-render on rotation/resize so the staff width follows the new panel width
  var bbResizeTimer = null;
  function bbRelayout() {
    if (bbResizeTimer) clearTimeout(bbResizeTimer);
    bbResizeTimer = setTimeout(function () {
      var boxes = document.querySelectorAll('.bb-score-inner');
      for (var i = 0; i < boxes.length; i++) boxes[i].removeAttribute('data-bb-key');
    }, 250);
  }
  window.addEventListener('resize', bbRelayout);
  window.addEventListener('orientationchange', bbRelayout);
  setInterval(render, 700);
})();
