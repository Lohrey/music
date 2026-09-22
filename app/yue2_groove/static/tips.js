// ⓘ parameter tips: an icon appended to every component label listed in
// window.__BB_TIPS__ (injected inline before this script) and one global
// floating layer, immune to component overflow / stacking contexts.
(function () {
  var SELECTOR = 'span[data-testid="block-info"], span.label-text, .block-title';
  var tip = null;
  function ensureTip() {
    if (tip && tip.isConnected) return tip;
    tip = document.createElement('div');
    tip.id = 'bb-tip';
    document.body.appendChild(tip);
    return tip;
  }
  function hide() { if (tip) { tip.className = ''; tip.removeAttribute('data-src'); } }
  function show(icon) {
    var text = icon.getAttribute('data-tip');
    if (!text) return;
    var el = ensureTip();
    el.textContent = text;
    el.className = 'bb-tip-show';
    el.setAttribute('data-src', text);
    var r = icon.getBoundingClientRect();
    var w = el.offsetWidth, h = el.offsetHeight;
    var vw = window.innerWidth, vh = window.innerHeight;
    var left = r.left;
    if (left + w > vw - 8) left = vw - w - 8;
    if (left < 8) left = 8;
    var top = r.bottom + 8;
    if (top + h > vh - 8) top = r.top - h - 8;
    if (top < 8) top = 8;
    el.style.left = left + 'px';
    el.style.top = top + 'px';
  }
  function inject() {
    var tips = window.__BB_TIPS__ || {};
    var nodes = document.querySelectorAll(SELECTOR);
    for (var i = 0; i < nodes.length; i++) {
      var el = nodes[i];
      if (el.getAttribute('data-bb-tip') === '1') continue;
      var key = (el.textContent || '').trim();
      if (!tips[key]) continue;
      el.setAttribute('data-bb-tip', '1');
      var mark = document.createElement('span');
      mark.className = 'bb-i';
      mark.setAttribute('data-tip', tips[key]);
      mark.setAttribute('tabindex', '0');
      mark.setAttribute('role', 'img');
      mark.setAttribute('aria-label', tips[key]);
      mark.textContent = 'i';
      el.appendChild(mark);
    }
  }
  document.addEventListener('mouseover', function (e) {
    var i = e.target && e.target.closest ? e.target.closest('.bb-i') : null;
    if (i) show(i);
  });
  document.addEventListener('mouseout', function (e) {
    var i = e.target && e.target.closest ? e.target.closest('.bb-i') : null;
    if (i) hide();
  });
  document.addEventListener('focusin', function (e) {
    if (e.target && e.target.classList && e.target.classList.contains('bb-i')) show(e.target);
  });
  document.addEventListener('focusout', function (e) {
    if (e.target && e.target.classList && e.target.classList.contains('bb-i')) hide();
  });
  // clicking the icon must not activate the surrounding radio/checkbox label,
  // and must *show* the tip (never toggle it away: hover already showed it)
  document.addEventListener('click', function (e) {
    var i = e.target && e.target.closest ? e.target.closest('.bb-i') : null;
    if (i) {
      e.preventDefault();
      e.stopPropagation();
      show(i);
      return;
    }
    if (tip && tip.className === 'bb-tip-show') hide();
  }, true);
  document.addEventListener('scroll', hide, true);
  window.addEventListener('resize', hide);
  setInterval(inject, 600);
})();
