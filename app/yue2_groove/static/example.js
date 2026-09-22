// Small "fill the repository example" buttons inside the STYLE / LYRICS
// boxes.  The example text itself is injected inline as window.__BB_EXAMPLES__
// before this script.
(function () {
  var FIELDS = { STYLE: 'style', LYRICS: 'lyrics' };
  function setNative(area, text) {
    var setter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, 'value').set;
    setter.call(area, text);
    area.dispatchEvent(new Event('input', { bubbles: true }));
    area.dispatchEvent(new Event('change', { bubbles: true }));
    area.focus();
  }
  function bindFocus(area) {
    if (area.getAttribute('data-bb-ph') === '1') return;
    area.setAttribute('data-bb-ph', '1');
    var hint = area.placeholder || '';
    // clicking in clears the grey example so a paste lands straight away;
    // leaving the field empty restores it
    area.addEventListener('focus', function () { if (!area.value) area.placeholder = ''; });
    area.addEventListener('blur', function () { if (!area.value) area.placeholder = hint; });
  }
  function inject() {
    var examples = window.__BB_EXAMPLES__ || {};
    var tabs = document.querySelectorAll('.tabitem');
    for (var tabIndex = 0; tabIndex < tabs.length; tabIndex++) {
      injectTab(tabs[tabIndex], examples);
    }
  }
  function injectTab(first, examples) {
    var nodes = first.querySelectorAll('span[data-testid="block-info"]');
    for (var i = 0; i < nodes.length; i++) {
      var info = nodes[i];
      var text = (info.textContent || '').trim();
      var key = FIELDS[text.split(/\s+/)[0]];
      if (!key) continue;
      var block = info.closest('.block') || first;
      var host = block.querySelector('.input-container');
      var area = host ? host.querySelector('textarea') : null;
      if (!host || !area || area.readOnly || area.disabled ||
          host.getAttribute('data-bb-eg') === '1') continue;
      host.setAttribute('data-bb-eg', '1');
      host.classList.add('bb-eg-host');
      bindFocus(area);
      var btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'bb-eg';
      btn.textContent = 'E';
      btn.title = 'Fill the repository example (City Lights)';
      btn.setAttribute('aria-label', btn.title);
      (function (target, value) {
        btn.addEventListener('click', function (event) {
          event.preventDefault();
          event.stopPropagation();
          setNative(target, value);
        });
      })(area, examples[key] || '');
      host.appendChild(btn);
    }
  }
  setInterval(inject, 600);
})();
