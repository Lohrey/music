// The current work: a hidden bridge textbox (#bb-current-work) that the
// client mirrors to localStorage, per browser rather than per server, so the
// band above the view survives a reload.
(function () {
  var KEY = 'bb-current';
  function area() {
    var box = document.getElementById('bb-current-work');
    return box ? box.querySelector('textarea') : null;
  }
  function setNative(el, value) {
    var setter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, 'value').set;
    setter.call(el, value);
    el.dispatchEvent(new Event('input', { bubbles: true }));
    el.dispatchEvent(new Event('change', { bubbles: true }));
  }
  var restored = false;
  setInterval(function () {
    var el = area();
    if (!el) return;
    var stored = '';
    try { stored = localStorage.getItem(KEY) || ''; } catch (e) {}
    if (!restored) {
      restored = true;
      if (!el.value && stored) setNative(el, stored);   // ask the server to validate it
      return;
    }
    if (el.value && el.value !== stored) {
      try { localStorage.setItem(KEY, el.value); } catch (e) {}   // server-set: remember it
    } else if (!el.value && stored) {
      try { localStorage.removeItem(KEY); } catch (e) {}          // server cleared a stale path
    }
  }, 500);
})();
