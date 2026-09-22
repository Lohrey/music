// Gradio's frontend re-applies a component's initial value when its tab is
// re-activated, which drops typed text and radio choices.  Keep a small
// client-side store and restore the user's values (dispatching events so
// Gradio's own state follows).
(function () {
  var store = {};
  var restoreUntil = 0;
  function tabIdx(el) {
    var t = el.closest('.tabitem'); if (!t) return -1;
    return Array.prototype.indexOf.call(document.querySelectorAll('.tabitem'), t);
  }
  function blockIdx(el) {
    var t = el.closest('.tabitem'), b = el.closest('.block');
    if (!t || !b) return -1;
    return Array.prototype.indexOf.call(t.querySelectorAll('.block'), b);
  }
  function labelOf(el) {
    var b = el.closest('.block') || el.parentElement;
    var info = b && b.querySelector('span[data-testid="block-info"]');
    if (!info) return '';
    return (info.textContent || '').replace(/[i+\-]+$/, '').trim();
  }
  function keyOf(el) { return labelOf(el) + '|' + tabIdx(el) + '|' + blockIdx(el); }
  function optionKey(el) {
    var l = el.closest('label');
    return keyOf(el) + '|' + (l ? (l.innerText || '').trim().slice(0, 40) : el.value);
  }
  function isOutput(el) { return !!el.closest('.bb-output'); }
  function setNative(el, value) {
    var proto = el.tagName === 'TEXTAREA' ? window.HTMLTextAreaElement.prototype
                                          : window.HTMLInputElement.prototype;
    var setter = Object.getOwnPropertyDescriptor(proto, 'value').set;
    setter.call(el, value);
    el.dispatchEvent(new Event('input', {bubbles: true}));
    el.dispatchEvent(new Event('change', {bubbles: true}));
  }
  var EDITABLE = 'textarea, input[type=text], input[type=number], input[type=search], input:not([type])';
  function snapshot(el) {
    if (!el || el.disabled || el.readOnly || isOutput(el)) return;
    if (el.matches('input[type=checkbox], input[type=radio]')) {
      store[optionKey(el)] = el.checked ? 1 : 0;
      return;
    }
    if (el.matches(EDITABLE)) store[keyOf(el)] = el.value;
  }
  function sync() {
    var inWindow = Date.now() < restoreUntil;
    var els = document.querySelectorAll('textarea, input');
    for (var i = 0; i < els.length; i++) {
      var el = els[i];
      if (el.disabled || el.readOnly || el.type === 'file' || isOutput(el)) continue;
      if (el.matches('input[type=checkbox], input[type=radio]')) {
        var ok = optionKey(el);
        if (!(ok in store)) continue;
        if (!!el.checked !== !!store[ok]) {
          if (inWindow && el.offsetParent !== null) el.click();
          else store[ok] = el.checked ? 1 : 0;  // server-driven update wins
        }
        continue;
      }
      if (!el.matches(EDITABLE)) continue;
      var key = keyOf(el);
      if (!(key in store)) continue;
      if (el.value !== store[key]) {
        if (inWindow) setNative(el, store[key]);
        else store[key] = el.value;  // server-driven update wins
      }
    }
  }
  document.addEventListener('input', function (e) {
    if (e.target && e.target.matches && e.target.matches('textarea, input')) snapshot(e.target);
  }, true);
  document.addEventListener('change', function (e) {
    if (e.target && e.target.matches && e.target.matches('textarea, input')) snapshot(e.target);
  }, true);
  document.addEventListener('click', function (e) {
    var el = e.target;
    if (el && el.matches && el.matches('input[type=checkbox], input[type=radio]')) {
      setTimeout(function () { snapshot(el); }, 0);
    }
    var btn = el && el.closest ? el.closest('button') : null;
    if (btn && btn.parentElement && btn.parentElement.className.indexOf('tab-container') >= 0 &&
        btn.parentElement.className.indexOf('visually-hidden') < 0) {
      restoreUntil = Date.now() + 2500;  // a real tab switch: re-apply user values briefly
    }
  }, true);
  setInterval(sync, 400);
})();
