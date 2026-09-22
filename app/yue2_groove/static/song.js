// Clicking a FAMILY row sets the current work through the same hidden bridge
// the handoffs use, so the band, the stage and the actions follow the clicked
// work.
(function () {
  function bridge() {
    var wrap = document.getElementById('bb-current-work');
    return wrap ? wrap.querySelector('textarea') : null;
  }
  function setNative(el, value) {
    var setter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, 'value').set;
    setter.call(el, value);
    el.dispatchEvent(new Event('input', { bubbles: true }));
    el.dispatchEvent(new Event('change', { bubbles: true }));
  }
  document.addEventListener('click', function (event) {
    var hit = event.target && event.target.closest ? event.target.closest('.bb-family-hit') : null;
    if (!hit) return;
    var run = hit.getAttribute('data-bb-run');
    var el = bridge();
    if (!run || !el) return;
    event.preventDefault();
    var active = document.querySelectorAll('.bb-family-hit.bb-family-active');
    for (var i = 0; i < active.length; i++) active[i].classList.remove('bb-family-active');
    hit.classList.add('bb-family-active');
    setNative(el, run);
  }, true);
})();
