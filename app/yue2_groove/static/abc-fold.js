// Collapsible ABC source: the component stays in the DOM (SCORE VIEW reads
// it), only its input area folds behind a chevron next to the label.
(function () {
  function init() {
    var box = document.getElementById('bb-abc-source');
    if (!box || box.getAttribute('data-bb-fold') === '1') return;
    var info = box.querySelector('span[data-testid="block-info"]') || box.querySelector('.block-title');
    if (!info) return;
    box.setAttribute('data-bb-fold', '1');
    var chev = document.createElement('span');
    chev.className = 'bb-fold';
    chev.setAttribute('role', 'button');
    chev.setAttribute('tabindex', '0');
    chev.setAttribute('aria-label', 'Collapse or expand the ABC source');
    chev.textContent = '+';
    info.appendChild(chev);
    function toggle(e) {
      if (e) { e.preventDefault(); e.stopPropagation(); }
      var folded = box.classList.toggle('bb-folded');
      chev.textContent = folded ? '+' : '-';
      var ta = box.querySelector('textarea');
      if (ta && !folded) {
        // re-measure: Gradio may have sized it while the box was folded/hidden
        ta.style.height = 'auto';
        var target = Math.min(Math.max(ta.scrollHeight + 4, 120), 420);
        ta.style.height = target + 'px';
        ta.style.overflowY = ta.scrollHeight > target ? 'auto' : 'hidden';
      }
    }
    chev.addEventListener('click', toggle, true);
    chev.addEventListener('keydown', function (e) {
      if (e.key === 'Enter' || e.key === ' ') toggle(e);
    });
    toggle();  // start folded: the rendered score below is the main view
  }
  setInterval(init, 600);
})();
