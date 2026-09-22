// SONG / STUDIO view switch.  The two roots are always mounted; the view is a
// class on <html> so a switch never remounts the Studio Blocks (see
// docs/VIEW_SWITCH_PREFLIGHT.md).  The inline boot script in the <head> sets
// window.__BB_VIEW_MODE__ / __BB_VIEW_FORCED__ first and applies the class
// before the body paints; this script wires the buttons, the hidden bridge and
// the settings-rail toggle (disabled in SONG).
(function () {
  var KEY = 'bb-view';
  function area() {
    var box = document.getElementById('bb-view');
    return box ? box.querySelector('textarea') : null;
  }
  function norm(view) { return view === 'studio' ? 'studio' : 'song'; }
  function setNative(el, value) {
    var setter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, 'value').set;
    setter.call(el, value);
    el.dispatchEvent(new Event('input', { bubbles: true }));
    el.dispatchEvent(new Event('change', { bubbles: true }));
  }
  function button(id) {
    var wrap = document.getElementById(id);
    return wrap ? (wrap.tagName === 'BUTTON' ? wrap : wrap.querySelector('button')) : null;
  }
  function applyRail(view) {
    // the settings rail sits inside the STUDIO root, so the toggle is a dead
    // control in SONG; disable it there and label it honestly
    var rail = button('bb-rail-btn');
    if (!rail) return;
    var songView = norm(view) !== 'studio';
    var hidden = document.documentElement.classList.contains('bb-rail-hidden');
    rail.disabled = songView;
    rail.classList.toggle('bb-on', !songView && !hidden);
    var label = songView ? 'Settings live in STUDIO'
              : (hidden ? 'Show the settings rail' : 'Hide the settings rail');
    rail.title = label;
    rail.setAttribute('aria-label', label);
  }
  var currentView = 'song';
  window.__bbApplyRail = function () { applyRail(currentView); };
  function apply(view) {
    view = norm(view);
    currentView = view;
    var root = document.documentElement;
    root.classList.toggle('bb-view-song', view !== 'studio');
    root.classList.toggle('bb-view-studio', view === 'studio');
    var song = button('bb-view-song-btn');
    var studio = button('bb-view-studio-btn');
    if (song) song.classList.toggle('bb-active', view !== 'studio');
    if (studio) studio.classList.toggle('bb-active', view === 'studio');
    applyRail(view);
  }
  window.__bbSetView = function (view) {
    view = norm(view);
    try { localStorage.setItem(KEY, view); } catch (e) {}
    apply(view);
    var el = area();
    if (el && el.value !== view) setNative(el, view);
  };
  var booted = false;
  function boot() {
    if (booted) return;
    var el = area();
    if (!el) return;
    booted = true;
    var mode = window.__BB_VIEW_MODE__ || 'auto';
    var view;
    if (mode === 'song' || mode === 'studio') {
      view = mode;
    } else {
      var stored = '';
      try { stored = localStorage.getItem(KEY) || ''; } catch (e) {}
      view = stored === 'studio' ? 'studio' : 'song';
    }
    apply(view);
    if (el.value !== view) setNative(el, view);
  }
  setInterval(function () {
    boot();
    var el = area();
    if (!el || !el.value) return;
    // a forced launch (--view / --tab) must not clobber the last stored choice;
    // only a real click through __bbSetView remembers a new one
    if (!window.__BB_VIEW_FORCED__) {
      try { localStorage.setItem(KEY, el.value); } catch (e) {}
    }
    apply(el.value);
  }, 400);
})();
