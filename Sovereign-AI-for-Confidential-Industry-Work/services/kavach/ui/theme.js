/* Kavach theme and view initializer.
   Loaded synchronously in <head> to eliminate any flash of unstyled content (FOUC),
   black/white theme flicker, or tab flash on page refresh.
*/
(function () {
  "use strict";
  try {
    var stored = localStorage.getItem("kavach-theme");

    // ?theme=light deep-links a theme, the same way #sovereignty below
    // deep-links a view. A projector, a kiosk shortcut or a screenshot needs
    // the light theme WITHOUT someone finding the toggle first, and the
    // choice is remembered so the rest of the session stays in it.
    var asked = (location.search.match(/[?&]theme=(light|dark)/) || [])[1];
    if (asked) localStorage.setItem("kavach-theme", asked);

    var theme = (asked || stored) === "light" ? "light" : "dark";
    document.documentElement.setAttribute("data-theme", theme);

    // If the URL has a hash matching a view, activate it instantly to prevent
    // the Overview panel from flashing before the target panel loads.
    var hash = (location.hash || "").replace(/^#/, "");
    var valid = ["overview", "runs", "corpus", "engineering", "sovereignty", "router"];
    if (hash && valid.indexOf(hash) !== -1) {
      document.documentElement.setAttribute("data-initial-view", hash);
    }
  } catch (e) {
    document.documentElement.setAttribute("data-theme", "dark");
  }
})();
