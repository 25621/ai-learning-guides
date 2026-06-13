/**
 * Thin wrapper around `@docusaurus/plugin-content-docs` that narrows the set of
 * paths watched by the dev server (`docusaurus start`).
 *
 * Why this exists:
 * The docs plugin is configured with `path: '..'`, so its content root is the
 * repository root. The stock plugin's `getPathsToWatch()` returns, among the
 * bounded `include` globs, an unbounded `<contentPath>/**\/_category_*` glob.
 * With the content root at the repo root, chokidar must recursively watch the
 * ENTIRE repository to satisfy that glob — including `venv/` (thousands of
 * Python `__pycache__` directories), `node_modules/`, and `tmp/`. On Linux,
 * chokidar creates one inotify instance per directory, so this quickly exceeds
 * `fs.inotify.max_user_instances` and the dev server crashes with
 * `ENOSPC: System limit for number of file watchers reached`.
 *
 * This site uses a fully manual `sidebars.js` and contains no `_category_*`
 * files, so the recursive category-metadata watch is pure overhead. We drop any
 * watch entry containing a `**` segment, which removes the repo-root recursive
 * watch while preserving the bounded, `include`-scoped watches that power doc
 * hot-reload (README.md, guides/*, shared/glossary.md, and the sidebar file).
 */
const docsPlugin = require('@docusaurus/plugin-content-docs');

async function docsPluginWatchScoped(context, options) {
  const instance = await docsPlugin.default(context, options);
  return {
    ...instance,
    getPathsToWatch() {
      return instance.getPathsToWatch().filter((p) => !String(p).includes('**'));
    },
  };
}

module.exports = docsPluginWatchScoped;
module.exports.default = docsPluginWatchScoped;
module.exports.validateOptions = docsPlugin.validateOptions;
