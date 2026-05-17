# Contributor notes

## Heading anchors

Heading anchor pins (`{#kebab-id}`) live in the English source markdown files
under `./**/*.md`, never inside `docusaurus-site/i18n/`. Docusaurus propagates
the pinned ID from the English source to every locale, so pinning in a
translation file silently diverges that locale's URL from the rest.

Keep pin IDs equal to the slug `github-slugger` would derive from the English
heading text (Docusaurus' default slugifier). That way the pin only locks
behavior against future heading rewording, without changing the public URL.
