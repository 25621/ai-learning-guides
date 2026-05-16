const {themes} = require('prism-react-renderer');

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'Reinforcement Learning: Beginner to Advanced',
  tagline: 'A structured guide to mastering reinforcement learning.',
  favicon: 'img/favicon.ico',

  url: 'https://25621.github.io',
  baseUrl: '/reinforcement-learning/',
  organizationName: '25621',
  projectName: 'reinforcement-learning',

  onBrokenLinks: 'warn',
  markdown: {
    format: 'md',
    hooks: {
      onBrokenMarkdownLinks: 'warn',
    },
  },
  trailingSlash: true,

  i18n: {
    defaultLocale: 'en',
    locales: ['en', 'ko', 'es', 'hi'],
  },

  themes: [
    [
      require.resolve("@easyops-cn/docusaurus-search-local"),
      {
        hashed: true,
        indexDocs: true,
        indexBlog: false,
        indexPages: false,
      },
    ],
  ],

  presets: [
    [
      'classic',
      {
        docs: {
          path: '..',
          routeBasePath: '/',
          sidebarPath: require.resolve('./sidebars.js'),
          editUrl: ({locale, docPath}) => {
            if (locale !== 'en') {
              return `https://github.com/25621/reinforcement-learning/edit/main/docusaurus-site/i18n/${locale}/docusaurus-plugin-content-docs/current/${docPath}`;
            }
            return `https://github.com/25621/reinforcement-learning/edit/main/${docPath}`;
          },
          include: [
            'README.md',
            'foundations/**/*.md',
            'tabular_methods/**/*.md',
            'function_approximation/**/*.md',
            'policy_gradients/**/*.md',
            'advanced_topics/**/*.md',
          ],
          exclude: [
            '**/_site/**',
            '**/mdbook-site/**',
            '**/docusaurus-site/**',
            '**/node_modules/**',
            '**/vendor/**',
          ],
        },
        blog: false,
        theme: {
          customCss: require.resolve('./src/css/custom.css'),
        },
        gtag: {
          trackingID: 'G-VLNEEY6SWG',
          anonymizeIP: true,
        },
      },
    ],
  ],

  themeConfig: {
    metadata: [
      {
        name: 'description',
        content:
          'A practical reinforcement learning curriculum with runnable Python implementations and explanations.',
      },
    ],
    colorMode: {
      defaultMode: 'light',
      respectPrefersColorScheme: true,
    },
    navbar: {
      title: 'Reinforcement Learning',
      logo: {
        alt: 'Docusaurus Logo',
        src: 'img/logo.svg',
      },
      items: [
        {
          type: 'localeDropdown',
          position: 'right',
        },
        {
          href: 'https://github.com/25621/reinforcement-learning',
          label: 'GitHub',
          position: 'right',
        },
      ],
    },
    footer: {
      style: 'dark',
      links: [],
      copyright: `Copyright © ${new Date().getFullYear()} 25621. All rights reserved. Built with Docusaurus.`,
    },
    prism: {
      theme: themes.github,
      darkTheme: themes.dracula,
      additionalLanguages: ['python'],
    },
  },
};

module.exports = config;
