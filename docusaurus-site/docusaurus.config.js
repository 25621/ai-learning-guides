const {themes} = require('prism-react-renderer');

const siteUrl = 'https://25621.github.io';
const siteBaseUrl = '/reinforcement-learning/';
const siteDescription =
  'Learn reinforcement learning from scratch with a structured 6-month roadmap covering MDPs, dynamic programming, Q-learning, policy gradients, actor-critic methods, PPO, SAC, and modern deep RL — with hands-on Python exercises and curated resources.';
const siteKeywords = [
  'reinforcement learning',
  'reinforcement learning tutorial',
  'deep reinforcement learning',
  'RL guide',
  'machine learning',
  'Markov decision process',
  'Q-learning',
  'policy gradient',
  'actor critic',
  'PPO',
  'SAC',
  'DQN',
  'Sutton and Barto',
  'OpenAI Spinning Up',
  'AI learning roadmap',
];

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'Reinforcement Learning: Beginner to Advanced',
  tagline:
    'A free, open-source roadmap and tutorial for mastering reinforcement learning — from fundamentals to research-level deep RL.',
  favicon: 'img/favicon.ico',

  url: siteUrl,
  baseUrl: siteBaseUrl,
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
    locales: ['en', 'es', 'fr', 'hi', 'ja', 'ko'],
  },

  headTags: [
    {
      tagName: 'link',
      attributes: {
        rel: 'preconnect',
        href: 'https://www.googletagmanager.com',
      },
    },
    {
      tagName: 'link',
      attributes: {
        rel: 'preconnect',
        href: 'https://www.google-analytics.com',
      },
    },
    {
      tagName: 'script',
      attributes: {
        type: 'application/ld+json',
      },
      innerHTML: JSON.stringify({
        '@context': 'https://schema.org',
        '@type': 'WebSite',
        name: 'Reinforcement Learning: Beginner to Advanced',
        url: `${siteUrl}${siteBaseUrl}`,
        description: siteDescription,
        inLanguage: ['en', 'es', 'fr', 'hi', 'ja', 'ko'],
        potentialAction: {
          '@type': 'SearchAction',
          target: `${siteUrl}${siteBaseUrl}search?q={search_term_string}`,
          'query-input': 'required name=search_term_string',
        },
      }),
    },
    {
      tagName: 'script',
      attributes: {
        type: 'application/ld+json',
      },
      innerHTML: JSON.stringify({
        '@context': 'https://schema.org',
        '@type': 'Course',
        name: 'Reinforcement Learning: Beginner to Advanced',
        description: siteDescription,
        url: `${siteUrl}${siteBaseUrl}`,
        provider: {
          '@type': 'Organization',
          name: 'Reinforcement Learning Project Contributors',
          url: `${siteUrl}${siteBaseUrl}`,
        },
        educationalLevel: 'Beginner to Advanced',
        learningResourceType: 'Tutorial',
        teaches: siteKeywords.join(', '),
        isAccessibleForFree: true,
        inLanguage: ['en', 'es', 'fr', 'hi', 'ja', 'ko'],
      }),
    },
  ],

  themes: [
    [
      require.resolve("@easyops-cn/docusaurus-search-local"),
      {
        hashed: true,
        indexDocs: true,
        indexBlog: false,
        indexPages: false,
        docsRouteBasePath: '/',
        highlightSearchTermsOnTargetPage: true,
        explicitSearchResultPath: true,
        searchResultLimits: 8,
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
          breadcrumbs: true,
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
        sitemap: {
          changefreq: 'weekly',
          priority: 0.7,
          ignorePatterns: ['/tags/**', '/search/**'],
          filename: 'sitemap.xml',
        },
        gtag: {
          trackingID: 'G-VLNEEY6SWG',
          anonymizeIP: true,
        },
      },
    ],
  ],

  themeConfig: {
    image: 'img/social-card.svg',
    metadata: [
      {name: 'description', content: siteDescription},
      {name: 'keywords', content: siteKeywords.join(', ')},
      {name: 'author', content: 'Reinforcement Learning Project Contributors'},
      {name: 'robots', content: 'index, follow, max-image-preview:large, max-snippet:-1, max-video-preview:-1'},
      {name: 'googlebot', content: 'index, follow'},
      {name: 'theme-color', content: '#2e8555'},
      {name: 'application-name', content: 'Reinforcement Learning Guide'},
      {name: 'apple-mobile-web-app-title', content: 'RL Guide'},
      {name: 'apple-mobile-web-app-capable', content: 'yes'},
      {name: 'mobile-web-app-capable', content: 'yes'},
      {name: 'format-detection', content: 'telephone=no'},

      {property: 'og:type', content: 'website'},
      {property: 'og:site_name', content: 'Reinforcement Learning: Beginner to Advanced'},
      {property: 'og:title', content: 'Reinforcement Learning: Beginner to Advanced'},
      {property: 'og:description', content: siteDescription},
      {property: 'og:image', content: `${siteUrl}${siteBaseUrl}img/social-card.svg`},
      {property: 'og:image:alt', content: 'Reinforcement Learning: Beginner to Advanced — open-source learning roadmap'},
      {property: 'og:url', content: `${siteUrl}${siteBaseUrl}`},
      {property: 'og:locale', content: 'en_US'},
      {property: 'og:locale:alternate', content: 'es_ES'},
      {property: 'og:locale:alternate', content: 'fr_FR'},
      {property: 'og:locale:alternate', content: 'hi_IN'},
      {property: 'og:locale:alternate', content: 'ja_JP'},
      {property: 'og:locale:alternate', content: 'ko_KR'},

      {name: 'twitter:card', content: 'summary_large_image'},
      {name: 'twitter:title', content: 'Reinforcement Learning: Beginner to Advanced'},
      {name: 'twitter:description', content: siteDescription},
      {name: 'twitter:image', content: `${siteUrl}${siteBaseUrl}img/social-card.svg`},
      {name: 'twitter:image:alt', content: 'Reinforcement Learning: Beginner to Advanced — open-source learning roadmap'},
    ],
    colorMode: {
      defaultMode: 'light',
      respectPrefersColorScheme: true,
    },
    docs: {
      sidebar: {
        hideable: true,
        autoCollapseCategories: true,
      },
    },
    navbar: {
      title: 'Reinforcement Learning',
      logo: {
        alt: 'Reinforcement Learning: Beginner to Advanced logo',
        src: 'img/logo.svg',
        width: 32,
        height: 32,
      },
      hideOnScroll: true,
      items: [
        {
          type: 'localeDropdown',
          position: 'right',
        },
        {
          href: 'https://github.com/25621/reinforcement-learning',
          label: 'GitHub',
          position: 'right',
          'aria-label': 'GitHub repository',
        },
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'Learn',
          items: [
            {label: 'Overview', to: '/'},
            {label: 'Foundations', to: '/foundations/multi_armed_bandit_explained/'},
            {label: 'Tabular Methods', to: '/tabular_methods/q_learning_frozen_lake_explained/'},
            {label: 'Policy Gradients', to: '/policy_gradients/ppo_scratch_explained/'},
          ],
        },
        {
          title: 'Community',
          items: [
            {
              label: 'GitHub',
              href: 'https://github.com/25621/reinforcement-learning',
            },
            {
              label: 'Issues',
              href: 'https://github.com/25621/reinforcement-learning/issues',
            },
            {
              label: 'Discussions',
              href: 'https://github.com/25621/reinforcement-learning/discussions',
            },
          ],
        },
        {
          title: 'Resources',
          items: [
            {
              label: 'Sutton & Barto',
              href: 'http://incompleteideas.net/book/the-book.html',
            },
            {
              label: 'OpenAI Spinning Up',
              href: 'https://spinningup.openai.com/',
            },
            {
              label: 'David Silver Lectures',
              href: 'https://www.youtube.com/watch?v=2pWv7GOvuf0',
            },
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} Reinforcement Learning Project Contributors. Built with Docusaurus.`,
    },
    prism: {
      theme: themes.github,
      darkTheme: themes.dracula,
      additionalLanguages: ['python'],
    },
  },
};

module.exports = config;
