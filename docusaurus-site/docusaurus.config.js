const {themes} = require('prism-react-renderer');

const siteUrl = 'https://25621.github.io';
const siteBaseUrl = '/ai-learning-guides/';
const siteDescription =
  'Learn reinforcement learning from scratch with a structured roadmap covering MDPs, dynamic programming, Q-learning, policy gradients, actor-critic methods, PPO, SAC, and modern deep RL.';
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
  title: 'AI Learning Guides',
  tagline:
    'A free, open-source roadmap and tutorial for mastering reinforcement learning — from fundamentals to research-level deep RL.',
  favicon: 'img/favicon.ico',

  url: siteUrl,
  baseUrl: siteBaseUrl,
  organizationName: '25621',
  projectName: 'ai-learning-guides',

  onBrokenLinks: 'warn',
  markdown: {
    format: 'md',
    hooks: {
      onBrokenMarkdownLinks: 'warn',
    },
  },
  trailingSlash: true,

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
        name: 'AI Learning Guides',
        url: `${siteUrl}${siteBaseUrl}`,
        description: siteDescription,
        inLanguage: 'en',
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
        name: 'AI Learning Guides',
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
        inLanguage: 'en',
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
          editUrl: ({docPath}) =>
            `https://github.com/25621/ai-learning-guides/edit/main/${docPath}`,
          include: [
            'README.md',
            'guides/*/README.md',
            'guides/*/projects/*/README.md',
            'shared/glossary.md',
          ],
          exclude: [
            '**/_site/**',
            '**/mdbook-site/**',
            '**/docusaurus-site/**',
            '**/node_modules/**',
            '**/vendor/**',
          ],
          sidebarItemsGenerator: async function ({
            defaultSidebarItemsGenerator,
            ...args
          }) {
            const sidebarItems = await defaultSidebarItemsGenerator(args);
            const item = args.item;
            const docs = args.docs;
            if (item.customProps && item.customProps.phase) {
              const phase = item.customProps.phase;
              return sidebarItems.filter(sidebarItem => {
                if (sidebarItem.type === 'doc') {
                  const doc = docs.find(d => d.id === sidebarItem.id);
                  if (doc) {
                    const match = doc.source.match(/guides\/ai-hardware\/projects\/(\d+)-/);
                    if (match) {
                      const projectNum = parseInt(match[1], 10);
                      if (phase === 1) {
                        return projectNum >= 1 && projectNum <= 5;
                      } else if (phase === 2) {
                        return projectNum >= 6 && projectNum <= 10;
                      }
                    }
                  }
                }
                return true;
              });
            }
            return sidebarItems;
          },
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
        ...(process.env.NODE_ENV === 'production' ? {
          gtag: {
            trackingID: 'G-LEDRXS502H',
            anonymizeIP: true,
          },
        } : {}),
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
      {property: 'og:site_name', content: 'AI Learning Guides'},
      {property: 'og:title', content: 'AI Learning Guides'},
      {property: 'og:description', content: siteDescription},
      {property: 'og:image', content: `${siteUrl}${siteBaseUrl}img/social-card.svg`},
      {property: 'og:image:alt', content: 'AI Learning Guides — open-source learning roadmap'},
      {property: 'og:url', content: `${siteUrl}${siteBaseUrl}`},
      {property: 'og:locale', content: 'en_US'},

      {name: 'twitter:card', content: 'summary_large_image'},
      {name: 'twitter:title', content: 'AI Learning Guides'},
      {name: 'twitter:description', content: siteDescription},
      {name: 'twitter:image', content: `${siteUrl}${siteBaseUrl}img/social-card.svg`},
      {name: 'twitter:image:alt', content: 'AI Learning Guides — open-source learning roadmap'},
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
      title: 'AI Learning Guides',
      logo: {
        alt: 'AI Learning Guides logo',
        src: 'img/logo.svg',
        width: 32,
        height: 32,
      },
      hideOnScroll: true,
      items: [
        {
          href: 'https://github.com/25621/ai-learning-guides',
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
          ],
        },
        {
          title: 'Community',
          items: [
            {
              label: 'GitHub',
              href: 'https://github.com/25621/ai-learning-guides',
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
              label: 'David Silver Lectures',
              href: 'https://www.youtube.com/watch?v=2pWv7GOvuf0&list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ',
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
