// @ts-check
import {themes as prismThemes} from 'prism-react-renderer';

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'Physical AI & Humanoid Robotics',
  tagline: 'Comprehensive Educational Guide from ROS 2 Foundations to Advanced Embodied AI',
  favicon: 'img/favicon.ico',

  // Set the production url of your site here
  url: 'https://your-username.github.io',
  // Set the /<baseUrl>/ pathname under which your site is served
  baseUrl: '/humanoid-robotics/',

  // GitHub Pages deployment config
  organizationName: 'your-username', // Usually your GitHub org/user name
  projectName: 'humanoid-robotics', // Usually your repo name

  onBrokenLinks: 'warn',
  markdown: {
    hooks: {
      onBrokenMarkdownLinks: 'warn',
    },
  },

  // Even if you don't use internalization, you can use this field to set useful
  // metadata like html lang. For example, if your site is Chinese, you may want
  // to replace "en" with "zh-Hans".
  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  
  presets: [
    [
      'classic',
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          sidebarPath: require.resolve('./sidebars.js'),
          // Please change this to your repo.
          editUrl: 'https://github.com/your-username/humanoid-robotics/tree/main/',
        },
        blog: false, // Disable blog for book-focused site
        theme: {
          customCss: require.resolve('./src/css/custom.css'),
        },
      }),
    ],
  ],

  themeConfig: /** @type {import('@docusaurus/preset-classic').ThemeConfig} */ ({
    // Replace with your project's social card
    image: 'img/docusaurus-social-card.jpg',
    navbar: {
      title: 'Physical AI & Humanoid Robotics',
      logo: {
        alt: 'Robot Logo',
        src: 'img/logo.svg',
      },
      items: [
        {
          type: 'docSidebar',
          sidebarId: 'tutorialSidebar',
          position: 'left',
          label: 'Book Contents',
        },
        {
          to: '/docs/hardware/',
          label: 'Hardware Guide',
          position: 'left'
        },
        {
          to: '/docs/resources/references',
          label: 'Resources',
          position: 'left'
        },
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'Book Contents',
          items: [
            {
              label: 'Quarter 1: ROS 2 Foundations',
              to: '/docs/quarter-1/01-robotics-overview',
            },
            {
              label: 'Quarter 2: Simulation',
              to: '/docs/quarter-2/06-physics-simulation',
            },
            {
              label: 'Quarter 3: AI Integration',
              to: '/docs/quarter-3/11-computer-vision',
            },
            {
              label: 'Quarter 4: Advanced Embodied AI',
              to: '/docs/quarter-4/16-multimodal-ai',
            },
          ],
        },
        {
          title: 'Resources',
          items: [
            {
              label: 'Hardware Requirements',
              to: '/docs/hardware/minimum-requirements',
            },
            {
              label: 'References',
              to: '/docs/resources/references',
            },
            {
              label: 'Glossary',
              to: '/docs/resources/glossary',
            },
          ],
        },
        {
          title: 'More',
          items: [
            {
              label: 'GitHub',
              href: 'https://github.com/your-username/humanoid-robotics',
            },
            {
              label: 'Report Issue',
              href: 'https://github.com/your-username/humanoid-robotics/issues',
            },
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} Physical AI & Humanoid Robotics. Built with Docusaurus.`,
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
      additionalLanguages: ['python', 'bash', 'json', 'yaml'],
    },
    // Context7 MCP integration placeholder
    // This will be enhanced with dynamic documentation fetching
    docs: {
      sidebar: {
        autoCollapseCategories: false,
      },
    },
  }),

  // Context7 MCP integration configuration
  // This section will be enhanced with actual MCP server configuration
  customFields: {
    mcpConfig: {
      serverUrl: process.env.CONTEXT7_MCP_SERVER || 'http://localhost:3001',
      libraries: [
        'ros2',
        'gazebo',
        'isaac-sim',
        'opencv',
        'pytorch',
        'tensorflow',
        'unity-robotics'
      ]
    }
  },

  // Enable enhanced search functionality
  themes: [
    [
      '@easyops-cn/docusaurus-search-local',
      {
        hashed: true,
        language: ['en'],
        highlightSearchTermsOnTargetPage: true,
        explicitSearchResultPath: true,
        indexDocs: true,
        indexBlog: false,
        indexPages: false,
        docsRouteBasePath: '/',
        searchResultLimits: 8,
        searchResultContextMaxLength: 50,
        docsPluginIdForPreferredVersion: 'default',
      },
    ],
  ],
};

export default config;