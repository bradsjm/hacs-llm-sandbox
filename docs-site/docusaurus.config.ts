import type {Config} from '@docusaurus/types';
import type {Preset} from '@docusaurus/preset-classic';

const config: Config = {
  title: 'Assist Agent Sandbox',
  tagline: 'Home Assistant Assist tools for bounded LLM reasoning',
  url: 'https://bradsjm.github.io',
  baseUrl: '/hacs-llm-sandbox/',
  organizationName: 'bradsjm',
  projectName: 'hacs-llm-sandbox',
  trailingSlash: false,
  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'warn',
  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },
  presets: [
    [
      'classic',
      {
        docs: {
          routeBasePath: '/',
          sidebarPath: './sidebars.ts',
          editUrl: 'https://github.com/bradsjm/hacs-llm-sandbox/edit/main/docs-site/',
          showLastUpdateTime: true,
        },
        blog: false,
        theme: {
          customCss: './src/css/custom.css',
        },
      } satisfies Preset.Options,
    ],
  ],
  themeConfig: {
    navbar: {
      title: 'Assist Agent Sandbox',
      items: [
        {
          type: 'docSidebar',
          sidebarId: 'docsSidebar',
          position: 'left',
          label: 'Docs',
        },
        {
          href: 'https://github.com/bradsjm/hacs-llm-sandbox',
          label: 'GitHub',
          position: 'right',
        },
        {
          href: 'https://github.com/bradsjm/hacs-llm-sandbox/issues',
          label: 'Issues',
          position: 'right',
        },
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'Use',
          items: [
            {label: 'Install', to: '/installation/install-with-hacs'},
            {label: 'Quickstart', to: '/usage/quickstart'},
            {label: 'Troubleshooting', to: '/troubleshooting'},
          ],
        },
        {
          title: 'Reference',
          items: [
            {label: 'Tools', to: '/reference/tools/execute-home-code'},
            {label: 'Configuration', to: '/reference/configuration-options'},
            {label: 'Limits', to: '/reference/limits'},
          ],
        },
        {
          title: 'Project',
          items: [
            {label: 'GitHub', href: 'https://github.com/bradsjm/hacs-llm-sandbox'},
            {label: 'Changelog', href: 'https://github.com/bradsjm/hacs-llm-sandbox/blob/main/CHANGELOG.md'},
          ],
        },
      ],
      copyright: `Copyright ${new Date().getFullYear()} Assist Agent Sandbox contributors.`,
    },
    prism: {
      additionalLanguages: ['python', 'sql', 'yaml'],
    },
  } satisfies Preset.ThemeConfig,
};

export default config;
