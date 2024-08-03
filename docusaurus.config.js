// @ts-check
// Note: type annotations allow type checking and IDEs autocompletion

const lightCodeTheme = require('prism-react-renderer/themes/github');
const darkCodeTheme = require('prism-react-renderer/themes/dracula');
const math = require('remark-math');
const katex = require('rehype-katex');

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'Notebook',
  tagline: '这里是Sonder的宝藏笔记本',
  url: 'https://space.keter.top',
  baseUrl: '/',
  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'warn',
  favicon: 'img/favicon.ico',
  organizationName: 'facebook', // Usually your GitHub org/user name.
  projectName: 'Jupiter-king.github.io', // Usually your repo name.

  presets: [
    [
      'classic',
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          sidebarPath: require.resolve('./sidebars.js'),
          // Please change this to your repo.
          editUrl: 'https://github.com/facebook/docusaurus/tree/main/packages/create-docusaurus/templates/shared/',
          remarkPlugins: [math],
          rehypePlugins: [katex],
        },
        blog: {
          showReadingTime: true,
          // Please change this to your repo.
          editUrl:
            'https://github.com/facebook/docusaurus/tree/main/packages/create-docusaurus/templates/shared/',
        },
        theme: {
          customCss: require.resolve('./src/css/custom.css'),
        },
      }),
    ],
  ],

  plugins: [
    [
      require.resolve("@easyops-cn/docusaurus-search-local"),
      {
        // ... Your options.
        // `hashed` is recommended as long-term-cache of index file is possible.
        hashed: true,
        // For Docs using Chinese, The `language` is recommended to set to:
        // ```
        language: ["en", "zh"],
        // ```
        // When applying `zh` in language, please install `nodejieba` in your project.
        translations: {
          search_placeholder: "Search",
          see_all_results: "See all results",
          no_results: "No results.",
          search_results_for: 'Search results for "{{ keyword }}"',
          search_the_documentation: "Search the documentation",
          count_documents_found: "{{ count }} document found",
          count_documents_found_plural: "{{ count }} documents found",
          no_documents_were_found: "No documents were found",
        },
      },
    ],
  ],

  themeConfig:
        /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
        ({
            metadata: [
                { name: 'keywords', content: 'notebook,Sonder,Algorithm,题解,算法,编程,学习笔记,深度学习,模型鲁棒性' },
                { name: 'google-site-verification', content: 'uxeTLCCDJzymPOdZrqyyHyxwESUlDjHAnBg-hXaq9-o' },
                { name: 'baidu-site-verification', content: 'codeva-Aw5EMWD75n' }
            ],
            navbar: {
                title: 'Sonder的笔记本',
                logo: {
                    alt: 'My Site Logo',
                    src: 'img/logo.svg',
                },
                items: [
                    {
                        to: "/docs/deep_learning/基础知识/深度学习基础知识目录",
                        activeBasePath: '/docs/deep_learning',
                        label: "Deep Learning",
                        position: "left",
                    },
                    {
                        to: "/docs/high_performance/CUDA编程/异构计算与CUDA",
                        activeBasePath: '/docs/high_performance',
                        label: "HPC&DS",
                        position: "left",
                    },
                    {
                        to: "/docs/algorithm/基础算法",
                        activeBasePath: '/docs/algorithm',
                        label: "Algorithm",
                        position: "left"
                    },
                    {
                      to: "/docs/cpp/类的声明和定义",
                      activeBasePath: '/docs/cpp',
                      label: "C++",
                      position: "left"
                    },
                    {
                        to: "/docs/others/环境保护大使/配置detectron2环境",
                        activeBasePath: '/docs/others',
                        label: "Others",
                        position: "left"
                    },
                    {
                      href: '/About',
                      label: 'About',
                      position: 'right',
                    },
                    {
                        href: 'https://github.com/coronaPolvo/Jupiter-king.github.io',
                        label: 'GitHub',
                        position: 'right',
                    },
                ],
            },
            footer: {
                style: 'light',
                copyright: `© ${new Date().getFullYear()} Sonder`,
            }, 
            prism: {
                theme: lightCodeTheme,
                darkTheme: darkCodeTheme,
            },
        }),
    stylesheets: [
      {
        href: 'https://cdn.jsdelivr.net/npm/katex@0.13.24/dist/katex.min.css',
        type: 'text/css',
        integrity:
          'sha384-odtC+0UGzzFL/6PNoE8rX/SPcQDXBJ+uRepguP4QkPCm2LBxH3FA3y+fKSiJ+AmM',
        crossorigin: 'anonymous',
      },
    ],
};

module.exports = config;
