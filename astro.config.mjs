// @ts-check
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';

import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

const repository = 'https://github.com/VicHxuan/knowledge-tree';
const homepage = 'https://vichxuan.github.io/';

// 顶级知识分组：文件夹即分类，页面由 autogenerate 自动生成
// 注意：Starlight v0.39+ 要求 label 与 autogenerate 通过 items 数组组合
const docGroup = (label, directory) => ({
	label,
	collapsed: true,
	items: [{ autogenerate: { directory, collapsed: true } }],
});

// https://astro.build/config
export default defineConfig({
	// GitHub Pages 项目站点：用户域名 + 仓库名子路径
	site: 'https://vichxuan.github.io',
	base: '/knowledge-tree',

	integrations: [
		starlight({
			title: 'VicHxuan Knowledge Tree',
			description: '传统规划控制、强化学习端到端与工程工具链的个人知识树',

			// 单语言站点：界面使用简体中文，URL 不带语言前缀
			defaultLocale: 'root',
			locales: {
				root: { label: '简体中文', lang: 'zh-CN' },
			},

			customCss: [
				'katex/dist/katex.min.css',
				'./src/styles/custom.css',
			],

			favicon: 'favicon.svg',

			social: [
				{ icon: 'github', label: 'GitHub', href: repository },
			],

			editLink: {
				baseUrl: `${repository}/edit/main/`,
			},

			sidebar: [
				{
					label: '导航',
					items: [
						{ slug: '' },
						{ label: '个人主页', link: homepage },
						{ label: 'GitHub 仓库', link: repository },
					],
				},
				docGroup('数学基础', 'mathematics'),
				docGroup('传统规划控制', 'classical-planning-control'),
				docGroup('强化学习端到端', 'rl-end-to-end'),
				docGroup('工程工具链', 'engineering-tools'),
				docGroup('学习路径', 'learning-paths'),
			],
		}),
	],

	markdown: {
		remarkPlugins: [remarkMath],
		rehypePlugins: [rehypeKatex],
	},
});
