// @ts-check
import { readFileSync } from 'node:fs';
import { defineConfig } from 'astro/config';
import react from '@astrojs/react';
import starlight from '@astrojs/starlight';

import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

const repository = 'https://github.com/VicHxuan/knowledge-tree';
const homepage = 'https://vichxuan.github.io/';

// 旧路由 → 新路由（V2 目录重组）：构建期生成 meta-refresh 兜底页
// 注意：Astro 不会给 redirects 的目标自动加 base，需在此手动补全
const SITE_BASE = '/knowledge-tree';
const rawRedirects = JSON.parse(readFileSync(new URL('./src/data/redirects.json', import.meta.url), 'utf8'));
const redirects = Object.fromEntries(
	Object.entries(rawRedirects).map(([from, to]) => [from, SITE_BASE + to]),
);

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
	redirects,

	integrations: [
		react(),
		starlight({
			title: 'VicHxuan Knowledge Tree',
			description: '传统规划控制、学习式规划与工程工具链的个人知识树',

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

			// V2：目录栏覆盖 —— 右侧目录上方叠加「前置知识 / 相关主题」知识关联区
			components: {
				TableOfContents: './src/components/TableOfContents.astro',
			},

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
						{ slug: 'learning-paths' },
						{ slug: 'knowledge-map' },
						{ label: '个人主页', link: homepage },
						{ label: 'GitHub 仓库', link: repository },
					],
				},
				docGroup('数学基础', 'mathematics'),
				docGroup('传统规划控制', 'classical-planning-control'),
				docGroup('学习式规划', 'learning-based-planning'),
				docGroup('工程工具链', 'engineering-tools'),
				docGroup('仿真与评估', 'simulation-evaluation'),
				docGroup('研究项目', 'projects'),
			],
		}),
	],

	markdown: {
		remarkPlugins: [remarkMath],
		rehypePlugins: [rehypeKatex],
	},
});
