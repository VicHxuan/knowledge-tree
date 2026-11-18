// @ts-check
import { readFileSync } from 'node:fs';
import { defineConfig } from 'astro/config';
import react from '@astrojs/react';
import starlight from '@astrojs/starlight';

import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

import { DOMAINS } from './src/data/domains.mjs';
import rehypeMermaid from './scripts/rehype-mermaid.mjs';
import rehypeArticleTools from './scripts/rehype-article-tools.mjs';
import rehypeHeadingAliases from './scripts/rehype-heading-aliases.mjs';
import rehypePaperLinks from './scripts/rehype-paper-links.mjs';
import preserveRedirectFragments from './scripts/preserve-redirect-fragments.mjs';

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
		preserveRedirectFragments(rawRedirects, SITE_BASE),
		starlight({
			title: 'VicHxuan Knowledge Tree',
			routeMiddleware: './src/routeData.ts',
			description: '传统规划控制、学习式规划与工程工具链的个人知识树',

			// 单语言站点：界面使用简体中文，URL 不带语言前缀
			defaultLocale: 'root',
			locales: {
				root: { label: '简体中文', lang: 'zh-CN' },
			},

			customCss: [
				'katex/dist/katex.min.css',
				'./src/styles/fonts.css',
				'./src/styles/tokens.css',
				'./src/styles/custom.css',
			],

			favicon: 'favicon.svg',
			head: ['noto-serif-sc-headings.woff2', 'source-serif-4-latin.woff2'].map((file) => ({
				tag: 'link',
				attrs: { rel: 'preload', as: 'font', type: 'font/woff2', crossorigin: 'anonymous', href: `${SITE_BASE}/fonts/${file}` },
			})),

			// 右侧先展示页内目录，再提供前置知识与相关主题。
			// V2：页脚覆盖 —— 保留默认页脚，追加 Mermaid 懒渲染 / 书签 / 反馈 / 最近阅读脚本
			components: {
				SiteTitle: './src/components/SiteTitle.astro',
				Hero: './src/components/HomeHero.astro',
				TableOfContents: './src/components/TableOfContents.astro',
				Footer: './src/components/Footer.astro',
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
						{ slug: 'papers' },
						{ slug: 'bookmarks' },
						{ label: '个人主页', link: homepage },
						{ label: 'GitHub 仓库', link: repository },
					],
				},
				// 六大内容域由 src/data/domains.mjs 派生（单一事实源）
				...DOMAINS.map((d) => docGroup(d.label, d.id)),
				docGroup('研究项目', 'projects'),
			],
		}),
	],

	markdown: {
		remarkPlugins: [remarkMath],
		rehypePlugins: [rehypeKatex, rehypeMermaid, rehypeArticleTools, rehypeHeadingAliases, rehypePaperLinks],
		syntaxHighlight: { type: 'shiki', excludeLangs: ['mermaid'] },
	},
});
