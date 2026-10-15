/**
 * rehype-mermaid.mjs — 把 ```mermaid 代码块转换为客户端按需渲染的占位元素。
 *
 * 渲染发生在浏览器端：src/components/Footer.astro 捆绑的脚本只在本页存在
 * .kt-mermaid 图时才动态 import('mermaid')（Vite 会把 mermaid 拆成独立
 * chunk——没有图表的页面永远不会下载它）。
 *
 * 本插件把
 *   <pre><code class="language-mermaid">…源码…</code></pre>
 * 替换为
 *   <figure class="kt-mermaid">
 *     <div class="kt-mermaid-src" hidden>…转义后的源码…</div>
 *     <div class="kt-mermaid-out" role="img" aria-label="Mermaid 图表"></div>
 *   </figure>
 *
 * 兼容两种输入形态：
 *   1. 原始 code 块（astro.config 已把 mermaid 加入 shiki excludeLangs）；
 *   2. 已被 shiki 高亮、内部为 token span 树的 code 块（递归提取纯文本）。
 *
 * 源码放在 hidden div 而非 <script> 里：script 是 raw-text 序列化、
 * 不做实体转义，源码含 "<" 时会产生非法 HTML；div 的文本会被正确转义。
 */
const LANG_CLASS = 'language-mermaid';

const isElement = (n) => n != null && n.type === 'element';

/** hast properties 里 className 可能是数组（remark-rehype）或字符串（部分插件）。 */
function classList(node) {
	const raw = node?.properties?.className ?? node?.properties?.class;
	if (!raw) return [];
	return typeof raw === 'string' ? raw.split(/\s+/).filter(Boolean) : raw;
}

/** 递归提取节点内的纯文本（兼容 shiki token 树）。 */
function textOf(node) {
	if (node == null) return '';
	if (node.type === 'text') return node.value;
	if (Array.isArray(node.children)) return node.children.map(textOf).join('');
	return '';
}

export default function rehypeMermaid() {
	return (tree) => {
		const walk = (node) => {
			if (!Array.isArray(node.children)) return;
			for (let i = 0; i < node.children.length; i++) {
				const child = node.children[i];
				walk(child);
				if (
					isElement(child) &&
					child.tagName === 'pre' &&
					Array.isArray(child.children) &&
					child.children.some((c) => isElement(c) && c.tagName === 'code' && classList(c).includes(LANG_CLASS))
				) {
					const code = child.children.find((c) => isElement(c) && c.tagName === 'code');
					const source = textOf(code).replace(/\n+$/, '');
					if (!source.trim()) continue;
					node.children[i] = {
						type: 'element',
						tagName: 'figure',
						properties: { className: ['kt-mermaid'] },
						children: [
							{
								type: 'element',
								tagName: 'div',
								properties: { className: ['kt-mermaid-src'], hidden: true },
								children: [{ type: 'text', value: source }],
							},
							{
								type: 'element',
								tagName: 'div',
								properties: { className: ['kt-mermaid-out'], role: 'img', ariaLabel: 'Mermaid 图表' },
								children: [
									{ type: 'element', tagName: 'span', properties: { className: ['kt-mermaid-loading'] }, children: [{ type: 'text', value: '图表渲染中…' }] },
								],
							},
						],
					};
				}
			}
		};
		walk(tree);
	};
}
