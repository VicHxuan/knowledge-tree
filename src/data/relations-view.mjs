/**
 * relations-view.mjs — 知识关联的统一读取层（元数据兼容层）。
 *
 * knowledge-relations.json 只保留「图结构」职责：节点 id + 边；
 * 展示信息（标题 / 状态 / 域 / 域标签 / 顺序）优先取 content-map.json
 * （事实源——sync 据此生成 draft 标记），缺失时回退到 relations 自带字段。
 * 这样新增文章即使忘了同步 relations 的展示字段，图谱与右栏关联卡也不会
 * 显示过期信息；字段漂移由 validate-docs 的 --consistency 检查告警。
 *
 * 消费方：src/components/knowledge-map/KnowledgeMap.tsx、
 *        src/components/TableOfContents.astro。
 */
import relations from './knowledge-relations.json';
import contentMap from './content-map.json';
import { DOMAIN_BY_ID } from './domains.mjs';

const MAP_BY_SLUG = new Map(contentMap.entries.map((e) => [e.slug, e]));

export const NODES = relations.nodes.map((n) => {
	const m = MAP_BY_SLUG.get(n.id);
	const domain = m ? m.slug.split('/')[0] : n.domain;
	return {
		id: n.id,
		label: m?.label ?? n.label ?? n.id,
		title: m?.title ?? n.title ?? n.label ?? n.id,
		domain,
		domainLabel: DOMAIN_BY_ID[domain]?.label ?? n.domainLabel ?? domain,
		status: m?.status ?? n.status,
		order: m?.order ?? n.order ?? 999,
	};
});

export const EDGES = relations.edges;

export const NODE_BY_ID = new Map(NODES.map((n) => [n.id, n]));

/** id 的直接前置知识（prerequisite 入边来源） */
export function prerequisitesOf(id) {
	return EDGES.filter((e) => e.type === 'prerequisite' && e.to === id).map((e) => e.from);
}

/** id 的后续依赖（prerequisite 出边目标）——「谁用到它」 */
export function dependentsOf(id) {
	return EDGES.filter((e) => e.type === 'prerequisite' && e.from === id).map((e) => e.to);
}

/** id 的相关主题（related 边对端，无方向） */
export function relatedOf(id) {
	return EDGES.filter((e) => e.type === 'related' && (e.from === id || e.to === id)).map((e) =>
		e.from === id ? e.to : e.from,
	);
}
