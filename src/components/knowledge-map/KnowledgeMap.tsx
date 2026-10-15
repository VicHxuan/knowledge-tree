import { useEffect, useMemo, useState, type CSSProperties } from 'react';
import {
	ReactFlow,
	Background,
	Controls,
	MiniMap,
	Panel,
	MarkerType,
	Position,
	Handle,
	type Edge,
	type Node,
	type NodeProps,
	type NodeTypes,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import './knowledge-map.css';
import { DOMAINS } from '../../data/domains.mjs';
import { NODES, EDGES, NODE_BY_ID } from '../../data/relations-view.mjs';

/**
 * KnowledgeMap — 知识图谱交互页（React Flow）。
 * 两种布局：依赖视图（按前置知识最长路径分层）/ 领域视图（按六大领域分列）。
 * 选中节点后进入「聚焦」态：该节点 + 直接前置 + 后续依赖 + 相关主题保持常亮，
 * 其余节点与边淡出，便于看清局部结构（不改布局，不做二次排布）。
 * 视图与选中节点会同步到 URL（?view=&node=），可从文章页直达定位。
 */

type Status = 'published' | 'draft';
type View = 'dependency' | 'domain';

interface RNode {
	id: string;
	label: string;
	title: string;
	domain: string;
	domainLabel: string;
	status: Status;
	order: number;
}
interface REdge {
	from: string;
	to: string;
	type: 'prerequisite' | 'related';
}
interface KTNodeData extends Record<string, unknown> {
	label: string;
	title: string;
	domain: string;
	domainLabel: string;
	status: Status;
	dim: boolean;
	selected: boolean;
	href: string;
}

const SITE_BASE = '/knowledge-tree';
const NODE_W = 190;
const NODE_H = 38;
const GAP_X = 64;
const GAP_Y = 12;

// 六大领域色与顺序来自 src/data/domains.mjs（单一事实源）
const DOMAIN_ORDER: string[] = DOMAINS.map((d) => d.id);
const DOMAIN_COLOR: Record<string, string> = Object.fromEntries(DOMAINS.map((d) => [d.id, d.color]));

const R_NODES = NODES as RNode[];
const R_EDGES = EDGES as REdge[];

/** 相邻关系索引：id → 前置 / 后续依赖 / 相关 */
const ADJ = (() => {
	const prereqs = new Map<string, string[]>();
	const dependents = new Map<string, string[]>();
	const related = new Map<string, string[]>();
	const push = (m: Map<string, string[]>, k: string, v: string) => {
		(m.get(k) ?? m.set(k, []).get(k)!).push(v);
	};
	for (const e of R_EDGES) {
		if (e.type === 'prerequisite') {
			push(prereqs, e.to, e.from);
			push(dependents, e.from, e.to);
		} else {
			push(related, e.from, e.to);
			push(related, e.to, e.from);
		}
	}
	return { prereqs, dependents, related };
})();

/** 依赖视图布局：按 prerequisite 最长路径分层（列），层内按领域+序号排（行）。 */
function layeredLayout(nodes: RNode[]): Map<string, { x: number; y: number }> {
	const pres = R_EDGES.filter((e) => e.type === 'prerequisite');
	const level = new Map<string, number>(nodes.map((n) => [n.id, 0]));
	for (let pass = 0; pass < nodes.length; pass++) {
		let changed = false;
		for (const e of pres) {
			const next = (level.get(e.from) ?? 0) + 1;
			if (next > (level.get(e.to) ?? 0)) {
				level.set(e.to, next);
				changed = true;
			}
		}
		if (!changed) break;
	}
	const cols = new Map<number, RNode[]>();
	for (const n of nodes) {
		const l = level.get(n.id) ?? 0;
		if (!cols.has(l)) cols.set(l, []);
		cols.get(l)!.push(n);
	}
	const pos = new Map<string, { x: number; y: number }>();
	const maxRows = Math.max(...[...cols.values()].map((c) => c.length), 1);
	for (const [l, group] of cols) {
		group.sort((a, b) =>
			a.domain === b.domain ? a.order - b.order : a.domain.localeCompare(b.domain),
		);
		const colHeight = group.length * (NODE_H + GAP_Y) - GAP_Y;
		const offset = ((maxRows * (NODE_H + GAP_Y) - GAP_Y) - colHeight) / 2;
		group.forEach((n, i) =>
			pos.set(n.id, { x: l * (NODE_W + GAP_X), y: offset + i * (NODE_H + GAP_Y) }),
		);
	}
	return pos;
}

/** 领域视图布局：六大领域分列，列内按 order 排。 */
function domainLayout(nodes: RNode[]): Map<string, { x: number; y: number }> {
	const groups = new Map<string, RNode[]>();
	for (const d of DOMAIN_ORDER) groups.set(d, []);
	for (const n of nodes) {
		if (!groups.has(n.domain)) groups.set(n.domain, []);
		groups.get(n.domain)!.push(n);
	}
	const pos = new Map<string, { x: number; y: number }>();
	const COL_W = NODE_W + 56;
	const HEADER_H = 34;
	groups.forEach((group, domain) => {
		const col = DOMAIN_ORDER.indexOf(domain);
		group.sort((a, b) => a.order - b.order);
		group.forEach((n, i) =>
			pos.set(n.id, { x: col * COL_W, y: HEADER_H + i * (NODE_H + GAP_Y) }),
		);
	});
	return pos;
}

function KTNode({ data }: NodeProps<Node<KTNodeData, 'kt'>>) {
	return (
		<div
			className="ktm-node"
			style={{ '--ktm-color': DOMAIN_COLOR[data.domain] ?? '#94a3b8' } as CSSProperties}
			data-status={data.status}
			data-dim={data.dim}
			data-selected={data.selected}
		>
			<Handle type="target" position={Position.Left} className="ktm-handle" />
			<span className="ktm-dot" />
			<span className="ktm-label">{data.label}</span>
			{data.status === 'draft' && <span className="ktm-flag">🚧</span>}
			<Handle type="source" position={Position.Right} className="ktm-handle" />
		</div>
	);
}

const NODE_TYPES: NodeTypes = { kt: KTNode };

export default function KnowledgeMap() {
	// 初始值支持从 URL 恢复（?view=domain&node=<slug>）；SSR 时 window 不存在，安全回落默认
	const [view, setView] = useState<View>(() => {
		if (typeof window === 'undefined') return 'dependency';
		return new URLSearchParams(window.location.search).get('view') === 'domain' ? 'domain' : 'dependency';
	});
	const [selected, setSelected] = useState<string | null>(() => {
		if (typeof window === 'undefined') return null;
		const n = new URLSearchParams(window.location.search).get('node');
		return n && NODE_BY_ID.has(n) ? n : null;
	});
	const [q, setQ] = useState('');
	const [status, setStatus] = useState<'all' | Status>('all');
	const [domain, setDomain] = useState<string>('all');
	const [showRelated, setShowRelated] = useState(false);

	// 视图与选中节点同步进 URL：可分享、可从文章页「在图谱中查看」直达
	useEffect(() => {
		if (typeof window === 'undefined') return;
		const params = new URLSearchParams();
		if (view !== 'dependency') params.set('view', view);
		if (selected) params.set('node', selected);
		const qs = params.toString();
		window.history.replaceState(null, '', window.location.pathname + (qs ? `?${qs}` : ''));
	}, [view, selected]);

	/** 聚焦集合：选中节点及其直接前置 / 后续依赖 / 相关主题 */
	const focusSet = useMemo(() => {
		if (!selected) return null;
		const set = new Set<string>([selected]);
		ADJ.prereqs.get(selected)?.forEach((id) => set.add(id));
		ADJ.dependents.get(selected)?.forEach((id) => set.add(id));
		ADJ.related.get(selected)?.forEach((id) => set.add(id));
		return set;
	}, [selected]);

	const filtered = useMemo(() => {
		const needle = q.trim().toLowerCase();
		return R_NODES.filter((n) => {
			if (status !== 'all' && n.status !== status) return false;
			if (domain !== 'all' && n.domain !== domain) return false;
			return true;
		}).map((n) => ({
			...n,
			dim:
				(needle.length > 0 &&
					!n.label.toLowerCase().includes(needle) &&
					!n.title.toLowerCase().includes(needle)) ||
				(focusSet !== null && !focusSet.has(n.id)),
		}));
	}, [q, status, domain, focusSet]);

	const ids = useMemo(() => new Set(filtered.map((n) => n.id)), [filtered]);

	const nodes: Node[] = useMemo(() => {
		const pos = view === 'dependency' ? layeredLayout(filtered) : domainLayout(filtered);
		return filtered.map((n) => ({
			id: n.id,
			type: 'kt' as const,
			position: pos.get(n.id) ?? { x: 0, y: 0 },
			data: {
				label: n.label,
				title: n.title,
				domain: n.domain,
				domainLabel: n.domainLabel,
				status: n.status,
				dim: n.dim,
				selected: n.id === selected,
				href: `${SITE_BASE}/${n.id}/`,
			} satisfies KTNodeData,
		}));
	}, [filtered, view, selected]);

	const edges: Edge[] = useMemo(() => {
		const out: Edge[] = [];
		for (const e of R_EDGES) {
			if (!ids.has(e.from) || !ids.has(e.to)) continue;
			if (e.type === 'related' && !showRelated) continue;
			// 聚焦态：不与选中节点相连的边整体淡出
			const dimmed = focusSet !== null && e.from !== selected && e.to !== selected;
			out.push({
				id: `${e.type}:${e.from}:${e.to}`,
				source: e.from,
				target: e.to,
				type: 'smoothstep',
				animated: false,
				className:
					(e.type === 'related' ? 'ktm-edge-related' : 'ktm-edge-prereq') +
					(dimmed ? ' ktm-edge-dim' : ''),
				markerEnd:
					e.type === 'prerequisite'
						? { type: MarkerType.ArrowClosed, width: 14, height: 14 }
						: undefined,
			});
		}
		return out;
	}, [ids, showRelated, focusSet, selected]);

	const selNode = selected ? NODE_BY_ID.get(selected) : undefined;
	const selPrereqs = selected ? (ADJ.prereqs.get(selected) ?? []) : [];
	const selDependents = selected ? (ADJ.dependents.get(selected) ?? []) : [];
	const selRelated = selected ? (ADJ.related.get(selected) ?? []) : [];
	const focusHint = focusSet ? `${focusSet.size} 个关联节点` : null;

	return (
		<div className="knowledge-map">
			<ReactFlow
				key={`${view}:${status}:${domain}:${showRelated}`}
				nodes={nodes}
				edges={edges}
				nodeTypes={NODE_TYPES}
				fitView
				fitViewOptions={{ padding: 0.15, maxZoom: 1 }}
				minZoom={0.15}
				proOptions={{ hideAttribution: true }}
				onNodeClick={(_, node) => setSelected(node.id)}
				onNodeDoubleClick={(_, node) => {
					const rn = NODE_BY_ID.get(node.id);
					if (rn?.status === 'published') window.location.assign(`${SITE_BASE}/${rn.id}/`);
				}}
				onPaneClick={() => setSelected(null)}
			>
				<Background gap={22} size={1} />
				<Controls position="bottom-right" showInteractive={false} />
				<MiniMap
					position="top-right"
					pannable
					zoomable
					nodeColor={(n) =>
						DOMAIN_COLOR[(n.data as KTNodeData).domain] ?? '#94a3b8'
					}
				/>
				<Panel position="top-left" className="ktm-toolbar">
					<div className="ktm-row">
						<div className="ktm-seg" role="group" aria-label="视图切换">
							<button
								type="button"
								className={view === 'dependency' ? 'on' : ''}
								onClick={() => setView('dependency')}
							>
								依赖视图
							</button>
							<button
								type="button"
								className={view === 'domain' ? 'on' : ''}
								onClick={() => setView('domain')}
							>
								领域视图
							</button>
						</div>
						<input
							type="search"
							value={q}
							placeholder="搜索知识点…"
							onChange={(e) => setQ(e.target.value)}
							aria-label="搜索知识点"
						/>
						{selected && (
							<button type="button" className="ktm-clear" onClick={() => setSelected(null)}>
								清除聚焦 ×
							</button>
						)}
					</div>
					<div className="ktm-row">
						<select
							value={status}
							onChange={(e) => setStatus(e.target.value as typeof status)}
							aria-label="状态筛选"
						>
							<option value="all">全部状态</option>
							<option value="published">已发布</option>
							<option value="draft">撰写中</option>
						</select>
						<select
							value={domain}
							onChange={(e) => setDomain(e.target.value)}
							aria-label="领域筛选"
						>
							<option value="all">全部领域</option>
							{DOMAINS.map((d) => (
								<option key={d.id} value={d.id}>
									{d.label}
								</option>
							))}
						</select>
						<label className="ktm-check">
							<input
								type="checkbox"
								checked={showRelated}
								onChange={(e) => setShowRelated(e.target.checked)}
							/>
							相关边
						</label>
					</div>
				</Panel>
				<Panel position="bottom-left" className="ktm-legend">
					{DOMAINS.map((d) => (
						<span key={d.id} className="ktm-legend-item">
							<i style={{ background: d.color }} />
							{d.label}
						</span>
					))}
					<span className="ktm-legend-item">
						<i className="ktm-legend-draft" />
						撰写中
					</span>
					<span className="ktm-legend-item">实线箭头 = 前置；虚线 = 相关</span>
				</Panel>
			</ReactFlow>

			{selNode && (
				<aside className="ktm-detail">
					<header>
						<h3>{selNode.title}</h3>
						<button type="button" onClick={() => setSelected(null)} aria-label="关闭详情">
							×
						</button>
					</header>
					<p className="ktm-detail-meta">
						<span className="ktm-badge" data-domain={selNode.domain}>
							{selNode.domainLabel}
						</span>
						<span className="ktm-badge" data-status={selNode.status}>
							{selNode.status === 'published' ? '已发布' : '撰写中 🚧'}
						</span>
					</p>
					{selPrereqs.length > 0 && (
						<section>
							<h4>前置知识</h4>
							<div className="ktm-chips">
								{selPrereqs.map((id) => (
									<button key={id} type="button" onClick={() => setSelected(id)}>
										{NODE_BY_ID.get(id)?.label ?? id}
									</button>
								))}
							</div>
						</section>
					)}
					{selDependents.length > 0 && (
						<section>
							<h4>后续依赖</h4>
							<div className="ktm-chips">
								{selDependents.map((id) => (
									<button key={id} type="button" onClick={() => setSelected(id)}>
										{NODE_BY_ID.get(id)?.label ?? id}
									</button>
								))}
							</div>
						</section>
					)}
					{selRelated.length > 0 && (
						<section>
							<h4>相关主题</h4>
							<div className="ktm-chips">
								{selRelated.map((id) => (
									<button key={id} type="button" onClick={() => setSelected(id)}>
										{NODE_BY_ID.get(id)?.label ?? id}
									</button>
								))}
							</div>
						</section>
					)}
					{selNode.status === 'published' ? (
						<a className="ktm-open" href={`${SITE_BASE}/${selNode.id}/`}>
							打开页面 →
						</a>
					) : (
						<p className="ktm-open ktm-open-draft">撰写中，暂无页面</p>
					)}
					{focusHint && <p className="ktm-focus-hint">聚焦中：{focusHint}常亮，其余淡出</p>}
				</aside>
			)}
		</div>
	);
}
