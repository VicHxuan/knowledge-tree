/**
 * domains.mjs — 六大知识领域的单一事实源（single source of truth）。
 *
 * 消费方：
 *   - astro.config.mjs          侧边栏内容分组（label + 目录 id）
 *   - scripts/sync-docs.mjs     tech_stack 域目录覆盖性校验（tsDir）与路由领域校验（id）
 *   - src/content/docs/index.mdx 首页领域卡片（icon / blurb / 计数）
 *   - src/components/knowledge-map/KnowledgeMap.tsx 图谱域色与领域视图排序
 *   - scripts/rehype-article-tools.mjs 文章面包屑的域标签
 *
 * 新增知识领域：在此登记一条，并在 content-map.json 中使用其 id；
 * 侧栏、首页卡片、图谱域色与筛选即自动生效，关系边引用完整文章 slug。
 * 图标名取自 Starlight 内置图标集（schema.d.ts StarlightIcon）。
 */
export const DOMAINS = [
	{
		id: 'mathematics',
		tsDir: '01_math-physic-base',
		label: '数学基础',
		icon: 'open-book',
		color: '#38bdf8',
		blurb: '线性代数与几何、微积分与优化、概率统计、动力学、最优控制——整棵知识树的根。',
	},
	{
		id: 'classical-planning-control',
		tsDir: '02_classical-planning-control',
		label: '传统规划控制',
		icon: 'code-branch',
		color: '#f472b6',
		blurb: 'sensor → SLAM → 搜索 → 轨迹优化 → 跟踪控制的经典管线。',
	},
	{
		id: 'learning-based-planning',
		tsDir: '03_learning-based-planning',
		label: '学习式规划',
		icon: 'analytics',
		color: '#34d399',
		blurb: '观测 → 策略或规划网络 → 动作与轨迹 → 控制执行：连接网络、强化学习与无人机导航。',
	},
	{
		id: 'engineering-tools',
		tsDir: '04_engineering-tools',
		label: '工程工具链',
		icon: 'desktop',
		color: '#fbbf24',
		blurb: '飞行软件栈、训练基础设施与版本管理：连接算法训练和机载执行。',
	},
	{
		id: 'simulation-evaluation',
		tsDir: '05_simulation-evaluation',
		label: '仿真与评估',
		icon: 'document',
		color: '#a78bfa',
		blurb: '仿真环境与时序、导航场景、固定协议评估与真机迁移。',
	},
	{
		id: 'real-flight',
		tsDir: '06_real-flight',
		label: '真机飞行',
		icon: 'rocket',
		color: '#f87171',
		blurb: '遥控链路、机载导航接口与分阶段验证。',
	},
];

/** id → 领域对象 */
export const DOMAIN_BY_ID = Object.fromEntries(DOMAINS.map((d) => [d.id, d]));
