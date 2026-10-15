/**
 * domains.mjs — 六大知识领域的单一事实源（single source of truth）。
 *
 * 消费方：
 *   - astro.config.mjs          侧边栏内容分组（label + 目录 id）
 *   - scripts/sync-docs.mjs     tech_stack 域目录覆盖性校验（tsDir）与站点域目录清理（id）
 *   - src/content/docs/index.mdx 首页领域卡片（icon / blurb / 计数）
 *   - src/components/knowledge-map/KnowledgeMap.tsx 图谱域色与领域视图排序
 *   - scripts/rehype-article-tools.mjs 文章面包屑的域标签
 *
 * 新增知识领域：在此登记一条，并在 content-map.json / knowledge-relations.json
 * 中使用其 id；侧栏、首页卡片、图谱域色与筛选即自动生效。
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
		blurb: 'sensor → 特征提取 → RL 推理 → control 的学习式管线：从网络骨干、PPO/SAC 到学习式轨迹规划。',
	},
	{
		id: 'engineering-tools',
		tsDir: '04_engineering-tools',
		label: '工程工具链',
		icon: 'desktop',
		color: '#fbbf24',
		blurb: 'Git 与 Docker、PX4 与 ROS、多 GPU 训练部署与分布式强化学习训练架构。',
	},
	{
		id: 'simulation-evaluation',
		tsDir: '05_simulation-evaluation',
		label: '仿真与评估',
		icon: 'document',
		color: '#a78bfa',
		blurb: '仿真平台、场景设计、评估方法与 sim-to-real。',
	},
	{
		id: 'real-flight',
		tsDir: '06_real-flight',
		label: '真机飞行',
		icon: 'rocket',
		color: '#f87171',
		blurb: '从仿真走向实机：遥控器与机载设备配置、外场飞行实践。',
	},
];

/** id → 领域对象 */
export const DOMAIN_BY_ID = Object.fromEntries(DOMAINS.map((d) => [d.id, d]));
