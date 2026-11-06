/** Standard article modules use IDs independent of their display titles. */
export function assignArticleModuleIds(tree, moduleTitles) {
	const ids = new Map(moduleTitles.map((title, index) => [title, `kt-section-${index + 1}`]));
	const text = (node) => node.value ?? (node.children ?? []).map(text).join('');
	const visit = (node) => {
		if (node.type === 'element' && node.tagName === 'h2') {
			const id = ids.get(text(node));
			if (id) { node.properties ??= {}; node.properties.id = id; }
		}
		for (const child of node.children ?? []) visit(child);
	};
	visit(tree);
}

/** Move per-page heading compatibility metadata when a public route changes. */
export function renameHeadingAliasPage(data, from, to = from) {
	if (data.version !== 1 || !data.pages || typeof data.pages !== 'object' || Array.isArray(data.pages)) {
		throw new Error('heading-aliases: expected version 1 and a pages map');
	}
	const result = structuredClone(data);
	if (from === to) return result;
	// Even an orphaned target entry may belong to an older public page. Do not
	// silently replace or merge it: the author must resolve that collision first.
	if (Object.hasOwn(result.pages, to)) throw new Error(`新 slug 与已有标题别名冲突: ${to}`);
	if (Object.hasOwn(result.pages, from)) {
		result.pages[to] = result.pages[from];
		delete result.pages[from];
	}
	return result;
}
