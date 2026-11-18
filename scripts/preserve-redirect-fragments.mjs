import { readFile, writeFile } from 'node:fs/promises';

/** Static Pages redirects must carry bookmarks and query strings to the new route. */
export default function preserveRedirectFragments(redirects, base) {
	return {
		name: 'knowledge-tree-redirect-fragments',
		hooks: {
			'astro:build:done': async ({ dir }) => {
				for (const [from, to] of Object.entries(redirects)) {
					const file = new URL(`${from.replace(/^\/+|\/+$/g, '')}/index.html`, dir);
					const html = await readFile(file, 'utf8');
					const marker = '<meta http-equiv="refresh"';
					if (!html.includes(marker)) throw new Error(`Missing static redirect: ${from}`);
					const target = JSON.stringify(`${base}${to.replace(/\/$/, '')}/`).replaceAll('<', '\\u003c');
					// Run before meta-refresh; the original refresh and link remain the no-JS fallback.
					const script = `<script>location.replace(${target}+location.search+location.hash)</script>`;
					await writeFile(file, html.replace(marker, script + marker));
				}
			},
		},
	};
}
