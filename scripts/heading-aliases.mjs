#!/usr/bin/env node
/**
 * Before a title-only edit:
 *   node scripts/heading-aliases.mjs capture /tmp/heading-baseline.json
 * After docs:sync, inspect the proposed per-page map, then persist it:
 *   node scripts/heading-aliases.mjs update /tmp/heading-baseline.json
 *   node scripts/heading-aliases.mjs update /tmp/heading-baseline.json --write
 * This deliberately rejects structural edits: review those mappings by hand.
 */
import { existsSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { parseArgs } from 'node:util';
import { createMarkdownProcessor } from '@astrojs/markdown-remark';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import rehypeHeadingAliases from './rehype-heading-aliases.mjs';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const ALIASES = resolve(ROOT, 'src/data/heading-aliases.json');

/** Empty alias spans cannot affect the text/IDs collected by this renderer. */
export async function captureHeadings(entries, docsRoot) {
	const renderer = await createMarkdownProcessor({
		syntaxHighlight: false, remarkPlugins: [remarkMath],
		rehypePlugins: [rehypeKatex, [rehypeHeadingAliases, {
			aliases: { version: 1, pages: {} }, docsRoot, articleSlugs: entries.map((entry) => entry.slug),
		}]],
	});
	const pages = {};
	for (const { slug } of entries) {
		const path = resolve(docsRoot, `${slug}.md`);
		const source = readFileSync(path, 'utf8').replace(/^---\r?\n[\s\S]*?\r?\n---(?:\r?\n|$)/, '');
		const result = await renderer.render(source, { fileURL: pathToFileURL(path) });
		pages[slug] = result.metadata.headings;
	}
	return { version: 1, pages };
}

/** Pair title-only changes, preserving already-published aliases across renames. */
export function deriveAliases(before, after, previous = { version: 1, pages: {} }) {
	for (const data of [before, after, previous]) {
		if (data.version !== 1 || !data.pages) throw new Error('heading-aliases: invalid data version/pages');
	}
	const pages = structuredClone(previous.pages);
	for (const [slug, oldHeadings] of Object.entries(before.pages)) {
		const current = after.pages[slug];
		if (!current || oldHeadings.length !== current.length) {
			throw new Error(`${slug}: heading count changed; map structural edits manually`);
		}
		const currentIds = new Set(current.map((heading) => heading.slug));
		const changed = new Map();
		oldHeadings.forEach((old, index) => {
			const next = current[index];
			if (old.depth !== next.depth) throw new Error(`${slug}: heading level changed at ${index + 1}`);
			if (old.slug === next.slug) return;
			if (currentIds.has(old.slug)) throw new Error(`${slug}: heading order changed near #${old.slug}; map manually`);
			changed.set(old.slug, next.slug);
		});
		const aliases = pages[slug] ?? {};
		for (const [oldId, target] of Object.entries(aliases)) {
			const next = changed.get(target) ?? target;
			if (!currentIds.has(next)) throw new Error(`${slug}: existing alias #${oldId} has no current target #${next}`);
			aliases[oldId] = next;
		}
		for (const [oldId, target] of changed) aliases[oldId] = target;
		if (Object.keys(aliases).length) pages[slug] = aliases;
	}
	return { version: 1, pages };
}

async function main() {
	const { positionals, values } = parseArgs({
		allowPositionals: true,
		options: { docs: { type: 'string' }, write: { type: 'boolean', default: false } },
	});
	const [command, baseline] = positionals;
	if (!['capture', 'update'].includes(command) || !baseline || positionals.length !== 2) {
		throw new Error('usage: heading-aliases.mjs capture|update BASELINE.json [--docs DIR] [--write]');
	}
	const entries = JSON.parse(readFileSync(resolve(ROOT, 'src/data/content-map.json'), 'utf8')).entries;
	const snapshot = await captureHeadings(entries, resolve(values.docs ?? resolve(ROOT, 'src/content/docs')));
	if (command === 'capture') {
		writeFileSync(resolve(baseline), JSON.stringify(snapshot, null, 2) + '\n', { flag: 'wx' });
		console.log(`captured ${entries.length} article heading lists: ${resolve(baseline)}`);
		return;
	}
	const previous = existsSync(ALIASES) ? JSON.parse(readFileSync(ALIASES, 'utf8')) : { version: 1, pages: {} };
	const aliases = deriveAliases(JSON.parse(readFileSync(resolve(baseline), 'utf8')), snapshot, previous);
	const serialized = JSON.stringify(aliases, null, 2) + '\n';
	if (!values.write) process.stdout.write(serialized);
	else {
		writeFileSync(ALIASES, serialized);
		console.log(`saved ${Object.values(aliases.pages).reduce((count, page) => count + Object.keys(page).length, 0)} aliases across ${Object.keys(aliases.pages).length} articles`);
	}
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
	main().catch((error) => { console.error(error.message); process.exitCode = 1; });
}
