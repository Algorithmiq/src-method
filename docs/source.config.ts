import { defineConfig, defineDocs } from 'fumadocs-mdx/config';
import { metaSchema, pageSchema } from 'fumadocs-core/source/schema';
import { z } from 'zod';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import rehypeCitation from 'rehype-citation';
import rehypeRaw from 'rehype-raw';
import remarkXref from './scripts/remark-xref.mjs';

// The tutorial pages are generated as `.md` (not `.mdx`) and embed their
// figures as raw `<img src="data:image/png;base64,…">` tags. In md-format
// MDX leaves `<` literal, so those tags become "raw" HTML nodes — which MDX's
// internal `rehype-remove-raw` strips just after the user rehype plugins run,
// leaving the images blank in the export. `rehype-raw` reparses raw HTML into
// real hast nodes *before* that removal, so the `<img>` survives. We gate it to
// `.md` files: `.mdx` pages carry MDX JSX nodes that rehype-raw would mangle.
function rehypeRawMarkdown() {
  // Pass MDX nodes through untouched: fumadocs injects an `mdxjsEsm` export
  // (for `includeProcessedMarkdown`) even into md-format trees, and rehype-raw
  // cannot compile MDX nodes.
  const transform = rehypeRaw({
    passThrough: [
      'mdxjsEsm',
      'mdxFlowExpression',
      'mdxTextExpression',
      'mdxJsxFlowElement',
      'mdxJsxTextElement',
    ],
  });
  return (tree: import('hast').Root, file: import('vfile').VFile) => {
    if (file.path && !file.path.endsWith('.md')) return tree;
    return transform(tree, file);
  };
}

// `rehypeCitation` links each `[@key]` to a `#bib-key` anchor on the same page,
// but the bibliography is only written out on the references page (see
// `suppressBibliography` below). Point the links there instead.
function rehypeCitationsToReferences() {
  const visit = (node: import('hast').Nodes) => {
    if (node.type === 'element' && node.tagName === 'a') {
      const href = node.properties?.href;
      if (typeof href === 'string' && href.startsWith('#bib-')) {
        node.properties.href = `/references${href}`;
      }
    }
    if ('children' in node) node.children.forEach(visit);
  };
  return (tree: import('hast').Root) => visit(tree);
}

// You can customize Zod schemas for frontmatter and `meta.json` here
// see https://fumadocs.dev/docs/mdx/collections
export const docs = defineDocs({
  dir: 'content/docs',
  docs: {
    schema: pageSchema.extend({
      githubPath: z.string().optional(),
    }),
    postprocess: {
      includeProcessedMarkdown: true,
    },
  },
  meta: {
    schema: metaSchema,
  },
});

export default defineConfig({
  mdxOptions: {
    // Tutorial pages inline figures as base64 `data:` URIs; keep them as plain
    // `<img src>` instead of turning images into static imports.
    remarkImageOptions: { useImport: false },
    remarkPlugins: [remarkMath, remarkXref],
    // `rehypeCitation` resolves `[@key]` references against bibliography.bib
    // (replacing sphinxcontrib-bibtex); `rehypeKatex` renders the math nodes
    // produced by `remarkMath`. Both run before the default fumadocs plugins.
    rehypePlugins: (plugins) => [
      rehypeRawMarkdown,
      rehypeKatex,
      [
        rehypeCitation,
        {
          bibliography: './bibliography.bib',
          linkCitations: true,
          // The references page hand-writes the formatted bibliography (with
          // `id="..."` anchors matching the bib keys), so suppress the
          // per-page bibliography that rehype-citation would otherwise append
          // to every page that uses an inline `[@key]` citation.
          suppressBibliography: true,
        },
      ],
      rehypeCitationsToReferences,
      ...plugins,
    ],
  },
});
