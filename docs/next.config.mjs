import { createMDX } from 'fumadocs-mdx/next';

const withMDX = createMDX();

/** @type {import('next').NextConfig} */
const config = {
  output: 'export',
  // GitHub Pages serves `dir/index.html` for `/dir/` reliably, unlike `dir.html`
  // next to a `dir/` folder of the same name.
  trailingSlash: true,
  // Empty for `npm run dev`; the Pages build sets it to the repository subpath.
  basePath: process.env.NEXT_PUBLIC_BASE_PATH ?? '',
  reactStrictMode: true,
  images: { unoptimized: true },
};

export default withMDX(config);
