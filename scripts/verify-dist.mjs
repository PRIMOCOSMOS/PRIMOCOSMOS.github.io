import { readFile, stat } from 'node:fs/promises'
import { resolve, relative, isAbsolute } from 'node:path'

// Pages must receive Vite's build output, never the source index.html.
const directory = resolve(process.argv[2] ?? 'dist')
const html = await readFile(resolve(directory, 'index.html'), 'utf8')
if (/(?:src|href)=["'][^"']*(?:\/src\/|\.(?:tsx?|jsx)(?:[?"']))/i.test(html)) {
  throw new Error('Uncompiled source entry found. Run npm run build and deploy dist/.')
}
const assets = [...html.matchAll(/(?:src|href)=["']([^"']+)["']/g)]
  .map(([, url]) => url)
  .filter(url => !/^(?:[a-z][a-z0-9+.-]*:|\/\/|#)/i.test(url))
if (!assets.some(url => /\.js(?:\?|$)/.test(url)) || !assets.some(url => /\.css(?:\?|$)/.test(url))) {
  throw new Error('The built entry must reference JavaScript and CSS assets.')
}
for (const asset of assets) {
  const path = resolve(directory, decodeURIComponent(asset.split(/[?#]/)[0]).replace(/^\//, ''))
  const local = relative(directory, path)
  if (local.startsWith('..') || isAbsolute(local) || !(await stat(path)).isFile()) {
    throw new Error(`Missing or invalid deployed asset: ${asset}`)
  }
}
console.log(`Verified built HTML and ${assets.length} local asset references.`)
