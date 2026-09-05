import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import ts from 'typescript'

// Validate the source-backed topology without requiring WebGL or a browser.
const compile = source => ts.transpileModule(source, { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.ESNext } }).outputText
const moduleURL = source => `data:text/javascript;base64,${Buffer.from(source).toString('base64')}`
const modelURL = moduleURL(compile(await readFile(new URL('../src/components/mmhvae/model.ts', import.meta.url), 'utf8')))
const anatomySource = compile(await readFile(new URL('../src/components/mmhvae/anatomy.ts', import.meta.url), 'utf8')).replace("'./model'", JSON.stringify(modelURL))
const { NODES, LEVELS, MODALITIES } = await import(modelURL)
const { anatomyFor } = await import(moduleURL(anatomySource))
const visited = new Set()
const pending = [...NODES.map(node => node.id), ...LEVELS.map(level => `layer-${level.l}`)]
while (pending.length) {
  const id = pending.pop()
  if (visited.has(id)) continue
  visited.add(id)
  const graph = anatomyFor(id)
  const ids = new Set(graph.parts.map(part => part.id))
  assert.equal(ids.size, graph.parts.length, `${id}: duplicate parts`)
  assert(graph.parts.length > 0, `${id}: empty anatomy`)
  for (const edge of graph.edges) {
    assert(ids.has(edge.from) && ids.has(edge.to), `${id}: dangling edge ${edge.from} → ${edge.to}`)
    assert.notEqual(edge.from, edge.to, `${id}: self dependency`)
  }
  // Every non-input part must be reachable from at least one input; cycles are invalid.
  const incoming = new Map(graph.parts.map(part => [part.id, 0]))
  graph.edges.forEach(edge => incoming.set(edge.to, incoming.get(edge.to) + 1))
  const queue = [...incoming.keys()].filter(key => incoming.get(key) === 0)
  let reached = 0
  while (queue.length) {
    const next = queue.shift(); reached++
    for (const edge of graph.edges.filter(edge => edge.from === next)) {
      incoming.set(edge.to, incoming.get(edge.to) - 1)
      if (incoming.get(edge.to) === 0) queue.push(edge.to)
    }
  }
  assert.equal(reached, ids.size, `${id}: cyclic computation`)
  for (const part of graph.parts) if (part.child) pending.push(part.child)
}
for (const modality of MODALITIES) {
  assert.equal(anatomyFor(`${modality.id}-output`).parts.filter(part => part.child?.includes('-resnet-')).length, 6)
  assert.equal(NODES.filter(node => node.mod === modality.id && node.kind === 'encoder').length, 7)
}
assert.equal(anatomyFor('layer-1').parts.filter(part => part.id.endsWith('-output')).length, 4)
assert(anatomyFor('decoder-4').parts.some(part => part.detail.includes('groups=768')))
assert(anatomyFor('sample-7').parts.some(part => part.shape === 'B × 256'))
assert(anatomyFor('poe-4').parts.some(part => part.detail.includes('1/scale')))
for (const level of LEVELS) {
  for (const observed of [['us'], ['t2', 'cet1', 'flair'], MODALITIES.map(mod => mod.id)]) {
    const graph = anatomyFor(`layer-${level.l}`, observed)
    assert.equal(graph.parts.filter(part => part.id.endsWith('-q')).length, observed.length)
  }
}
console.log(`Verified ${visited.size} MMHVAE module, nested-block and layer graphs.`)
