import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import ts from 'typescript'

// Validate the source-backed topology without requiring WebGL or a browser.
const compile = source => ts.transpileModule(source, { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.ESNext } }).outputText
const moduleURL = source => `data:text/javascript;base64,${Buffer.from(source).toString('base64')}`
const modelURL = moduleURL(compile(await readFile(new URL('../src/components/mmhvae/model.ts', import.meta.url), 'utf8')))
const topologyURL = moduleURL(compile(await readFile(new URL('../src/components/mmhvae/topology.ts', import.meta.url), 'utf8')).replace("'./model'", JSON.stringify(modelURL)))
const anatomySource = compile(await readFile(new URL('../src/components/mmhvae/anatomy.ts', import.meta.url), 'utf8')).replace("'./model'", JSON.stringify(modelURL)).replace("'./topology'", JSON.stringify(topologyURL))
const { NODES, LEVELS, MODALITIES } = await import(modelURL)
const { anatomyFor } = await import(moduleURL(anatomySource))
const { modelTopology } = await import(topologyURL)
const topology = modelTopology()
for (const link of topology.links) assert(topology.positions.has(link.from) && topology.positions.has(link.to), `Missing topology endpoint: ${link.from} → ${link.to}`)
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
assert(anatomyFor('decoder-4/se/768').parts.filter(part => part.role === 'input').every(part => part.shape.startsWith('768')))
assert(anatomyFor('flair-resnet-6').parts.some(part => part.role === 'output' && part.sourceName.endsWith('.convt2') && part.shape.startsWith('8 ×')))
for (const level of LEVELS) {
  for (const observed of [['us'], ['t2', 'cet1', 'flair'], MODALITIES.map(mod => mod.id)]) {
    const graph = anatomyFor(`layer-${level.l}`, observed)
    assert.equal(graph.parts.filter(part => part.id.includes('-expert-')).length, observed.length)
    assert(graph.parts.every(part => part.position?.length === 3))
    if (level.l < 7) for (const key of [`posterior-${level.l+1}`, `sample-${level.l+1}`, `up-${level.l}`, `decoder-${level.l}`, `feature-${level.l}`, `prior-head-${level.l}`, `prior-${level.l}`, `poe-${level.l}`, `posterior-${level.l}`, `sample-${level.l}`]) assert(graph.parts.some(part => part.id === key), `Missing layer stage: ${key}`)
  }
}
console.log(`Verified ${visited.size} MMHVAE module, nested-block and layer graphs.`)

const anatomyURL = moduleURL(anatomySource)
const navigationURL = moduleURL(compile(await readFile(new URL('../src/components/mmhvae/navigation.ts', import.meta.url), 'utf8')).replace("'./model'", JSON.stringify(modelURL)).replace("'./anatomy'", JSON.stringify(anatomyURL)).replace("'./topology'", JSON.stringify(topologyURL)))
const { childrenOf, canonicalPath, graphFor, navName, atomInfo, principle } = await import(navigationURL)
const { detailLayout } = await import(moduleURL(compile(await readFile(new URL('../src/components/mmhvae/detailLayout.ts', import.meta.url), 'utf8'))))
const indexed = new Set(), queue = ['root'], observed = MODALITIES.map(m => m.id)
while(queue.length) {
  const id = queue.shift(); if(indexed.has(id)) continue; indexed.add(id)
  assert(indexed.size < 10000, 'Containment hierarchy must terminate')
  assert(navName(id, observed), `Missing readable name: ${id}`)
  const children = childrenOf(id, observed), graph = graphFor(id, observed)
  for(const child of children) { assert.equal(canonicalPath(child).at(-2), id, `Inconsistent parent of ${child}`); queue.push(child) }
  if(!graph) continue
  for(const compact of [false,true]) {
  const layout = detailLayout(graph,compact)
  for(const part of graph.parts) {
    const pos = layout.positions.get(part.id)
    assert(pos.every(Number.isFinite), `${id}: invalid position`)
    if(part.role) assert(pos.some((v,i)=>v<layout.min[i]-2||v>layout.max[i]+2), `${id}: context inside module`)
    else assert(pos.every((v,i)=>v>=layout.min[i]&&v<=layout.max[i]), `${id}: child outside module`)
  }
  }
  const atom = atomInfo(id, observed)
  if(atom) { assert.equal(children.length, 0); assert(principle(atom.part).formula) }
}
for(const n of NODES) assert(indexed.has(n.id), `Missing module in containment index: ${n.id}`)
console.log(`Verified ${indexed.size} hierarchy entries, atomic explanations and outside context rails.`)

for(const id of ['encoders','encoder:us','encoder:flair','core','outputs']) {
  const graph=graphFor(id,observed)
  assert.equal(graph.layout,'overview')
  for(const compact of [false,true])for(const [key,point] of detailLayout(graph,compact).positions)assert.deepEqual(point,topology.positions.get(key),`${id}: overview coordinates changed`)
}
for(const id of ['us-encoder-4','decoder-4','decoder-4/se/768','flair-output','flair-resnet-2','poe-4','sample-4']) {
  const graph=graphFor(id,observed),layout=detailLayout(graph)
  const spine=layout.spine.map(key=>layout.positions.get(key))
  assert(spine.length>1)
  spine.forEach((point,i)=>{assert.equal(point[0],0);assert.equal(point[2],0);if(i)assert(point[1]<spine[i-1][1],`${id}: trunk must descend`)})
}
console.log('Verified vertical computational trunks and exact overview subset coordinates.')
