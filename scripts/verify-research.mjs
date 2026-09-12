import assert from 'node:assert/strict'
import {build} from 'esbuild'
import katex from 'katex'
import * as T from 'three'
import {mkdir} from 'node:fs/promises'
await mkdir('tmp',{recursive:true})
await build({stdin:{contents:`export * from './src/components/researchLab/catalog'; export * from './src/components/researchLab/sharedPrivateModel'; export * from './src/components/researchLab/pnpModel'; export * from './src/components/researchLab/ssdiffModel'; export * from './src/components/researchLab/metscModel'; export * from './src/components/researchLab/pigmentModel'; export * from './src/components/researchLab/operators'; export * from './src/components/researchLab/coreOperators'; export * from './src/components/researchLab/scientificVisuals'; export * from './src/components/mmhvae/detailLayout';`,resolveDir:process.cwd()},bundle:true,platform:'node',format:'esm',packages:'external',outfile:'tmp/research-verification.mjs'})
const lib=await import('../tmp/research-verification.mjs')
const models=[lib.sharedPrivateModel(),lib.pnpModel(),lib.ssdiffModel(),lib.metscModel(),lib.pigmentModel()]
const operations=new Map();let entries=0
for(const model of models){
 for(const e of Object.values(model.entries)){
  entries++;assert(e.children.length||e.math,`${model.id}/${e.id}: unexplained leaf`)
  if(e.parent)assert(model.entries[e.parent].children.includes(e.id))
  for(const edge of e.edges){assert(model.entries[edge.from],`${e.id}: ${edge.from}`);assert(model.entries[edge.to],`${e.id}: ${edge.to}`)}
  for(const formula of [e.formula,...(e.math?.steps.map(s=>s.formula)??[])])if(formula)katex.renderToString(formula,{throwOnError:true,strict:'ignore'})
  const g=lib.graphForEntry(model,e.id),ids=new Set(g.parts.map(p=>p.id));assert(g.parts.length,`${e.id} empty graph`)
  for(const edge of g.edges){assert(ids.has(edge.from),`${e.id}: invisible from ${edge.from}`);assert(ids.has(edge.to),`${e.id}: invisible to ${edge.to}`)}
  const layout=lib.detailLayout(g)
  for(const p of g.parts){assert(layout.positions.get(p.id).every(Number.isFinite));if(p.role)assert(p.role==='input'?layout.positions.get(p.id)[0]<layout.min[0]:layout.positions.get(p.id)[0]>layout.max[0],`${e.id}: external port must stay outside cage`)}
  if(e.math)operations.set(`${e.math.operation}:${e.math.kernel}:${e.math.bias}`,e.math)
 }
 for(const e of Object.values(model.entries).filter(e=>e.overview&&e.id!=='root'))for(const p of lib.graphForEntry(model,e.id).parts.filter(p=>!p.role))if(model.overview.includes(p.id))assert.deepEqual(p.position,model.entries[p.id].position,'Overview subset preserves placement')
}
const matrix=new T.Matrix4(),position=new T.Vector3(),scale=new T.Vector3(),rotation=new T.Quaternion()
for(const spec of operations.values())for(let stage=0;stage<spec.steps.length;stage++){
 const visual=lib.makeScientificVisual({...spec,stage})
 for(const probe of [0,.42,1]){
  if(/^(lab|core):/.test(spec.operation)){const n=lib.scientificExample({...spec,stage},probe);assert([...n.a,...n.b,...n.out].every(Number.isFinite),spec.operation);assert(!/NaN|undefined/.test(n.readout),spec.operation)}
  visual.update({probe,phase:.61,temperature:1})
  visual.group.traverse(o=>{assert(o.position.toArray().every(Number.isFinite),spec.operation);if(o.isInstancedMesh&&o.parent.userData.crystalTensor)for(let i=0;i<o.count;i++){o.getMatrixAt(i,matrix);matrix.decompose(position,rotation,scale);assert(position.toArray().every(Number.isFinite));assert(scale.toArray().every(v=>Math.abs(v-1)<1e-6))}})
 }
}
const ss=models[2],met=models[3],pig=models[4]
assert.equal(ss.entries.encoder.children.length,12);assert.equal(ss.entries.decoder.children.length,6)
assert.equal(ss.entries['features/mean'].math.operation,'lab:token-mean')
assert.equal(met.entries.encoder.children.length,6);assert.equal(met.entries.sparse.children.filter(id=>/iteration\d+$/.test(id)).length,9)
assert.equal(met.entries['fractions/split'].math.operation,'lab:split:unequal');assert.equal(pig.sourceMode,'paper')
assert(lib.graphForEntry(met,'sparse/iteration5').edges.some(e=>e.from.startsWith('port:')&&e.to==='sparse/iteration5/add'))
const n=(op,p=.42,stage=2)=>lib.scientificExample({...lib.coreSpec('noise'),operation:op,stage},p)
for(let row=0;row<4;row++)assert(Math.abs(n('core:softmax').out.slice(row*4,row*4+4).reduce((a,b)=>a+b)-1)<1e-12)
assert(Math.abs(n('core:angular',1).out[0])<1e-12)
assert.deepEqual(n('core:dropout:0').a,n('core:dropout:0').out)
assert.notDeepEqual(n('lab:poe-precision:one').out,n('lab:poe-precision:two').out)
const input=Array.from({length:16},(_,i)=>({re:Math.sin(i),im:Math.cos(i)})),inverse=lib.complexDFT2(lib.complexDFT2(input),true)
input.forEach((v,i)=>assert(Math.hypot(v.re-inverse[i].re,v.im-inverse[i].im)<1e-12))
console.log(`Verified ${models.length} models, ${entries} recursive entries, ${operations.size} operator configurations, KaTeX, boundary routing, subset layouts, crystal transforms and numerical invariants.`)
