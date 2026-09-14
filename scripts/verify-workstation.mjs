import assert from 'node:assert/strict'
import {build} from 'esbuild'
import {writeFile,mkdir} from 'node:fs/promises'
import katex from 'katex'
await mkdir('tmp',{recursive:true})
await build({stdin:{contents:"export * from './src/components/workstation/engine'; export * from './src/components/workstation/models'",resolveDir:process.cwd()},bundle:true,format:'esm',platform:'node',outfile:'tmp/workstation-engine.mjs'})
const {execute,DEFAULT,MODULES,size}=await import('../tmp/workstation-engine.mjs')
const fixtures=[];let values=0
for(const config of [DEFAULT,{...DEFAULT,batch:2,channels:4,out:4,groups:2,stride:2,training:true,preNorm:false,causal:true,dim:8,heads:4,seed:93}])for(const def of MODULES){
 const run=execute(def.id,config);assert(run.steps.length);assert.deepEqual(run.tensors,execute(def.id,config).tensors,'Same seed must be deterministic')
 for(const t of run.tensors){assert.equal(t.values.length,size(t.shape));assert(t.values.every(Number.isFinite));values+=t.values.length}
 for(const step of run.steps){katex.renderToString(step.formula,{throwOnError:true,strict:'ignore'});for(const i of [0,Math.floor(step.output.values.length/2),step.output.values.length-1])for(const term of step.trace(i)){assert(Number.isFinite(term.value));if(term.index>=0)assert.equal(run.tensors.find(t=>t.id===term.tensor).values[term.index],term.value);if(term.factorTensor)assert(term.factorIndex<run.tensors.find(t=>t.id===term.factorTensor).values.length)}}
 fixtures.push({id:def.id,config,tensors:run.tensors,steps:run.steps.map(({trace,...s})=>({...s,inputs:s.inputs.map(t=>t.id),output:s.output.id})),output:run.output.id})
}
assert.throws(()=>execute('attention',{...DEFAULT,dim:5,heads:2}),/head/)
assert.throws(()=>execute('conv2d',{...DEFAULT,channels:3,groups:2}),/groups/)
assert.throws(()=>execute('linear',DEFAULT,[1]),/实际为/)
const changed=execute('conv2d',{...DEFAULT,channels:4,out:6,kernel:3,stride:2,padding:1});assert.deepEqual(changed.output.shape,[1,6,3,3])
assert.equal(execute('mobilev2',{...DEFAULT,out:2}).steps.at(-1).settings.operation,'add')
assert.notEqual(execute('mobilev2',{...DEFAULT,stride:2,out:2}).steps.at(-1).settings.operation,'add')
await writeFile('tmp/workstation-fixtures.json',JSON.stringify(fixtures))
console.log(`Verified ${MODULES.length} modules / ${fixtures.length} configurations, ${values.toLocaleString()} stored values, all traces, formulas, shape constraints and deterministic seeds. PyTorch fixtures ready.`)

