import assert from 'node:assert/strict'
import {build} from 'esbuild'
import {writeFile,mkdir} from 'node:fs/promises'
import katex from 'katex'
await mkdir('tmp',{recursive:true})
await build({stdin:{contents:"export * from './src/components/workstation/engine'; export * from './src/components/workstation/models'; export * from './src/components/workstation/scaffold'; export * from './src/components/workstation/catalog'",resolveDir:process.cwd()},bundle:true,format:'esm',platform:'node',outfile:'tmp/workstation-engine.mjs'})
const {execute,DEFAULT,MODULES,size,buildScaffold,tensorLayout,CATALOG,principalSteps}=await import('../tmp/workstation-engine.mjs')
const fixtures=[];let values=0
for(const config of [DEFAULT,{...DEFAULT,batch:2,channels:4,out:4,groups:2,stride:2,training:true,preNorm:false,causal:true,dim:8,heads:4,seed:93}])for(const def of MODULES){
 const run=execute(def.id,config);
 const graph=buildScaffold(run),nodeMap=new Map(graph.nodes.map(n=>[n.tensor.id,n]));
 for(const edge of graph.edges)assert(nodeMap.get(edge.from).level<nodeMap.get(edge.to).level,'Graph must follow actual dependency depth');
 for(const t of run.tensors){const l=tensorLayout(t);assert.equal(l.positions.length,t.values.length);assert.equal(new Set(l.positions.map(p=>p.join(','))).size,t.values.length,'Every coordinate occupies its own crystal');}
 assert.equal(graph.nodes.filter(n=>n.step).length,principalSteps(run).length,'First level exposes all principal operators');
assert(run.steps.length);assert.deepEqual(run.tensors,execute(def.id,config).tensors,'Same seed must be deterministic')
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
const indexed=CATALOG.flatMap(r=>r.branches.flatMap(b=>b.ids));assert.equal(new Set(indexed).size,MODULES.length);assert.equal(indexed.length,MODULES.length);
const v3=execute('mobilev3',{...DEFAULT,channels:8,expansion:5});assert.equal(v3.steps.find(s=>s.title==='压缩通道').output.shape[1],16,'Torchvision SE rounding includes the 0.9 safeguard');
fixtures.push({id:'mobilev3-edge',config:{...DEFAULT,channels:8,expansion:5},tensors:v3.tensors,steps:v3.steps.map(({trace,...s})=>({...s,inputs:s.inputs.map(t=>t.id),output:s.output.id})),output:v3.output.id});
const bn=execute('BatchNorm2d',DEFAULT);assert(bn.steps.filter(s=>'constant' in s.settings&&s.settings.constant!==undefined).every(s=>s.inputs.length===0&&s.trace(0).length===0),'Fixed buffers must not pretend to read input values');
await writeFile('tmp/workstation-fixtures.json',JSON.stringify(fixtures))
console.log(`Verified ${MODULES.length} modules / ${fixtures.length} configurations, ${values.toLocaleString()} stored values, all traces, formulas, shape constraints and deterministic seeds. PyTorch fixtures ready.`)

