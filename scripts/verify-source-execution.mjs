import assert from 'node:assert/strict';
import {build} from 'esbuild';
import katex from 'katex';
await build({stdin:{contents:"export * from './src/components/researchLab/sourceExecution';export * from './src/components/researchLab/sharedPrivateModel';export * from './src/components/researchLab/pnpModel';export * from './src/components/researchLab/ssdiffModel';export * from './src/components/researchLab/metscModel';export * from './src/components/researchLab/pigmentModel';export * from './src/components/workstation/scaffold';export * from './src/components/workstation/sourceScaffold';",resolveDir:process.cwd()},bundle:true,platform:'node',format:'esm',packages:'external',outfile:'tmp/source-execution-check.mjs'});
const lib=await import('../tmp/source-execution-check.mjs');
const mm=lib.makeMmhvaeRun(['us','t2']);
const tensor=(id)=>{const t=mm.tensors.find(t=>t.id===id);assert(t,id);return t};
assert.deepEqual(tensor('atom:us-expert-7:linear:W').window.dimensions,[512,1152]);
assert.deepEqual(tensor('atom:us-encoder-1:bn_0:mean').window.dimensions,[16,1,1]);
assert.deepEqual(tensor('atom:decoder-6:dw_conv_1:W').window.dimensions,[768,1,5,5]);
assert.deepEqual(tensor('atom:sample-1:mean:data').window.dimensions,[8,192,192]);
assert.equal(mm.steps.filter(s=>s.settings.sourceId==='atom:us-expert-1:cat').length,1);
assert.equal(mm.steps.filter(s=>s.settings.sourceId==='atom:prior-head-1:pz'&&s.settings.sourceConvolution).length,1);
for(const id of ['us','t2','cet1','flair'])assert(mm.sourceGraph.paths[`${id}-image`]);
assert(!mm.sourceGraph.paths['cet1-encoder-1'],'Only observed encoders execute');
const split=mm.steps.find(s=>s.settings.sourceId==='atom:us-expert-7:scale');assert.equal(split.settings.start,256);assert(split.trace(0)[0].label.includes('256'));
const weight=mm.steps.find(s=>s.settings.sourceId==='atom:poe-7:expert-weight-0');assert.equal(weight.settings.activation,'exp-negative');assert(!weight.formula.includes('sigma^2'));
let total=0;
for(const [name,run] of [['mmhvae',mm],...['sharedPrivateModel','pnpModel','ssdiffModel','metscModel','pigmentModel'].map(id=>[id,lib.makePaperRun(lib[id]())])]){
 assert.deepEqual(run.sourceGraph.warnings,[]);assert.equal(new Set(run.steps.map(s=>s.id)).size,run.steps.length);
 const tensors=new Map(run.tensors.map(t=>[t.id,t])),produced=new Set();
 for(const t of run.tensors){assert(t.window);assert.equal(t.values.length,t.window.coordinates.length);assert.equal(t.values.length,t.window.positions.length);assert(t.window.positions.flat().every(Number.isFinite));for(const c of t.window.coordinates)c.forEach((v,i)=>assert(v>=0&&(t.window.dimensions[i]===null||v<t.window.dimensions[i]),`${name}/${t.id}: coordinate bound`));}
 for(const s of run.steps){assert(s.settings.symbolic);katex.renderToString(s.formula,{throwOnError:true});for(const input of s.inputs)assert(tensors.has(input.id));for(const i of new Set([0,s.output.values.length-1]))for(const term of s.trace(i)){assert(tensors.has(term.tensor));assert(term.index===-1||term.index>=0&&term.index<tensors.get(term.tensor).values.length);if(term.factorTensor){assert(tensors.has(term.factorTensor));assert(term.factorIndex===-1||term.factorIndex>=0&&term.factorIndex<tensors.get(term.factorTensor).values.length)}}produced.add(s.output.id);}
 const graph=lib.layoutSourceScaffold(lib.buildScaffold(run,run.steps,true),run);assert(graph.nodes.every(n=>n.position.every(Number.isFinite)));
 const min=Math.min(...graph.nodes.map(n=>n.position[1])),left=Math.min(...graph.nodes.map(n=>n.position[0])),right=Math.max(...graph.nodes.map(n=>n.position[0]));
 console.log(name,run.steps.length,'root calculation steps; source coordinate and DAG checks passed',{height:Math.round(-min),width:Math.round(right-left)});total+=run.steps.length;
}
const ss=lib.makePaperRun(lib.ssdiffModel()),sum=ss.steps.find(s=>s.id==='encoder/block0/attention/softmax::row-sum');assert(sum);assert.equal(sum.output.window.dimensions.at(-1),1);assert.equal(sum.output.window.dimensions[0],16);
console.log(`Verified ${total} root mathematical stages across six papers.`);
