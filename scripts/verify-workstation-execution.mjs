import assert from 'node:assert/strict';
import katex from 'katex';
import {MODULES,DEFAULT,moduleConfig,execute,executionGraph,executionCoordinates,streamedIndex,buildScaffold,stratifyScaffold,activationValue} from '../tmp/workstation-engine.mjs';
const close=(a,b)=>assert(Math.abs(a-b)<=1e-9*Math.max(1,Math.abs(a),Math.abs(b)),`${a} != ${b}`);
let stages=0;
for(const def of MODULES){
 const original=execute(def.id,moduleConfig(def.id,DEFAULT)),{run,blocks}=executionGraph(original);
 assert.equal(new Set(run.steps.map(s=>s.id)).size,run.steps.length,'Every arithmetic stage needs a unique address');
 for(const s of original.steps)assert(run.steps.some(v=>v.output.id===s.output.id),`${def.id}: hidden operator ${s.title}`);
 const graph=stratifyScaffold(buildScaffold(run,run.steps,true)),nodes=new Map(graph.nodes.map(n=>[n.tensor.id,n]));
 for(const e of graph.edges)assert(nodes.get(e.from).level<nodes.get(e.to).level,'Strict dependency levels');
 for(const block of blocks){
  for(const index of [0,block.source.output.values.length-1]){
   block.select(index);const coordinates=executionCoordinates(block,index);
   assert(coordinates.get(block.source.output.id).has(index));
   for(const s of block.steps){
    katex.renderToString(s.formula,{throwOnError:true,strict:'ignore'});
    assert.equal(s.output.shape.reduce((a,b)=>a*b,1),s.output.values.length);
    assert(s.output.values.every(Number.isFinite));
    for(const i of coordinates.get(s.output.id)??[]){assert(i>=0&&i<s.output.values.length);for(const t of s.trace(i)){assert(Number.isFinite(t.value));if(t.index>=0){const tensor=run.tensors.find(v=>v.id===t.tensor);assert(tensor,`${def.id}: missing ${t.tensor}`);assert.equal(t.value,tensor.values[t.index]);}}}
   }
   if(block.streamed)close(block.steps.at(-2).output.values.at(-1),block.source.output.values[index]);
   if(block.source.kind==='Linear'){const sum=block.steps[1].output,bias=block.source.inputs[2];close(sum.values[index]+(bias?.values[index%bias.values.length]??0),block.source.output.values[index]);}
   if(block.source.kind==='Sigmoid'||block.source.kind==='SiLU'){const gate=block.steps.at(-2).output.values[index],x=block.source.inputs[0].values[index];close(gate*(block.source.kind==='SiLU'?x:1),block.source.output.values[index]);}
   if(block.source.kind==='GELU')close(block.steps.at(-2).output.values[index]*block.source.inputs[0].values[index],block.source.output.values[index]);
  }
  stages+=block.steps.length;
 }
}
assert.deepEqual(Array.from({length:8},(_,pass)=>streamedIndex(4,pass,0,true)),[0,1,2,3,0,1,2,3]);
assert.equal(streamedIndex(4,9,2,false),2);
console.log(`Verified ${stages} concrete arithmetic stages across all ${MODULES.length} modules: unique IDs, exact coordinates, formulas, full dependency DAGs, streaming registers and output sweeps.`);

for(const kind of ['ReLU','ReLU6','LeakyReLU','Sigmoid','Tanh','SiLU','GELU','Hardswish','Hardsigmoid','Softplus']){const values=[-12,-6,-3,0,1,3,6,12],r=execute(kind,{...DEFAULT,dim:8},values);r.output.values.forEach((v,i)=>close(v,activationValue(kind,values[i])));const graph=executionGraph(r,r.steps,true);assert.equal(graph.run.steps.length,1,'Activation must preserve its layer instead of replacing it with prose stages');}
for(const batch of [1,2]){const r=execute('mlp',{...DEFAULT,batch,dim:6,hidden:8,out:4,layers:2}),graph=executionGraph(r,r.steps,true);assert.equal(graph.run.steps.length,5);let edges=0;for(const s of graph.run.steps.filter(s=>s.kind==='Linear'))for(let i=0;i<s.output.values.length;i++)edges+=s.trace(i).filter(t=>t.factorTensor===s.inputs[1].id).length;assert.equal(edges,batch*(6*8+8*8+8*4));}
console.log('Neural layer silhouettes, all MLP weights across batches and ten exact activation transfer functions passed.');

// Equal dependency depths do not license mixing input/kernel/bias/compute planes.
for(const id of ['conv1d','conv2d','mlp','vae-head','vae-mlp','vae-poe']){
 const r=execute(id,DEFAULT),g=executionGraph(r,r.steps,true),graph=stratifyScaffold(buildScaffold(g.run,g.run.steps,true));
 const essential=graph.nodes.filter(n=>n.tensor.id===r.input.id||n.tensor.parameter||n.tensor.constant);
 for(const a of essential)for(const b of graph.nodes)if(a!==b&&a.position[1]===b.position[1])assert.equal(a.tensor.parameter,b.tensor.parameter,'Input and parameter tiers must be distinct');
 if(id.startsWith('conv')){const inputs=graph.nodes.filter(n=>n.external);assert.equal(new Set(inputs.map(n=>n.position[1])).size,inputs.length,'Convolution input/kernel/bias/padding occupy separate tiers');}
 if(id==='vae-head'){const heads=graph.nodes.filter(n=>n.step?.kind==='Linear');assert.equal(new Set(heads.map(n=>n.position[1])).size,2,'Mean and log-variance have distinct role tiers');}
 if(id==='vae-mlp'){const heads=graph.nodes.filter(n=>n.step?.kind==='Linear'&&n.step.group.startsWith('后验参数')).sort((a,b)=>a.level-b.level),std=graph.nodes.find(n=>n.tensor.name==='标准差 σ');assert.equal(heads[1].level-heads[0].level,1,'Gaussian heads stay adjacent to their shared feature stage');assert(heads.every(n=>n.level<std.level),'Both heads precede the standard-deviation transform');}
}
console.log('Heterogeneous input/parameter/math tiers and distinct Gaussian heads passed.');
