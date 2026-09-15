import assert from 'node:assert/strict';
import {execute,DEFAULT,moduleConfig,MODULES,computation,exactRectangle,tensorLayout} from '../tmp/workstation-engine.mjs';
const close=(a,b)=>assert(Math.abs(a-b)<1e-10,`${a} != ${b}`);
for(const t of [0,1,999]){
 const c={...DEFAULT,timeStep:t,previousStep:-1},r=execute('diffusion-reverse',c),x0=r.tensors.find(v=>v.name==='裁剪预测到 [−1,1]'),ab=r.tensors.find(v=>v.name==='α 累乘');
 const sample=r.steps.at(-1),mean=sample.inputs[0],random=sample.inputs[1];
 if(t===0){assert(random.values.every(v=>v===0));for(let i=0;i<r.output.values.length;i++)close(r.output.values[i],x0.values[i]);}
 else assert(random.values.some(v=>v!==0));
 const d=execute('diffusion-ddim',{...moduleConfig('diffusion-ddim',c),eta:0});assert(d.steps.at(-1).inputs[1].values.every(v=>v===0));
 const clean=d.tensors.find(v=>v.name==='预测干净样本');for(let i=0;i<d.output.values.length;i++)close(d.output.values[i],clean.values[i]);
}
for(const allNegatives of [false,true]){const r=execute('cut-nce',{...DEFAULT,batch:2,allNegatives,temperature:.01});assert(r.output.values.every(Number.isFinite));const s=r.steps.find(s=>s.settings.operation==='mask-diagonal');assert(s.output.values.some(v=>v===-10));assert(r.steps.some(s=>s.settings.operation==='detach'));}
for(const def of MODULES){const r=execute(def.id,moduleConfig(def.id,DEFAULT));for(const s of r.steps)for(const i of [0,s.output.values.length-1]){const c=computation(s,i);for(const t of [c.products,c.prefix,c.sums].filter(Boolean)){assert.equal(t.values.length,t.shape.reduce((a,b)=>a*b,1));assert(t.values.every(Number.isFinite),`${def.id}/${s.title} intermediate finite`)}if(s.kind==='MaxPool2d')close(c.prefix.values.at(-1),s.output.values[i]);}}
console.log('DDPM t=0/noise boundaries, DDIM eta=0/s=-1, PatchNCE scopes and every derived 3D arithmetic stage passed.');

assert.deepEqual(exactRectangle(48),[6,8]);assert.deepEqual(exactRectangle(8),[2,4]);for(const count of [6,8,24,37,48,1000]){const [rows,cols]=exactRectangle(count);assert.equal(rows*cols,count);const layout=tensorLayout({id:'layout-check',name:'vector',shape:[count],values:Array(count).fill(0)});assert.equal(layout.positions.length,count)}console.log('Exact rectangular packing verified, including 48 = 6 × 8 without missing cells.');
