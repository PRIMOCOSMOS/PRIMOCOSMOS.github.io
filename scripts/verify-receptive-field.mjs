import assert from 'node:assert/strict';
import {build} from 'esbuild';
await build({stdin:{contents:"export * from './src/components/workstation/models'; export * from './src/components/workstation/engine'; export * from './src/components/workstation/scaffold'; export * from './src/components/workstation/catalog'; export * from './src/components/workstation/generative'; export * from './src/components/workstation/receptiveField'; export * from './src/components/mmhvae/sourceTensor';",resolveDir:process.cwd()},bundle:true,platform:'node',format:'esm',packages:'external',outfile:'tmp/receptive-field-check.mjs'});
const {execute,DEFAULT,moduleConfig,tensorLayout,receptiveField,coords,sourceTensor}=await import('../tmp/receptive-field-check.mjs');
let tested=0;
for(const id of ['conv1d','conv2d','conv3d','transpose2d']){
 const run=execute(id,moduleConfig(id,{...DEFAULT,kernel:3,padding:1,dilation:1,stride:1}));
 for(const s of run.steps.filter(s=>s.kind.startsWith('Conv')))for(const out of [0,Math.floor(s.output.values.length/2),s.output.values.length-1]){
  const field=receptiveField(s,out),input=s.inputs[0],output=coords(out,s.output.shape),group=Math.floor(output[1]/(s.output.shape[1]/Number(s.settings.groups))),expected=s.trace(out).filter(t=>t.tensor===input.id&&t.index>=0).map(t=>t.index);
  assert.deepEqual(field.indices,[...new Set(expected)]);
  for(const i of field.indices){const c=coords(i,input.shape);assert.equal(c[0],output[0]);assert.equal(Math.floor(c[1]/(input.shape[1]/Number(s.settings.groups))),group);}
  assert.deepEqual(field.output,output);tested++;
 }
 if(id==='conv3d')for(const t of run.tensors.filter(t=>t.shape.length===5)){
  const l=tensorLayout(t);assert(l.volume);assert.equal(l.positions.length,t.values.length);assert.equal(new Set(l.positions.map(p=>p.join(','))).size,t.values.length);
  assert.equal(new Set(l.positions.map(p=>p[1])).size,t.shape[2],'Depth occupies a real third spatial axis');
  const [,,d,h,w]=t.shape;for(let i=0;i<t.values.length;i++){const c=coords(i,t.shape);assert.equal(l.positions[i][1],(c[2]-(d-1)/2)*.82);if(c[4]+1<w)assert(Math.abs(l.positions[i+1][0]-l.positions[i][0]-.57)<1e-10);if(c[3]+1<h)assert(Math.abs(l.positions[i+w][2]-l.positions[i][2]-.57)<1e-10);}
 }
}
const image=sourceTensor('tensor','B × 16 × 32 × 32','#aaccee'),volume=sourceTensor('tensor','B × 16 × 8 × 32 × 32','#aaccee'),tokens=sourceTensor('tensor','B × G × 1536','#aaccee');
assert.equal(image.userData.volume,false);assert.equal(volume.userData.volume,true);assert.equal(tokens.userData.volume,false);assert.equal(tokens.userData.tensorAxes[1],'G');
console.log(`Verified ${tested} exact receptive fields, batch/group boundaries, every Conv3D coordinate, and source tensor image/volume/token distinctions.`);

const grouped=execute('conv3d',{...DEFAULT,batch:2,channels:4,out:4,spatial:7,depth:5,kernel:3,stride:2,padding:2,dilation:2,groups:2});
const conv=grouped.steps.find(s=>s.kind==='Conv3d'),oi=(((1*4+3)*3+1)*4+1)*4+1,expected=[];
for(const channel of [2,3])for(const d of [0,2,4])for(const h of [0,2,4])for(const w of [0,2,4])expected.push((((1*4+channel)*5+d)*7+h)*7+w);
assert.deepEqual(receptiveField(conv,oi).indices,expected,'Independent dilated/strided/grouped Conv3D stencil');
assert.equal(receptiveField(conv,oi).indices.length,54);
assert.equal(sourceTensor('tensor','128 × 32³','#aaccee').userData.volume,true);
assert.deepEqual(sourceTensor('network','B × (G+1) × 1536','#aaccee').userData.tensorAxes,['B','(G+1)','1536']);
console.log('Independent 54-voxel dilated/grouped/batched receptive field and symbolic source axes passed.');
