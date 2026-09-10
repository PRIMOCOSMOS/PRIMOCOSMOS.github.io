import assert from 'node:assert/strict'
import {readFile} from 'node:fs/promises'
import ts from 'typescript'
import * as T from 'three'
const compile=async name=>ts.transpileModule(await readFile(new URL(`../src/components/mmhvae/${name}.ts`,import.meta.url),'utf8'),{compilerOptions:{target:ts.ScriptTarget.ES2022,module:ts.ModuleKind.ESNext}}).outputText
const url=source=>`data:text/javascript;base64,${Buffer.from(source).toString('base64')}`
const three=JSON.stringify(import.meta.resolve('three'))
const primitiveURL=url((await compile('crystalPrimitives')).replace("'three'",three).replace("'three/addons/utils/BufferGeometryUtils.js'",JSON.stringify(import.meta.resolve('three/addons/utils/BufferGeometryUtils.js'))))
const {createCrystalTensor,createArrowStream}=await import(primitiveURL)
const root=new T.Group(),tensor=createCrystalTensor(root,3,.42),positions=[[0,0,0],[1,0,0],[2,0,0]]
tensor.update([-100,0,100],positions,{focus:1,active:[0]})
assert.equal(tensor.body.count,3);assert.equal(tensor.lit.count,2)
assert(tensor.body.material.opacity<tensor.lit.material.opacity/4,'Active and passive opacity must be distinguishable')
const matrix=new T.Matrix4(),scale=new T.Vector3(),position=new T.Vector3(),rotation=new T.Quaternion()
for(let i=0;i<3;i++){tensor.body.getMatrixAt(i,matrix);matrix.decompose(position,rotation,scale);assert.deepEqual(scale.toArray(),[1,1,1]);assert.deepEqual(position.toArray(),positions[i])}
assert.deepEqual(['width','height','depth'].map(k=>tensor.body.geometry.parameters[k]),[.42,.42,.42])
tensor.update([0,0,0],positions,{focus:2,hidden:[0]});assert.equal(tensor.body.count,2);assert.equal(tensor.lit.count,1)
const stream=createArrowStream(root),curve=new T.CatmullRomCurve3([new T.Vector3(0,2,0),new T.Vector3(1,0,1),new T.Vector3(0,-2,0)])
for(const phase of [0,.23,.9]){stream.update(curve,phase);stream.arrows.forEach((arrow,i)=>{const t=.08+((phase+i/3)%1)*.84;assert(arrow.position.distanceTo(curve.getPointAt(t))<1e-10);assert(new T.Vector3(0,1,0).applyQuaternion(arrow.quaternion).dot(curve.getTangentAt(t))>.999999)})}
stream.update([[0,0,0],[0,0,0]],0);assert.equal(stream.group.visible,false,'Identity mapping must not imply transport')
const mathematicsURL=url(await compile('mathematics'))
const {makeMathVisual}=await import(url((await compile('mathVisuals')).replace("'three'",three).replace("'./mathematics'",JSON.stringify(mathematicsURL)).replace("'./crystalPrimitives'",JSON.stringify(primitiveURL))))
const kinds=['convolution','depthwise','linear','normalization','activation','pooling','concatenation','split','distribution','fusion','sampling','interpolation','addition','multiplication','reshape','broadcast','padding','tensor','mask']
for(const kind of kinds)for(const stage of [0,1,2,3]){
 const visual=makeMathVisual({kind,operation:kind==='fusion'?'normalize':kind==='activation'?'silu':kind==='reshape'?'flatten':kind,stage,kernel:3,stride:2,padding:1,weightNorm:stage===3,bias:true,reflect:true})
 for(const probe of [0,.7,1])for(const phase of [0,.35,.81,1]){
  visual.update({probe,phase,temperature:1})
  visual.group.traverse(o=>{
   assert(o.position.toArray().every(Number.isFinite),`${kind}: finite placement`)
   assert(o.geometry?.type!=='SphereGeometry',`${kind}: no data-flow balls`)
   if(o.isInstancedMesh&&o.parent.userData.crystalTensor)for(let i=0;i<o.count;i++){o.getMatrixAt(i,matrix);matrix.decompose(position,rotation,scale);assert(scale.toArray().every(x=>Math.abs(x-1)<1e-6),`${kind}: uniform tensor element`);assert(position.toArray().every(Number.isFinite))}
  })
 }
 if(kind==='addition'||kind==='multiplication'||kind==='mask'){
  const routes=visual.group.children.filter(o=>o.userData.directionalFlow)
  assert.equal(routes.length,2)
  for(const route of routes){const track=route.children.find(o=>o.isLine),p=track.geometry.attributes.position;assert(p.getY(0)>p.getY(p.count-1),'Both operands must flow downward to result')}
 }
}
console.log('Verified uniform crystal geometry and highlight states, arrow tangents, identity paths, and all 19 math families across stages/probe endpoints.')
