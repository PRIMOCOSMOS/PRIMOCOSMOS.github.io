import * as T from 'three'
import {createCrystalTensor,createArrowStream,type Position3} from '../mmhvae/crystalPrimitives'
import {makeMathVisual,type MathVisual,type MathFrame} from '../mmhvae/mathVisuals'
import type {MathStage} from '../mmhvae/mathematics'
import {scientificExample} from './operators'
const positions=(count:number,y:number):Position3[]=>Array.from({length:count},(_,i)=>[(i%4-1.5)*1.15,y+Math.floor(i/16)*.65,(Math.floor(i/4)%4-1.5)*1.15])
export function makeScientificVisual(spec:MathStage):MathVisual{
 if(!/^(lab|core):/.test(spec.operation))return makeMathVisual(spec)
 const group=new T.Group(),op=spec.operation.split(':')[1],initial=scientificExample(spec,.42),input=createCrystalTensor(group,initial.a.length),output=createCrystalTensor(group,initial.out.length)
 const aPos=positions(initial.a.length,1.8),outPos=positions(initial.out.length,-1.6),streams=Array.from({length:op==='fft'||op==='ifft'?4:2},(_,i)=>createArrowStream(group,i?'#efc399':'#c4f3fa',.13,2))
 if(op==='split')outPos.forEach((pos,i)=>{const mode=spec.operation.split(':')[2],chunk=mode==='unequal'?(i===6?1:0):Math.floor(i/(mode==='3'?4:4));pos[0]=(i%4-1.5)*.85+(chunk-1)*2.1;pos[2]=(Math.floor(i/4)%2)*1.1+chunk*1.1;pos[1]=-1.6-chunk*.5})
 const hasOperand=['norm-affine','adain','latent-mask','spatial-mask','poe-precision','likelihood','kl','gradient','l1','complex-multiply','subtract','magnitude','spectral','scores','weighted','vq','noise','ddim','tensor-signal','rotation','angular','dropout'].includes(op)
 const bPos:Position3[]=positions(initial.b.length,0).map(p=>[p[0]-5.3,p[1],p[2]]),operand=hasOperand?createCrystalTensor(group,initial.b.length):null
 let barrier:T.Mesh|undefined
 if(op==='detach'){
  barrier=new T.Mesh(new T.PlaneGeometry(5.2,.9),new T.MeshStandardMaterial({color:'#e6b490',transparent:true,opacity:.16,side:T.DoubleSide,depthWrite:false}));barrier.rotation.x=-Math.PI/2;group.add(barrier)
  const cross=new T.LineSegments(new T.BufferGeometry().setFromPoints([new T.Vector3(-.3,0,-.3),new T.Vector3(.3,0,.3),new T.Vector3(.3,0,-.3),new T.Vector3(-.3,0,.3)]),new T.LineBasicMaterial({color:'#efc399'}));group.add(cross)
 }
 const geometric=['angular','rotation','tensor-signal','vq'].includes(op),origin=new T.Vector3(1.8,0,0)
 let directional:T.ArrowHelper|undefined,reference:T.ArrowHelper|undefined,codeMarkers:ReturnType<typeof createCrystalTensor>|undefined
 if(geometric){
  const geometry=new T.SphereGeometry(1.4,24,16),material=new T.MeshPhysicalMaterial({color:'#a6d9e9',transparent:true,opacity:.09,roughness:.12,metalness:.15,depthWrite:false}),shell=new T.Mesh(geometry,material);shell.position.copy(origin)
  if(op==='tensor-signal')shell.scale.set(1.6,.78,.78)
  group.add(shell);const wire=new T.LineSegments(new T.WireframeGeometry(geometry),new T.LineBasicMaterial({color:'#7cbed5',transparent:true,opacity:.13}));wire.position.copy(origin);wire.scale.copy(shell.scale);group.add(wire)
  directional=new T.ArrowHelper(new T.Vector3(1,0,0),origin,2.3,'#edcfaa',.4,.2);reference=new T.ArrowHelper(new T.Vector3(1,0,0),origin,2.3,'#a8eafa',.4,.2);group.add(directional,reference)
  if(op==='vq'){codeMarkers=createCrystalTensor(group,4);reference.visible=false}
 }
 let phaseCurve:T.Line|undefined,lastCurveProbe=-1
 if(op==='fft'||op==='ifft'){
  phaseCurve=new T.Line(new T.BufferGeometry(),new T.LineBasicMaterial({color:'#a4d5ee',transparent:true,opacity:.6}));group.add(phaseCurve)
 }
 group.userData.annotations=[{text:op==='fft'?'x_{h,w}':op==='ifft'?'k_{u,v}':'X',position:[0,3.4,0]},{text:op==='fft'?'\u005cmathcal F(x)':op==='ifft'?'\u005cmathcal F^{-1}(k)':'Y',position:[0,-2.6,0]}]
 if(op==='detach')group.userData.annotations.push({text:'\u005cpartial y/\u005cpartial x=0',position:[3.4,0,0]})
 const update=(f:MathFrame)=>{
  if(directional){const t=f.probe*Math.PI;directional.setDirection(new T.Vector3(Math.cos(t),0,(op==='rotation'?-1:1)*Math.sin(t)));if(op==='vq'){const codes:Position3[]=[[-.8,.1,-.2],[-.2,.4,-.5],[.4,0,.1],[.9,.5,-.4]],target=[f.probe*2-1,.25,-.3],dist=codes.map(c=>c.reduce((s,v,i)=>s+(v-target[i])**2,0)),nearest=dist.indexOf(Math.min(...dist));codeMarkers!.update(dist,codes.map(c=>[c[0]*2+origin.x,c[1]*2,c[2]*2]),{focus:nearest,active:[nearest]});directional.position.set(target[0]*2+origin.x,target[1]*2,target[2]*2);const delta=new T.Vector3(...codes[nearest]).sub(new T.Vector3(...target)).multiplyScalar(2);directional.setDirection(delta.clone().normalize());directional.setLength(Math.max(.15,delta.length()),.18,.1)}}
  const data=scientificExample(spec,f.probe),i=Math.min(data.index,aPos.length-1),target=outPos[Math.min(data.index,outPos.length-1)]
  const row=Math.floor(i/4),col=i%4,all=['fft','ifft','spectral','vq','l1norm'].includes(op)
  const active=all?data.a.map((_,i)=>i):['scores','weighted','softmax','layernorm'].includes(op)?[0,1,2,3].map(k=>row*4+k):[i]
  input.update(data.a,aPos,{focus:f.phase<.65?i:-1,active})
  output.update(data.out,outPos,{focus:f.phase>=.65?data.index:-1,active:op==='detach'?[data.index]:[]})
  operand?.update(data.b,bPos,{focus:Math.min(i,bPos.length-1),active:op==='scores'?[0,1,2,3].map(k=>col*4+k):op==='weighted'?[0,1,2,3].map(k=>k*4+col):all?data.b.map((_,i)=>i):[]})
  group.userData.readout=data.readout
  if(op==='fft'||op==='ifft'){
   for(let k=0;k<4;k++){const source=(Math.floor(f.phase*4)%4)*4+k;streams[k].update([aPos[source],[k-1.5,0,0],target],f.phase)}
   if(lastCurveProbe!==f.probe){lastCurveProbe=f.probe;phaseCurve!.geometry.dispose();phaseCurve!.geometry=new T.BufferGeometry().setFromPoints(Array.from({length:65},(_,j)=>{const theta=j/64*Math.PI*2*(1+Math.floor(f.probe*4));return new T.Vector3(j/64*4-2,Math.sin(theta)*.4,Math.cos(theta)*.4)}))}
  }else if(op==='detach'){
   streams[0].update([aPos[i],target],f.phase)
   streams[1].update([target,[target[0],-.08,target[2]]],f.phase)
  }else if(op==='gradient'){
   streams[0].update([target,[0,0,1],aPos[i]],f.phase)
   streams[1].update([aPos[i],target],(f.phase+.5)%1)
  }else{
   streams[0].update([aPos[i],target],f.phase)
   streams[1].group.visible=!!operand
   if(operand)streams[1].update([bPos[Math.min(i,bPos.length-1)],[0,0,0],target],f.phase)
  }
 }
 return {group,update}
}
