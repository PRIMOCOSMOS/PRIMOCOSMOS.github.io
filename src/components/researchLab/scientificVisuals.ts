import * as T from 'three'
import {createCrystalTensor,createArrowStream,type Position3} from '../mmhvae/crystalPrimitives'
import {makeMathVisual,type MathVisual,type MathFrame} from '../mmhvae/mathVisuals'
import type {MathStage} from '../mmhvae/mathematics'
import {scientificExample} from './operators'
const positions=(count:number,y:number):Position3[]=>Array.from({length:count},(_,i)=>[(i%4-1.5)*1.15,y+Math.floor(i/16)*.65,(Math.floor(i/4)%4-1.5)*1.15])
export function makeScientificVisual(spec:MathStage):MathVisual{
 if(!spec.operation.startsWith('lab:'))return makeMathVisual(spec)
 const group=new T.Group(),op=spec.operation.slice(4).split(':')[0],initial=scientificExample(spec,.42),input=createCrystalTensor(group,initial.a.length),output=createCrystalTensor(group,initial.out.length)
 const aPos=positions(initial.a.length,1.8),outPos=positions(initial.out.length,-1.6),streams=Array.from({length:op==='fft'||op==='ifft'?4:2},(_,i)=>createArrowStream(group,i?'#efc399':'#c4f3fa',.13,2))
 const hasOperand=['norm-affine','adain','latent-mask','spatial-mask','poe-precision','likelihood','kl','gradient','l1','complex-multiply','subtract','magnitude','spectral'].includes(op)
 const bPos:Position3[]=initial.b.map((_,i)=>[-4.5,(i%4-1.5)*.65,Math.floor(i/4)*.65-1]),operand=hasOperand?createCrystalTensor(group,initial.b.length):null
 let barrier:T.Mesh|undefined
 if(op==='detach'){
  barrier=new T.Mesh(new T.PlaneGeometry(5.2,.9),new T.MeshStandardMaterial({color:'#e6b490',transparent:true,opacity:.16,side:T.DoubleSide,depthWrite:false}));barrier.rotation.x=-Math.PI/2;group.add(barrier)
  const cross=new T.LineSegments(new T.BufferGeometry().setFromPoints([new T.Vector3(-.3,0,-.3),new T.Vector3(.3,0,.3),new T.Vector3(.3,0,-.3),new T.Vector3(-.3,0,.3)]),new T.LineBasicMaterial({color:'#efc399'}));group.add(cross)
 }
 let phaseCurve:T.Line|undefined
 if(op==='fft'||op==='ifft'){
  phaseCurve=new T.Line(new T.BufferGeometry(),new T.LineBasicMaterial({color:'#a4d5ee',transparent:true,opacity:.6}));group.add(phaseCurve)
 }
 group.userData.annotations=[{text:op==='fft'?'x_{h,w}':op==='ifft'?'k_{u,v}':'X',position:[0,3.4,0]},{text:op==='fft'?'\u005cmathcal F(x)':op==='ifft'?'\u005cmathcal F^{-1}(k)':'Y',position:[0,-2.6,0]}]
 if(op==='detach')group.userData.annotations.push({text:'\u005cpartial y/\u005cpartial x=0',position:[3.4,0,0]})
 const update=(f:MathFrame)=>{
  const data=scientificExample(spec,f.probe),i=Math.min(data.index,aPos.length-1),target=outPos[Math.min(data.index,outPos.length-1)]
  input.update(data.a,aPos,{focus:f.phase<.65?i:-1,active:op==='fft'||op==='ifft'||op==='spectral'?data.a.map((_,i)=>i):[i]})
  output.update(data.out,outPos,{focus:f.phase>=.65?data.index:-1,active:op==='detach'?[data.index]:[]})
  operand?.update(data.b,bPos,Math.min(i,bPos.length-1))
  group.userData.readout=data.readout
  if(op==='fft'||op==='ifft'){
   for(let k=0;k<4;k++){const source=(Math.floor(f.phase*4)%4)*4+k;streams[k].update([aPos[source],[k-1.5,0,0],target],f.phase)}
   phaseCurve!.geometry.dispose();phaseCurve!.geometry=new T.BufferGeometry().setFromPoints(Array.from({length:65},(_,j)=>{const theta=j/64*Math.PI*2*(1+Math.floor(f.probe*4));return new T.Vector3(j/64*4-2,Math.sin(theta)*.4,Math.cos(theta)*.4)}))
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
