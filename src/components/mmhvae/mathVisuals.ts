import * as T from 'three'
import {activateValue,bilinear,convolutionDemo,demoValues,fusionValues,mean,normalSamples,uniformSamples,normalizeValues,type MathStage} from './mathematics'

type V=[number,number,number]
export interface MathFrame{phase:number;probe:number;temperature:number}
export interface MathVisual{group:T.Group;update:(frame:MathFrame)=>void}
const blue=new T.Color('#72cbe5'),warm=new T.Color('#e8af77'),white=new T.Color('#efffd5')
const v=(p:V)=>new T.Vector3(...p)
const gridPositions=(n:number,channels=1,width=5.4):V[]=>Array.from({length:n*n*channels},(_,i)=>[((i%n)/(n-1||1)-.5)*width,Math.floor(i/(n*n))*.78,((Math.floor(i/n)%n)/(n-1||1)-.5)*width])

/** Numeric teaching models. No operator glyphs are used inside a mathematical view. */
export function makeMathVisual(spec:MathStage):MathVisual{
 const group=new T.Group(),updates:((f:MathFrame)=>void)[]=[],stage=spec.stage
 const line=(points:V[],color='#678da0',opacity=.55)=>{const l=new T.Line(new T.BufferGeometry().setFromPoints(points.map(v)),new T.LineBasicMaterial({color,transparent:true,opacity}));group.add(l);return l}
 const orb=(color='#edffcf',r=.13)=>{const m=new T.Mesh(new T.SphereGeometry(r,12,8),new T.MeshStandardMaterial({color,emissive:color,emissiveIntensity:.18}));group.add(m);return m}
 const wireBox=(w:number,h:number,d:number,color='#dfffc0')=>{const m=new T.LineSegments(new T.EdgesGeometry(new T.BoxGeometry(w,h,d)),new T.LineBasicMaterial({color,transparent:true,opacity:.9}));group.add(m);return m}
 const tiles=(count:number,size=.4)=>{
  const mesh=new T.InstancedMesh(new T.BoxGeometry(size,.12,size),new T.MeshStandardMaterial({roughness:.45,metalness:.12}),count);mesh.instanceMatrix.setUsage(T.DynamicDrawUsage);group.add(mesh);const obj=new T.Object3D()
  return (values:number[],positions:V[],selected=-1)=>{for(let i=0;i<count;i++){obj.position.set(...positions[i]);obj.scale.set(1,1+Math.min(Math.abs(values[i]??0),3)*3,1);obj.updateMatrix();mesh.setMatrixAt(i,obj.matrix);mesh.setColorAt(i,i===selected?white:(values[i]??0)<0?warm.clone().lerp(new T.Color('#543e37'),.5-Math.min(.5,Math.abs(values[i])*.35)):blue.clone().lerp(new T.Color('#193743'),.5-Math.min(.5,Math.abs(values[i])*.35)))}mesh.instanceMatrix.needsUpdate=true;if(mesh.instanceColor)mesh.instanceColor.needsUpdate=true;mesh.computeBoundingSphere()}
 }
 const connect=(a:V,b:V,color?:string)=>line([a,[(a[0]+b[0])/2,(a[1]+b[1])/2+.5,(a[2]+b[2])/2],b],color)
 const particle=orb(),cursor=wireBox(.65,.7,.65)
 const flow=(positions:V[],phase:number)=>{const t=(phase%1)*(positions.length-1),i=Math.floor(t);particle.position.lerpVectors(v(positions[i]),v(positions[Math.min(i+1,positions.length-1)]),t-i)}
 

 if(spec.kind==='convolution'||spec.kind==='depthwise'){
  const {k,n,channels,values,weights,outN,output}=convolutionDemo(spec),positions=gridPositions(n,channels)
  const last=spec.weightNorm?3:2
  const set=tiles(stage===last?output.length:values.length,Math.min(.5,4.7/n)),outPositions=gridPositions(outN,spec.kind==='depthwise'?channels:1)
  const kernelSet=tiles(k*k,.22),kernelPos=gridPositions(k,1,Math.min(2.8,k*.42)).map(p=>[p[0],2.8,p[2]] as V),window=wireBox(k*5.4/(n-1),.5,k*5.4/(n-1))
  if(stage===last){kernelPos.forEach(p=>p[0]-=4);connect([-4,2.8,0],[0,0,0]);window.visible=false}
  updates.push(f=>{const idx=Math.min(outN*outN-1,Math.floor(f.probe*outN*outN)),x=(idx%outN)*spec.stride-spec.padding+(k-1)/2,y=Math.floor(idx/outN)*spec.stride-spec.padding+(k-1)/2;window.position.set((x/(n-1)-.5)*5.4,.25,(y/(n-1)-.5)*5.4)
   const productStage=stage===(spec.weightNorm?2:1),current=stage===last?output:productStage?values.map((val,i)=>{const row=Math.floor(i/n)%n,col=i%n,wy=row-(Math.floor(idx/outN)*spec.stride-spec.padding),wx=col-((idx%outN)*spec.stride-spec.padding);return wy>=0&&wy<k&&wx>=0&&wx<k?val*weights[Math.floor(i/(n*n))*k*k+wy*k+wx]:0}):values
   set(current,stage===last?outPositions:positions,stage===last?idx:Math.max(0,Math.min(values.length-1,Math.round(y)*n+Math.round(x))))
   group.userData.readout=`y_{0,${Math.floor(idx/outN)},${idx%outN}}=${output[idx].toFixed(3)}`;kernelSet(weights.slice(0,k*k),kernelPos,Math.floor(f.phase*k*k)% (k*k));cursor.visible=false;flow([[window.position.x,2.8,window.position.z],[0,1,0],[0,-.8,0]],f.phase)
  })
 }else if(spec.kind==='linear'||spec.kind==='attention'||spec.kind==='composition'){
  const n=6,m=4,x=demoValues.slice(0,n),rawW=Array.from({length:n*m},(_,i)=>Math.sin(i*1.4)*.55),W=spec.weightNorm?rawW.map((w,i)=>w/Math.hypot(...rawW.slice(Math.floor(i/n)*n,(Math.floor(i/n)+1)*n))):rawW,positions:Array<V>=Array.from({length:n*m},(_,i)=>[(i%n-2.5)*.8,0,(Math.floor(i/n)-1.5)*1.05]),set=tiles(n*m,.62),input=tiles(n,.38),output=tiles(m,.5)
  const xp:V[]=x.map((_,i)=>[(i-2.5)*.8,1.5,-3]),yp:V[]=Array.from({length:m},(_,i)=>[4.1,0,(i-1.5)*1.05])
  for(let i=0;i<m;i++)for(let j=0;j<n;j++)connect(positions[i*n+j],yp[i],i%2?'#ad8265':'#648c9a')
  const y=Array.from({length:m},(_,i)=>x.reduce((s,a,j)=>s+a*W[i*n+j],.1))
  updates.push(f=>{const row=Math.min(m-1,Math.floor(f.probe*m));group.userData.readout=`y_${row}=${y[row].toFixed(3)}`;set(stage===1?W.map((a,i)=>a*x[i%n]):W,positions,row*n+Math.floor(f.phase*n)%n);input(x,xp);output(y,yp,row);cursor.position.set(0,.3,(row-1.5)*1.05);cursor.scale.set(8,1,1.5);flow([xp[Math.floor(f.phase*n)%n],positions[row*n+Math.floor(f.phase*n)%n],yp[row]],f.phase)})
 }else if(spec.kind==='activation'){
  const op=spec.operation,range=op.includes('clamp')?20:4,normalizer=op==='clamp-exp'?Math.exp(10*Math.tanh(2)):op==='clamp'?10:op==='sigmoid'||op==='tanh'?1:4
  for(let c=0;c<3;c++){const z=(c-1)*1.25;line([[-3,-1.7,z],[3,-1.7,z]],'#617f91');line([[0,-1.7,z],[0,2,z]],'#617f91');line(Array.from({length:101},(_,i)=>{const x=(i/100*2-1)*range;return [x/range*3,activateValue(x,op)/normalizer*2,z] as V}),'#9de4dc',.95)}
  const a=orb('#e8af77'),b=orb('#92d9ec');cursor.visible=false
  const set=tiles(12,.55),pos=gridPositions(2,3,1.2).map(p=>[p[0]+4,p[1]-.5,p[2]] as V)
  updates.push(f=>{const x=(f.probe*2-1)*range,y=activateValue(x,op),z=Math.sin(f.phase*2*Math.PI)*1.25;group.userData.readout=`f(${x.toFixed(2)})=${y.toPrecision(3)}`;a.position.set(x/range*3,-1.7,z);b.position.set(x/range*3,y/normalizer*2,z);particle.position.lerpVectors(a.position,b.position,f.phase);set(demoValues.slice(0,12).map(a=>stage===0?a:activateValue(a*range/2,op)/Math.max(1,normalizer)),pos,Math.floor(f.phase*12)%12)})
 }else if(spec.kind==='fusion'){
  cursor.visible=false
  const positions:V[]=[[-2.2,0,-.8],[0,0,-.8],[2.2,0,-.8]],bars=positions.map(p=>{const m=new T.Mesh(new T.BoxGeometry(.7,1,.7),new T.MeshStandardMaterial({color:'#83cfe0'}));group.add(m);m.position.set(...p);return m})
  const output=tiles(2,.8),outPos:V[]=[[-.7,-1.4,2.5],[.7,-1.4,2.5]]
  for(const p of positions)connect(p,[0,-1.4,2.5]);line([[-3.2,0,-.8],[3.2,0,-.8]],'#bad1d9')
  updates.push(f=>{const data=fusionValues(.45+f.probe*1.35,f.temperature),w=data.weights.reduce((a,b)=>a+b),m=data.contributions.reduce((a,b)=>a+b),values=spec.operation==='weighted-mean'?stage===0?data.mus:data.contributions:spec.operation==='normalize'?[w,m,0]:data.weights
   group.userData.readout=spec.operation==='weight'?`w=${w.toFixed(3)}`:spec.operation==='weighted-mean'?`m=${m.toFixed(3)}`:`μ=${data.mu.toFixed(3)},s=${data.baseScale.toFixed(3)}`;bars.forEach((bar,i)=>{const val=values[i];bar.scale.y=Math.max(.02,Math.abs(val));bar.position.y=val/2;(bar.material as T.MeshStandardMaterial).color.copy(val<0?warm:blue)})
   output(spec.operation==='normalize'?[data.mu,data.baseScale]:spec.operation==='weighted-mean'?[m,0]:[w,0],outPos);flow([positions[Math.floor(f.phase*3)%3],[0,1.4,.8],[0,-1.4,2.5]],f.phase)
  })
 }else if(spec.kind==='distribution'){
  cursor.visible=false
  const count=32,geo=new T.PlaneGeometry(5.8,5.8,count,count);geo.rotateX(-Math.PI/2);const mesh=new T.Mesh(geo,new T.MeshStandardMaterial({color:'#8dd2e5',side:T.DoubleSide,transparent:true,opacity:.62,roughness:.4,metalness:.1}));group.add(mesh)
  const base=geo.attributes.position.array.slice(),pos=geo.attributes.position as T.BufferAttribute
  const contours=[1,1.6,2.1].map(r=>{const ring=new T.Mesh(new T.RingGeometry(r-.015,r+.015,56),new T.MeshBasicMaterial({color:'#a6daea',side:T.DoubleSide}));ring.rotation.x=-Math.PI/2;group.add(ring);return ring})
  line([[-3,0,0],[3,0,0]]);line([[0,0,-3],[0,0,3]]);const bars=tiles(3,.65),bp:V[]=[[-2,-.8,3.8],[0,-.8,3.8],[2,-.8,3.8]];let previous=-1
  updates.push(f=>{const s=.45+f.probe*1.35;group.userData.readout=`s=${s.toFixed(2)},s²=${(s*s).toFixed(3)}`;if(previous!==f.probe){previous=f.probe;for(let i=0;i<pos.count;i++){const x=base[i*3],z=base[i*3+2];pos.setY(i,Math.exp(-.5*(x*x/s**2+z*z/.65**2))/(2*Math.PI*s*.65)*8)}pos.needsUpdate=true;geo.computeVertexNormals();contours.forEach((ring,i)=>{ring.scale.set(s,.65,1);ring.position.y=Math.exp(-.5*[1,1.6,2.1][i]**2)/(2*Math.PI*s*.65)*8});bars([0,s,s*s],bp)}particle.position.set((f.phase*2-1)*2.6,.05,0)})
 }else if(spec.kind==='sampling'){
  const geometry=new T.BufferGeometry().setFromPoints(normalSamples.map(p=>v(p as V))),points=new T.Points(geometry,new T.PointsMaterial({size:.095,color:'#a4ddeb',sizeAttenuation:true}));group.add(points);line([[-3,0,0],[3,0,0]]);line([[0,-2,0],[0,2,0]]);line([[0,0,-3],[0,0,3]]);cursor.visible=false
  const from=orb('#e9b58a',.09),to=orb('#eaffb9',.15),pos=geometry.attributes.position as T.BufferAttribute
  updates.push(f=>{const mix=stage===0?0:stage===1?f.phase:1;for(let i=0;i<normalSamples.length;i++){const source=uniformSamples[i].slice(0,3).map(x=>(x-.5)*4),target=normalSamples[i];pos.setXYZ(i,...source.map((x,j)=>x+(target[j]-x)*mix) as V)}pos.needsUpdate=true;geometry.computeBoundingSphere();const index=Math.min(normalSamples.length-1,Math.floor(f.probe*normalSamples.length));from.position.set(...uniformSamples[index].slice(0,3).map(x=>(x-.5)*4) as V);to.position.set(...normalSamples[index] as V);group.userData.readout=`ε_1=${normalSamples[index][0].toFixed(3)},ε_2=${normalSamples[index][1].toFixed(3)}`;particle.position.lerpVectors(from.position,to.position,mix)})
 }else if(spec.kind==='interpolation'){
  const values=demoValues.slice(0,16),inputPos=gridPositions(4),outPos=gridPositions(8),output=Array.from({length:64},(_,i)=>bilinear(values,4,(i%8+.5)/2-.5,(Math.floor(i/8)+.5)/2-.5).value),set=tiles(stage===2?64:16,stage===2?.48:.7)
  const lines=Array.from({length:4},()=>line([[0,0,0],[0,0,0]],'#dcbd86',.8));cursor.visible=false
  updates.push(f=>{const i=Math.min(63,Math.floor(f.probe*64)),x=(i%8+.5)/2-.5,y=(Math.floor(i/8)+.5)/2-.5,result=bilinear(values,4,x,y),target:V=[(Math.max(0,Math.min(3,x))/3-.5)*5.4,1.8,(Math.max(0,Math.min(3,y))/3-.5)*5.4];set(stage===2?output:values,stage===2?outPos:inputPos,stage===2?i:result.ids[0]);lines.forEach((l,j)=>{l.geometry.setFromPoints([v(inputPos[result.ids[j]]),v(target)]);(l.material as T.LineBasicMaterial).opacity=.15+.85*result.weights[j]});group.userData.readout=`y_{${Math.floor(i/8)},${i%8}}=${result.value.toFixed(3)}`;particle.position.set(...target);particle.scale.setScalar(.8+Math.abs(result.value)*.6)})
 }else if(spec.kind==='normalization'||spec.kind==='pooling'){
  const positions=gridPositions(4,3),data=demoValues.slice(0,48),sets=Array.from({length:3},(_,c)=>data.slice(c*16,(c+1)*16)),centered=sets.flatMap(a=>a.map(x=>x-mean(a))),norm=sets.flatMap(normalizeValues),means=sets.map(mean),set=tiles(stage===2&&spec.kind==='pooling'?3:48,.46)
  const centers:V[]=[[-3.7,0,0],[-3.7,.9,0],[-3.7,1.8,0]];centers.forEach(p=>orb('#eec89a').position.set(...p));for(let i=0;i<16;i++)connect(positions[i],centers[0])
  updates.push(f=>{const c=Math.min(2,Math.floor(f.probe*3));group.userData.readout=`μ_${c}=${means[c].toFixed(3)},v_${c}=${mean(sets[c].map(x=>(x-means[c])**2)).toFixed(3)}`;set(spec.kind==='pooling'?stage===2?means:data:stage===0?data:stage===1?centered.map(x=>x*x):norm,stage===2&&spec.kind==='pooling'?centers:positions,stage===2&&spec.kind==='pooling'?c:c*16+Math.floor(f.phase*16)%16);cursor.scale.set(9,1,9);cursor.position.set(0,c*.78+.25,0);flow([positions[c*16+Math.floor(f.phase*16)%16],centers[c]],f.phase)})
 }else if(spec.kind==='concatenation'||spec.kind==='split'){
  const positions=gridPositions(3,4),values=Array.from({length:36},(_,i)=>i<18?.25+(i%9)*.06:-.25-(i%9)*.06),set=tiles(36,.68)
  updates.push(f=>{const shift=spec.kind==='concatenation'?stage===0?2.6:stage===1?(1-f.phase)*2.6:0:stage===0?0:stage===1?f.phase*2.6:2.6,pos=positions.map((p,i)=>[p[0]+(i<18?-shift:shift),p[1],p[2]] as V),idx=Math.min(35,Math.floor(f.probe*36));set(values,pos,idx);cursor.position.set(...pos[idx]);particle.position.set(...pos[Math.floor(f.phase*36)%36])})
 }else if(spec.kind==='addition'||spec.kind==='multiplication'||spec.kind==='mask'){
  const a=demoValues.slice(0,16),b=Array.from({length:16},(_,i)=>spec.kind==='mask'?i%4===0?0:1:spec.kind==='multiplication'?.35+(i%4)*.15:demoValues[i+16]),out=a.map((v,i)=>spec.kind==='addition'?v+b[i]:spec.kind==='mask'?2*((v+1)/2*b[i])-1:v*b[i]),grid=gridPositions(4,1,2.9),left=grid.map(p=>[p[0]-2.1,1,p[2]] as V),right=grid.map(p=>[p[0]+2.1,1,p[2]] as V),result=grid.map(p=>[p[0],-1,p[2]] as V),inputA=tiles(16,.65),inputB=tiles(16,.65),output=tiles(16,.65)
  const l=connect(left[0],result[0]),r=connect(right[0],result[0]);updates.push(f=>{const i=Math.min(15,Math.floor(f.probe*16));group.userData.readout=`${a[i].toFixed(2)}${spec.kind==='addition'?'+':spec.kind==='mask'?',m=':'×'}${b[i].toFixed(2)} → ${out[i].toFixed(3)}`;inputA(a,left,i);inputB(b,right,i);output(out,result,i);l.geometry.setFromPoints([v(left[i]),v(result[i])]);r.geometry.setFromPoints([v(right[i]),v(result[i])]);cursor.position.set(...result[i]);flow([left[i],result[i],right[i]],f.phase)})
 }else if(spec.kind==='padding'){
  const input=demoValues.slice(0,16),grid=gridPositions(6,1,5.5),ref=(x:number)=>x<0?-x:x>3?6-x:x,output=grid.map((_,i)=>input[ref(Math.floor(i/6)-1)*4+ref(i%6-1)]),set=tiles(36,.72),mapping=line([[0,0,0],[0,0,0]],'#ebc48c')
  updates.push(f=>{const i=Math.min(35,Math.floor(f.probe*36)),y=Math.floor(i/6)-1,x=i%6-1,source=(ref(y)+1)*6+ref(x)+1;set(output,grid,i);cursor.position.set(...grid[i]);mapping.geometry.setFromPoints([v(grid[i]),v(grid[source])]);flow([grid[source],[grid[source][0],1.4,grid[source][2]],[grid[i][0],1.4,grid[i][2]],grid[i]],f.phase)})
 }else{
  const n=spec.kind==='broadcast'?27:48,positions=spec.kind==='broadcast'?gridPositions(3,3):gridPositions(4,3),values=Array.from({length:n},(_,i)=>spec.operation==='zeros'?0:spec.operation==='ones'?1:spec.kind==='broadcast'?.25+Math.floor(i/9)*.3:demoValues[i]),vectorPos:V[]=values.map((_,i)=>[((i%12)-5.5)*.46,Math.floor(i/12)*.65,0]),set=tiles(n,.42)
  updates.push(f=>{let pos=positions;if(spec.kind==='reshape'){const t=stage===1?f.phase:stage===2?1:0;pos=positions.map((p,i)=>p.map((a,k)=>spec.operation==='flatten'?a+(vectorPos[i][k]-a)*t:vectorPos[i][k]+(a-vectorPos[i][k])*t) as V)}if(spec.kind==='broadcast'&&stage<2)pos=positions.map((p,i)=>[p[0]*(stage===0?0:f.phase),p[1],p[2]*(stage===0?0:f.phase)] as V);const index=Math.min(n-1,Math.floor(f.probe*n));set(values,pos,index);cursor.position.set(...pos[index]);particle.position.set(...pos[Math.floor(f.phase*n)%n])})
 }
 return {group,update:frame=>updates.forEach(fn=>fn(frame))}
}
