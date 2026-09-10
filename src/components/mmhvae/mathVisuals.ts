import * as T from 'three'
import {activateValue,bilinear,convolutionDemo,demoValues,fusionValues,mean,normalSamples,uniformSamples,normalizeValues,type MathStage} from './mathematics'
import {createCrystalTensor,createArrowStream} from './crystalPrimitives'

type V=[number,number,number]
export interface MathFrame{phase:number;probe:number;temperature:number}
export interface MathVisual{group:T.Group;update:(frame:MathFrame)=>void}
const v=(p:V)=>new T.Vector3(...p)
const gridPositions=(n:number,channels=1,width=5.4):V[]=>Array.from({length:n*n*channels},(_,i)=>[((i%n)/(n-1||1)-.5)*width,Math.floor(i/(n*n))*.78,((Math.floor(i/n)%n)/(n-1||1)-.5)*width])

/** Numeric teaching models. No operator glyphs are used inside a mathematical view. */
export function makeMathVisual(spec:MathStage):MathVisual{
 const group=new T.Group(),updates:((f:MathFrame)=>void)[]=[],stage=spec.stage
 const line=(points:V[],color='#678da0',opacity=.55)=>{const l=new T.Line(new T.BufferGeometry().setFromPoints(points.map(v)),new T.LineBasicMaterial({color,transparent:true,opacity}));group.add(l);return l}
 const marker=(color='#e3fbff')=>{const crystal=createCrystalTensor(group,1,.34);crystal.update([1],[[0,0,0]],0);crystal.lit.setColorAt(0,new T.Color(color));return crystal.group}
 const wireBox=(w:number,h:number,d:number,color='#dfffc0')=>{const m=new T.LineSegments(new T.EdgesGeometry(new T.BoxGeometry(w,h,d)),new T.LineBasicMaterial({color,transparent:true,opacity:.9}));group.add(m);return m}
 const tiles=(count:number,size=.42)=>createCrystalTensor(group,count,size).update
 const cursor=wireBox(.65,.7,.65),streams:ReturnType<typeof createArrowStream>[]=[]
 const flow=(positions:V[],phase:number,index=0,strength=1)=>{const stream=streams[index]??(streams[index]=createArrowStream(group,index%2?'#efc399':'#bbf0ff',.13,2));stream.update(positions,phase,strength)}
 

 if(spec.kind==='convolution'||spec.kind==='depthwise'){
  const {k,n,channels,values,weights,outN,output}=convolutionDemo(spec),positions=gridPositions(n,channels,4.8).map(p=>[p[0]+.5,p[1]+.6,p[2]] as V)
  const last=spec.weightNorm?3:2
  const cellSize=Math.min(.42,4.1/n),set=tiles(values.length,cellSize),outSet=tiles(output.length,cellSize),outPositions=gridPositions(outN,spec.kind==='depthwise'?channels:1,4.8).map(p=>[p[0]+.5,p[1]-2.5,p[2]] as V)
  const kernelSet=tiles(k*k,.3),kernelPos=gridPositions(k,1,Math.min(1.7,k*.42)).map(p=>[p[0]-4,p[1]+1.8,p[2]] as V),window=wireBox(k*4.8/(n-1),.5,k*4.8/(n-1))
  updates.push(f=>{const idx=Math.min(outN*outN-1,Math.floor(f.probe*outN*outN)),x=(idx%outN)*spec.stride-spec.padding+(k-1)/2,y=Math.floor(idx/outN)*spec.stride-spec.padding+(k-1)/2,term=Math.floor(f.phase*k*k*channels)%(k*k*channels),channel=Math.floor(term/(k*k)),local=term%(k*k),cycle=f.phase*k*k*channels%1;window.position.set((x/(n-1)-.5)*4.8+.5,channel*.78+.6,(y/(n-1)-.5)*4.8)
   const productStage=stage===(spec.weightNorm?2:1),current=stage===last?output:productStage?values.map((val,i)=>{const row=Math.floor(i/n)%n,col=i%n,wy=row-(Math.floor(idx/outN)*spec.stride-spec.padding),wx=col-((idx%outN)*spec.stride-spec.padding);return wy>=0&&wy<k&&wx>=0&&wx<k?val*weights[Math.floor(i/(n*n))*k*k+wy*k+wx]:0}):values
   const row0=Math.floor(idx/outN)*spec.stride-spec.padding,col0=idx%outN*spec.stride-spec.padding,reflect=(a:number)=>a<0?-a:a>=n?2*n-2-a:a,resolve=(row:number,col:number)=>{if(spec.reflect){row=reflect(row);col=reflect(col)}return row>=0&&row<n&&col>=0&&col<n?channel*n*n+row*n+col:-1},patch=Array.from({length:k*k},(_,j)=>resolve(row0+Math.floor(j/k),col0+j%k)).filter(j=>j>=0),source=resolve(row0+Math.floor(local/k),col0+local%k),outIndex=spec.kind==='depthwise'?channel*outN*outN+idx:idx
   set(stage===last?values:current,positions,{focus:cycle<.65?source:-1,active:patch});outSet(output,outPositions,cycle>=.65?outIndex:-1)
   group.userData.readout=`y_{${spec.kind==='depthwise'?channel:0},${Math.floor(idx/outN)},${idx%outN}}=${output[outIndex].toFixed(3)}`;kernelSet(weights.slice(channel*k*k,(channel+1)*k*k),kernelPos,local);cursor.visible=false
   const sourcePos:V=source>=0?positions[source]:[(col0+local%k)/(n-1)*4.8-1.9,channel*.78+.6,(row0+Math.floor(local/k))/(n-1)*4.8-2.4]
   flow([kernelPos[local],sourcePos],cycle,0);flow([sourcePos,outPositions[outIndex]],cycle,1)
  })
 }else if(spec.kind==='linear'||spec.kind==='attention'||spec.kind==='composition'){
  const n=6,m=4,x=demoValues.slice(0,n),rawW=Array.from({length:n*m},(_,i)=>Math.sin(i*1.4)*.55),W=spec.weightNorm?rawW.map((w,i)=>w/Math.hypot(...rawW.slice(Math.floor(i/n)*n,(Math.floor(i/n)+1)*n))):rawW,positions:Array<V>=Array.from({length:n*m},(_,i)=>[(i%n-2.5)*.8,0,(Math.floor(i/n)-1.5)*1.05]),set=tiles(n*m),input=tiles(n),output=tiles(m)
  const xp:V[]=x.map((_,i)=>[(i-2.5)*.8,1.5,-3]),yp:V[]=Array.from({length:m},(_,i)=>[4.1,0,(i-1.5)*1.05])
  const y=Array.from({length:m},(_,i)=>x.reduce((s,a,j)=>s+a*W[i*n+j],.1))
  updates.push(f=>{const row=Math.min(m-1,Math.floor(f.probe*m)),col=Math.floor(f.phase*n)%n,cycle=f.phase*n%1;group.userData.readout=`y_${row}=${y[row].toFixed(3)}`;set(stage===1?W.map((a,i)=>a*x[i%n]):W,positions,{focus:row*n+col,active:Array.from({length:n},(_,j)=>row*n+j)});input(x,xp,col);output(y,yp,cycle>.65?row:-1);cursor.position.set(0,.3,(row-1.5)*1.05);cursor.scale.set(8,1,1.5);flow([xp[col],positions[row*n+col],yp[row]],cycle)})
 }else if(spec.kind==='activation'){
  const op=spec.operation,range=op.includes('clamp')?20:4,normalizer=op==='clamp-exp'?Math.exp(10*Math.tanh(2)):op==='clamp'?10:op==='sigmoid'||op==='tanh'?1:4
  for(let c=0;c<3;c++){const z=(c-1)*1.25;line([[-3,-1.7,z],[3,-1.7,z]],'#617f91');line([[0,-1.7,z],[0,2,z]],'#617f91');line(Array.from({length:101},(_,i)=>{const x=(i/100*2-1)*range;return [x/range*3,activateValue(x,op)/normalizer*2,z] as V}),'#9de4dc',.95)}
  const a=marker('#e8af77'),b=marker('#92d9ec');cursor.visible=false
  const set=tiles(12,.55),pos=gridPositions(2,3,1.2).map(p=>[p[0]+4,p[1]-.5,p[2]] as V)
  updates.push(f=>{const x=(f.probe*2-1)*range,y=activateValue(x,op),z=0;group.userData.readout=`f(${x.toFixed(2)})=${y.toPrecision(3)}`;a.position.set(x/range*3,-1.7,z);b.position.set(x/range*3,y/normalizer*2,z);flow([a.position.toArray() as V,b.position.toArray() as V],f.phase);set(demoValues.slice(0,12).map(a=>stage===0?a:activateValue(a*range/2,op)/Math.max(1,normalizer)),pos,Math.floor(f.phase*12)%12)})
 }else if(spec.kind==='fusion'){
  cursor.visible=false
  const positions:V[]=[[-2.2,1,-.8],[0,1,-.8],[2.2,1,-.8]],inputs=tiles(3,.6)
  const output=tiles(spec.operation==='normalize'?2:1,.6),outPos:V[]=spec.operation==='normalize'?[[-.7,-1.4,1],[.7,-1.4,1]]:[[0,-1.4,1]]
  updates.push(f=>{const data=fusionValues(.45+f.probe*1.35,f.temperature),w=data.weights.reduce((a,b)=>a+b),m=data.contributions.reduce((a,b)=>a+b),values=spec.operation==='weighted-mean'?stage===0?data.mus:data.contributions:spec.operation==='normalize'?[w,m,0]:data.weights
   const term=Math.floor(f.phase*3)%3,cycle=f.phase*3%1
   group.userData.readout=spec.operation==='weight'?`w=${w.toFixed(3)}`:spec.operation==='weighted-mean'?`m=${m.toFixed(3)}`:`μ=${data.mu.toFixed(3)},s=${data.baseScale.toFixed(3)}`;inputs(values,positions,{focus:term,active:[0,1,2],hidden:spec.operation==='normalize'?[2]:[]})
   output(spec.operation==='normalize'?[data.mu,data.baseScale]:spec.operation==='weighted-mean'?[m]:[w],outPos,cycle>.65?0:-1)
   if(spec.operation==='normalize'){flow([positions[0],outPos[0]],f.phase,0);flow([positions[1],outPos[0]],f.phase,1);flow([positions[0],outPos[1]],f.phase,2)}else positions.forEach((p,j)=>flow([p,outPos[0]],cycle,j,j===term?1:.3))
  })
 }else if(spec.kind==='distribution'){
  cursor.visible=false
  const count=32,geo=new T.PlaneGeometry(5.8,5.8,count,count);geo.rotateX(-Math.PI/2);const mesh=new T.Mesh(geo,new T.MeshStandardMaterial({color:'#8dd2e5',side:T.DoubleSide,transparent:true,opacity:.62,roughness:.4,metalness:.1}));group.add(mesh)
  const base=geo.attributes.position.array.slice(),pos=geo.attributes.position as T.BufferAttribute
  const contours=[1,1.6,2.1].map(r=>{const ring=new T.Mesh(new T.RingGeometry(r-.015,r+.015,56),new T.MeshBasicMaterial({color:'#a6daea',side:T.DoubleSide}));ring.rotation.x=-Math.PI/2;group.add(ring);return ring})
  line([[-3,0,0],[3,0,0]]);line([[0,0,-3],[0,0,3]]);const bars=tiles(3,.65),bp:V[]=[[-2,-.8,3.8],[0,-.8,3.8],[2,-.8,3.8]];let previous=-1
  updates.push(f=>{const s=.45+f.probe*1.35;group.userData.readout=`s=${s.toFixed(2)},s²=${(s*s).toFixed(3)}`;if(previous!==f.probe){previous=f.probe;for(let i=0;i<pos.count;i++){const x=base[i*3],z=base[i*3+2];pos.setY(i,Math.exp(-.5*(x*x/s**2+z*z/.65**2))/(2*Math.PI*s*.65)*8)}pos.needsUpdate=true;geo.computeVertexNormals();contours.forEach((ring,i)=>{ring.scale.set(s,.65,1);ring.position.y=Math.exp(-.5*[1,1.6,2.1][i]**2)/(2*Math.PI*s*.65)*8})}bars([0,s,s*s],bp,{focus:1,active:[0,2]});flow([bp[1],[0,.1,2.7],[0,8/(2*Math.PI*s*.65),0]],f.phase)})
 }else if(spec.kind==='sampling'){
  const geometry=new T.BufferGeometry().setFromPoints(normalSamples.map(p=>v(p as V))),points=new T.Points(geometry,new T.PointsMaterial({size:.095,color:'#a4ddeb',sizeAttenuation:true}));group.add(points);line([[-3,0,0],[3,0,0]]);line([[0,-2,0],[0,2,0]]);line([[0,0,-3],[0,0,3]]);cursor.visible=false
  const from=marker('#e9b58a'),to=marker('#e3fbff'),pos=geometry.attributes.position as T.BufferAttribute
  updates.push(f=>{const mix=stage===0?0:stage===1?f.phase:1;for(let i=0;i<normalSamples.length;i++){const source=uniformSamples[i].slice(0,3).map(x=>(x-.5)*4),target=normalSamples[i];pos.setXYZ(i,...source.map((x,j)=>x+(target[j]-x)*mix) as V)}pos.needsUpdate=true;geometry.computeBoundingSphere();const index=Math.min(normalSamples.length-1,Math.floor(f.probe*normalSamples.length));from.position.set(...uniformSamples[index].slice(0,3).map(x=>(x-.5)*4) as V);to.position.set(...normalSamples[index] as V);group.userData.readout=`ε_1=${normalSamples[index][0].toFixed(3)},ε_2=${normalSamples[index][1].toFixed(3)}`;flow([from.position.toArray() as V,to.position.toArray() as V],f.phase)})
 }else if(spec.kind==='interpolation'){
  const values=demoValues.slice(0,16),inputPos=gridPositions(4).map(p=>[p[0],1.4,p[2]] as V),outPos=gridPositions(8).map(p=>[p[0],-1.6,p[2]] as V),output=Array.from({length:64},(_,i)=>bilinear(values,4,(i%8+.5)/2-.5,(Math.floor(i/8)+.5)/2-.5).value),set=tiles(16),outSet=tiles(64)
  cursor.visible=false
  updates.push(f=>{const i=Math.min(63,Math.floor(f.probe*64)),x=(i%8+.5)/2-.5,y=(Math.floor(i/8)+.5)/2-.5,result=bilinear(values,4,x,y),active=result.ids.filter((_,j)=>result.weights[j]>0);set(values,inputPos,{focus:active[Math.floor(f.phase*active.length)%active.length],active});outSet(output,outPos,f.phase>.65?i:-1);result.ids.forEach((id,j)=>flow([inputPos[id],outPos[i]],f.phase,j,result.weights[j]));group.userData.readout=`y_{${Math.floor(i/8)},${i%8}}=${result.value.toFixed(3)}`})
 }else if(spec.kind==='normalization'||spec.kind==='pooling'){
  const positions=gridPositions(4,3),data=demoValues.slice(0,48),sets=Array.from({length:3},(_,c)=>data.slice(c*16,(c+1)*16)),centered=sets.flatMap(a=>a.map(x=>x-mean(a))),norm=sets.flatMap(normalizeValues),means=sets.map(mean),set=tiles(stage===2&&spec.kind==='pooling'?3:48,.46)
  const centers:V[]=[[-4,0,0],[-4,.9,0],[-4,1.8,0]],statSet=stage===2&&spec.kind==='pooling'?()=>{}:tiles(3,.46),inputSet=stage===2&&spec.kind==='pooling'?tiles(48,.46):null
  updates.push(f=>{const c=Math.min(2,Math.floor(f.probe*3)),index=c*16+Math.floor(f.phase*16)%16,active=Array.from({length:16},(_,j)=>c*16+j);group.userData.readout=`μ_${c}=${means[c].toFixed(3)},v_${c}=${mean(sets[c].map(x=>(x-means[c])**2)).toFixed(3)}`;set(spec.kind==='pooling'?stage===2?means:data:stage===0?data:stage===1?centered.map(x=>x*x):norm,stage===2&&spec.kind==='pooling'?centers:positions,stage===2&&spec.kind==='pooling'?c:{focus:index,active});inputSet?.(data,positions,{focus:index,active});statSet(spec.kind==='normalization'&&stage===1?sets.map(a=>mean(a.map(x=>(x-mean(a))**2))):means,centers,c);cursor.scale.set(9,1,9);cursor.position.set(0,c*.78+.25,0);flow(stage===2&&spec.kind==='normalization'?[centers[c],positions[index]]:[positions[index],centers[c]],f.phase*16%1)})
 }else if(spec.kind==='concatenation'||spec.kind==='split'){
  const positions=gridPositions(3,4),values=Array.from({length:36},(_,i)=>i<18?.25+(i%9)*.06:-.25-(i%9)*.06),set=tiles(36,.68)
  updates.push(f=>{const shift=spec.kind==='concatenation'?stage===0?2.6:stage===1?(1-f.phase)*2.6:0:stage===0?0:stage===1?f.phase*2.6:2.6,pos=positions.map((p,i)=>[p[0]+(i<18?-shift:shift),p[1],p[2]] as V),idx=Math.min(35,Math.floor(f.probe*36)),source:V=[positions[idx][0]+(idx<18?-2.6:2.6),positions[idx][1],positions[idx][2]];set(values,pos,{focus:idx,active:Array.from({length:9},(_,j)=>Math.floor(idx/9)*9+j)});cursor.position.set(...pos[idx]);flow(spec.kind==='concatenation'?[source,positions[idx]]:[positions[idx],source],f.phase)})
 }else if(spec.kind==='addition'||spec.kind==='multiplication'||spec.kind==='mask'){
  const a=demoValues.slice(0,16),b=Array.from({length:16},(_,i)=>spec.kind==='mask'?i%4===0?0:1:spec.kind==='multiplication'?.35+(i%4)*.15:demoValues[i+16]),out=a.map((v,i)=>spec.kind==='addition'?v+b[i]:spec.kind==='mask'?2*((v+1)/2*b[i])-1:v*b[i]),grid=gridPositions(4,1,2.9),left=grid.map(p=>[p[0]-2.1,1,p[2]] as V),right=grid.map(p=>[p[0]+2.1,1,p[2]] as V),result=grid.map(p=>[p[0],-1,p[2]] as V),inputA=tiles(16,.65),inputB=tiles(16,.65),output=tiles(16,.65)
  updates.push(f=>{const i=Math.min(15,Math.floor(f.probe*16));group.userData.readout=`${a[i].toFixed(2)}${spec.kind==='addition'?'+':spec.kind==='mask'?',m=':'×'}${b[i].toFixed(2)} → ${out[i].toFixed(3)}`;inputA(a,left,{focus:f.phase<.65?i:-1,active:[i]});inputB(b,right,{focus:f.phase<.65?i:-1,active:[i]});output(out,result,f.phase>=.65?i:-1);cursor.position.set(...result[i]);flow([left[i],result[i]],f.phase,0);flow([right[i],result[i]],f.phase,1)})
 }else if(spec.kind==='padding'){
  const input=demoValues.slice(0,16),grid=gridPositions(6,1,5.5),ref=(x:number)=>x<0?-x:x>3?6-x:x,output=grid.map((_,i)=>input[ref(Math.floor(i/6)-1)*4+ref(i%6-1)]),set=tiles(36,.72),mapping=line([[0,0,0],[0,0,0]],'#ebc48c')
  updates.push(f=>{const i=Math.min(35,Math.floor(f.probe*36)),y=Math.floor(i/6)-1,x=i%6-1,source=(ref(y)+1)*6+ref(x)+1;set(output,grid,{focus:f.phase<.65?source:i,active:[source,i]});cursor.position.set(...grid[i]);mapping.geometry.setFromPoints([v(grid[i]),v(grid[source])]);flow(source===i?[grid[i],grid[i]]:[grid[source],[grid[source][0],1.4,grid[source][2]],[grid[i][0],1.4,grid[i][2]],grid[i]],f.phase)})
 }else{
  const n=spec.kind==='broadcast'?27:48,positions=spec.kind==='broadcast'?gridPositions(3,3):gridPositions(4,3),values=Array.from({length:n},(_,i)=>spec.operation==='zeros'?0:spec.operation==='ones'?1:spec.kind==='broadcast'?.25+Math.floor(i/9)*.3:demoValues[i]),vectorPos:V[]=values.map((_,i)=>[((i%12)-5.5)*.46,Math.floor(i/12)*.65,0]),set=tiles(n,.42)
  if(spec.kind==='broadcast'){const sourceSet=tiles(3,.42);updates.push(f=>sourceSet([.25,.55,.85],[[-4,0,0],[-4,.78,0],[-4,1.56,0]],Math.min(2,Math.floor(f.probe*3))))}
  updates.push(f=>{let pos=positions;if(spec.kind==='reshape'){const t=stage===1?f.phase:stage===2?1:0;pos=positions.map((p,i)=>p.map((a,k)=>spec.operation==='flatten'?a+(vectorPos[i][k]-a)*t:vectorPos[i][k]+(a-vectorPos[i][k])*t) as V)}const index=Math.min(n-1,Math.floor(f.probe*n));if(spec.kind==='broadcast'){const c=Math.floor(index/9),source:V=[-4,c*.78,0];set(values,pos,{focus:index,active:Array.from({length:9},(_,j)=>c*9+j)});cursor.position.set(...pos[index]);flow([source,pos[index]],f.phase)}else{set(values,pos,index);cursor.position.set(...pos[index]);if(spec.kind==='reshape')flow(spec.operation==='flatten'?[positions[index],vectorPos[index]]:[vectorPos[index],positions[index]],f.phase)}})
 }
 const annotation=(text:string,position:V)=>({text,position})
 group.userData.annotations=spec.kind==='convolution'||spec.kind==='depthwise'?[annotation('X \\;[C,H,W]',[.5,3,0]),annotation('W \\;[C,k,k]',[-4,2.8,0]),annotation('Y \\;[C_{out},H_{out},W_{out}]',[.5,-3.2,0])]:spec.kind==='interpolation'?[annotation('X \\;[4,4]',[0,2.4,0]),annotation('Y \\;[8,8]',[0,-2.7,0])]:['addition','multiplication','mask'].includes(spec.kind)?[annotation('A',[-2.1,2,0]),annotation(spec.kind==='mask'?'m':'B',[2.1,2,0]),annotation('Y',[0,-2,0])]:spec.kind==='fusion'?[annotation(spec.operation==='normalize'?'w,\\;m':'w_p,\\;w_1,\\;w_2',[0,2,0]),annotation(spec.operation==='normalize'?'\\mu,\\;s':spec.operation==='weighted-mean'?'m':'w',[0,-2.3,1])]:spec.kind==='broadcast'?[annotation('g \\;[C,1,1]',[-4,2.5,0]),annotation('g_{c,h,w}=g_c',[0,2.8,0])]:[]
 return {group,update:frame=>{updates.forEach(fn=>fn(frame));group.userData.phase=frame.phase}}
}
