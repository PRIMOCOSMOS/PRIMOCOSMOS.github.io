/** Source-coordinate calculation graphs for the shared workstation renderer.
 * These are symbolic tensors, NOT checkpoint activations. Windowing only limits
 * the addressed coordinates on screen; source channel counts are never resized.
 */
import type {Run,Step,Tensor,Term} from '../workstation/engine'
import {offset,coords} from '../workstation/engine'
import {sourceWindow} from '../mmhvae/sourceTensor'
import {paperModule,type SourceModule} from './sourceArchitecture'
import {MODALITIES} from '../mmhvae/model'
import {mathSpec,type MathSpec} from '../mmhvae/mathematics'
import type {PaperModel} from './catalog'
import {concreteMmhvaePlan} from './mmhvaeExecution'

type Leaf={module:SourceModule;path:string[];group:string;lane:string};
type Link={from:string;to:string;label?:string};
export interface SourcePlan {leaves:Leaf[];links:Link[];paths:Record<string,string[]>;titles:Record<string,string>}
export function flattenSource(root:SourceModule):SourcePlan{
 const leaves:Leaf[]=[],links:Link[]=[],paths:Record<string,string[]>={},titles:Record<string,string>={},ends=new Map<string,{input:string[];output:string[]}>();
 const visit=(m:SourceModule,path:string[],lane:string):{input:string[];output:string[]}=>{
  if(m.expand)m=m.expand();
  titles[m.id]=m.title;const current=[...path,m.id];
  if(!m.children.length){leaves.push({module:m,path:current,group:path.at(-1)??m.id,lane});paths[m.id]=[m.id];const end={input:[m.id],output:[m.id]};ends.set(m.id,end);return end}
  const children=m.children.map(c=>visit(c,current,path.length===0?c.id:lane));
  const ids=new Set(m.children.map(c=>c.id)),incoming=new Set(m.edges.filter(e=>!(/下一轮/.test(e.label??''))&&ids.has(e.from)&&ids.has(e.to)).map(e=>e.to)),outgoing=new Set(m.edges.filter(e=>!(/下一轮/.test(e.label??''))&&ids.has(e.from)&&ids.has(e.to)).map(e=>e.from));
  // An explicit edge owns the connection. Do not invent a chain from UI order.
  for(const e of m.edges){if(/下一轮/.test(e.label??''))continue;const a=ends.get(e.from),b=ends.get(e.to);if(a&&b)for(const from of a.output)for(const to of b.input)if(from!==to)links.push({from,to,label:e.label})}
  const input=children.flatMap((c,i)=>incoming.has(m.children[i].id)?[]:c.input),output=children.flatMap((c,i)=>outgoing.has(m.children[i].id)?[]:c.output);
  paths[m.id]=m.children.flatMap(c=>paths[c.id]??[]);const end={input,output};ends.set(m.id,end);return end;
 };
 visit(root,[],'');return {leaves,links:[...new Map(links.map(e=>[`${e.from}>${e.to}`,e])).values()],paths,titles};
}
export const paperSourcePlan=(model:PaperModel)=>flattenSource(paperModule(model,'root',20));

const product=(a:number[])=>a.reduce((n,v)=>n*v,1);
const key=(c:number[])=>c.join(',');
const R=String.raw;
/** Exact source addresses. Unknown dimensions remain named symbols; we display
 * only coordinate zero on an unknown axis, never assert its extent is one. */
function tensor(id:string,name:string,dims:(number|null)[],label:string,page:number,parameter=false):Tensor{
 const axes=dims.length?dims:[null],shape=axes.map(d=>d??1),str=axes.map((d,i)=>d??`d${i}`).join(' × '),w=sourceWindow(str,'tensor',page);
 const positions=w.positions.length?w.positions:[[0,0,0] as [number,number,number]],coordinates=w.coordinates.length?w.coordinates:[axes.map(()=>0)];
 const y=(Math.max(...positions.map(p=>p[1]))+Math.min(...positions.map(p=>p[1])))/2;
 return {id,name,shape,values:positions.map(()=>0),parameter,window:{shapeLabel:label||str,dimensions:axes,coordinates,positions:positions.map(p=>[p[0],p[1]-y,p[2]]),label:parameter?w.label.replace('C ','O ').replace('D ','I ').replace('行 ','O ').replace('列 ','I '):w.label,symbolic:true}};
}
function dimsFor(m:SourceModule,inputs:Tensor[]):(number|null)[]{
 if(m.dimensions)return [...m.dimensions];
 let dims=sourceWindow(m.shape,m.glyph).dimensions;
 const op=m.operator?.kind,x=inputs[0]?.window?.dimensions;
 if(x&&/gelu|leaky|softplus|layernorm|dropout|softmax|l1norm|threshold/.test(m.operator?.operation??''))return [...x];
 if(x&&(['activation','normalization','addition','multiplication','mask'].includes(op??'')||/形状不变|尺寸不变/.test(m.shape)))return /ch/.test(m.shape)&&dims.length===1?[dims[0],...x.slice(1)]:[...x];
 if(op==='linear'&&x){const out=Number(m.shape.split('→').at(-1)?.match(/\d+/)?.[0]);if(out)return [...x.slice(0,-1),out]}
 if(!dims.length&&x)return [...x];
 if(dims.length===1&&/ch/.test(m.shape)&&x&&x.length>=3)return [dims[0],...x.slice(-2)];
 return dims.length?dims:x??[null];
}
export function compileSource(plan:SourcePlan,page=0):Run{
 const tensors:Tensor[]=[],steps:Step[]=[],outputs=new Map<string,Tensor>(),warnings:string[]=[],all=new Map(plan.leaves.map(l=>[l.module.id,l]));
 const pending=[...plan.leaves],done=new Set<string>(),ordered:Leaf[]=[];
 while(pending.length){const ready=pending.filter(l=>plan.links.filter(e=>e.to===l.module.id).every(e=>done.has(e.from)||!all.has(e.from)));if(!ready.length){warnings.push('源码索引存在循环依赖：'+pending.map(l=>l.module.id).join(', '));break}for(const l of ready){pending.splice(pending.indexOf(l),1);done.add(l.module.id);ordered.push(l)}}
 const lookup=new WeakMap<Tensor,Map<string,number>>();
 const at=(t:Tensor,c:number[])=>{let map=lookup.get(t);if(!map){map=new Map(t.window!.coordinates.map((c,i)=>[key(c),i]));lookup.set(t,map)}return map.get(key(c))??-1};
 const term=(t:Tensor,c:number[],label?:string,factor?:number):Term=>({tensor:t.id,index:at(t,c),value:NaN,label:label??`${t.name}[${c.join(',')}]`,factor});
 const aligned=(t:Tensor,c:number[])=>{const delta=c.length-t.shape.length;return t.shape.map((d,i)=>d===1?0:c[i+delta]??0)};
 const make=(id:string,name:string,dims:(number|null)[],label='',param=false)=>{const t=tensor(id,name,dims,label,page,param);tensors.push(t);return t};
 const add=(leaf:Leaf,suffix:string,title:string,inputs:Tensor[],output:Tensor,formula:string,trace:Step['trace'],kind='Source',extra:Record<string,unknown>={})=>{
  const m=leaf.module;steps.push({id:`${m.id}::${suffix}`,group:leaf.group,title,kind,inputs,output,formula:formula||m.formula||m.operator?.formula||'',detail:m.source,source:m.source,trace,settings:{symbolic:true,groupTitle:plan.titles[leaf.group]??leaf.group,sourceKind:leaf.module.operator?.kind,sourceId:m.id,ancestors:leaf.path,lane:leaf.lane,operation:m.operator?.operation,...m.settings,...extra}});return output;
 };
 const unary=(leaf:Leaf,suffix:string,title:string,x:Tensor,formula:string,kind='Source')=>{const out=make(`${leaf.module.id}::${suffix}:data`,title,x.window!.dimensions);return add(leaf,suffix,title,[x],out,formula,i=>[term(x,out.window!.coordinates[i])],kind)};
 const normalizedWeight=(leaf:Leaf,dims:(number|null)[],spec:MathSpec)=>{
  const id=leaf.module.id;if(!spec.weightNorm)return make(id+':W','W',dims,'',true);
  const v=make(id+':V','权重方向 V',dims,'',true),g=make(id+':g','权重长度 g',[dims[0]],'',true),norm=make(id+':weight-norm','每输出通道的 L2 范数',[dims[0]]);
  add(leaf,'weight-norm','WeightNorm · 方向范数',[v],norm,R`n_o=\sqrt{\sum_jV_{o,j}^2}`,i=>v.window!.coordinates.filter(c=>c[0]===norm.window!.coordinates[i][0]).map(c=>term(v,c)));
  const w=make(id+':W','归一化卷积 / 线性权重 W',dims);
  add(leaf,'weight-scale','WeightNorm · 方向与长度合成',[v,norm,g],w,R`W_{o,j}=g_oV_{o,j}/n_o`,i=>{const c=w.window!.coordinates[i];return [term(v,c),term(norm,[c[0]]),term(g,[c[0]])]});return w;
 };
 for(const leaf of ordered){const m=leaf.module,spec=m.operator??mathSpec({id:m.id,title:m.title,glyph:m.glyph,shape:m.shape,detail:'',sourceName:m.source},m.formula),kind=spec.kind,op=spec.operation;
  let ins=[...new Map(plan.links.filter(e=>e.to===m.id).map(e=>outputs.get(e.from)).filter((t):t is Tensor=>!!t).map(t=>[t.id,t])).values()];
  const dims=dimsFor({...m,operator:spec},ins),out=make(m.id+':data',m.title,dims,m.shape);
  if(!ins.length&&kind!=='tensor'){const inputDims=kind==='linear'?[Number(m.shape.split('→')[0].match(/\d+/)?.[0])||null]:dims;const input=make(m.id+':input',`输入 · ${m.title}`,inputDims);ins=[input]}
  const x=ins[0],co=(i:number)=>out.window!.coordinates[i];
  if(kind==='linear'){
   const cin=x.window!.dimensions.at(-1),cout=dims.at(-1),w=normalizedWeight(leaf,[cout??null,cin??null],spec),bias=spec.bias?make(m.id+':bias','b',[cout??null],'',true):undefined;
   add(leaf,'linear',m.title,[x,w,...bias?[bias]:[]],out,R`y_{r,o}=b_o+\sum_i x_{r,i}W_{o,i}`,i=>{const c=co(i),o=c.at(-1)!;return [...x.window!.coordinates.filter(v=>v.slice(0,-1).every((a,j)=>a===c[j])).map(v=>({...term(x,v),factorTensor:w.id,factorIndex:at(w,[o,v.at(-1)!]),label:`x[${v}] · W[${o},${v.at(-1)}]`})),...bias?[term(bias,[o])]:[]]},'Linear');
  }else if(kind==='convolution'||kind==='depthwise'){
   const inputDims=x.window!.dimensions,k=spec.kernel,s=spec.stride,p=spec.padding,cin=inputDims.at(-3)??null,cout=dims.at(-3)??null,dw=kind==='depthwise',w=normalizedWeight(leaf,[cout,dw?1:cin,k,k],spec);
   const products=make(m.id+':products','感受野逐项乘积',[dw?1:cin,k,k]);
   // Scratch plane represents the stencil of the active output; all channels
   // remain in its declared shape. Spatial indexing is exact, not a shrunken CNN.
   let targetIndex=0;const selectOutput=(i:number)=>{targetIndex=i%out.values.length};
   const stencil=(c:number[])=>products.window!.coordinates.map(v=>{const channel=dw?c.at(-3)!:v[0],h=c.at(-2)!*s-p+v[1],j=c.at(-1)!*s-p+v[2],source=[...c.slice(0,-3),channel,h,j],wi=[c.at(-3)!,dw?0:channel,v[1],v[2]];
    for(let axis=source.length-2;axis<source.length;axis++){const d=x.shape[axis];if(spec.reflect&&d>1){const period=2*(d-1);let n=((source[axis]%period)+period)%period;source[axis]=n>=d?period-n:n}}
    const valid=source.every((v,i)=>v>=0&&v<x.shape[i]);return {source,wi,valid};});
   add(leaf,'products','卷积 · 输入 × 核权重',[x,w],products,R`P_{c,u,v}=X_{c,sh-p+u,sw-p+v}W_{o,c,u,v}`,i=>{const t=stencil(out.window!.coordinates[targetIndex])[i];return t?.valid?[{...term(x,t.source),factorTensor:w.id,factorIndex:at(w,t.wi),label:`X[${t.source}] × W[${t.wi}]`}]:[]},'Source',{kernel:k,stride:s,padding:p,sourceConvolution:true,receptiveInput:x.id,receptiveOutput:out.id,selectOutput,receptiveTarget:()=>targetIndex});
   const sum=make(m.id+':sum','跨核位置与输入通道归约',dims);
   add(leaf,'sum','卷积 · 归约',[products],sum,dw?R`a_{c,h,w}=\sum_{u,v}P_{c,u,v}`:R`a_{o,h,w}=\sum_{c,u,v}P_{c,u,v}`,()=>products.window!.coordinates.map(c=>term(products,c)));
   const bias=spec.bias?make(m.id+':bias','b',[cout],'',true):undefined;
   add(leaf,'write',m.title,[sum,...bias?[bias]:[]],out,spec.formula||R`Y_{o,h,w}=a_{o,h,w}+b_o`,i=>[term(sum,co(i)),...bias?[term(bias,[co(i).at(-3)!])]:[]]);
  }else if(kind==='normalization'||/layernorm|norm-affine|adain/.test(op)){
   const layer=/layernorm/.test(op),axes=x.window!.dimensions,statsDims=layer?[...axes.slice(0,-1),1]:[...axes.slice(0,-2),1,1],mu=make(m.id+':mean','均值 μ',statsDims);
   const same=(c:number[],v:number[])=>layer?v.slice(0,-1).every((a,j)=>a===c[j]):v.slice(0,-2).every((a,j)=>a===c[j]);
   add(leaf,'mean',layer?'沿特征轴求均值':'每通道空间均值',[x],mu,R`\mu_g=\frac1{|g|}\sum_{i\in g}X_i`,i=>x.window!.coordinates.filter(v=>same(mu.window!.coordinates[i],v)).map(v=>term(x,v)));
   const centered=make(m.id+':center','中心化 X−μ',axes);add(leaf,'center','逐坐标中心化',[x,mu],centered,R`C_i=X_i-\mu_{g(i)}`,i=>{const c=centered.window!.coordinates[i];return [term(x,c),term(mu,aligned(mu,c))]});
   const square=unary(leaf,'square','偏差平方',centered,R`Q_i=C_i^2`),variance=make(m.id+':variance','总体方差',statsDims);
   add(leaf,'variance','归约偏差平方',[square],variance,R`v_g=\frac1{|g|}\sum_{i\in g}Q_i`,i=>square.window!.coordinates.filter(v=>same(variance.window!.coordinates[i],v)).map(v=>term(square,v)));
   const std=unary(leaf,'std','稳定项与开平方',variance,R`d_g=\sqrt{v_g+\varepsilon}`);
   const affine=layer||/norm-affine/.test(op),normalized=affine?make(m.id+':normalized','标准化响应',dims):out;
   add(leaf,'normalize','除以所属组标准差',[centered,std],normalized,R`Z_i=C_i/d_{g(i)}`,i=>{const c=normalized.window!.coordinates[i];return [term(centered,c),term(std,aligned(std,c))]});
   if(affine){const axis=layer?dims.length-1:0,gamma=make(m.id+':gamma','γ',[dims[axis]],'',true),beta=make(m.id+':beta','β',[dims[axis]],'',true),scaled=make(m.id+':affine-scale','逐特征缩放',dims);add(leaf,'affine-scale','归一化 · 乘 γ',[normalized,gamma],scaled,R`S_i=\gamma_i Z_i`,i=>{const c=scaled.window!.coordinates[i];return [term(normalized,c),term(gamma,[c[axis]])]});add(leaf,'affine-shift',m.title,[scaled,beta],out,R`Y_i=S_i+\beta_i`,i=>[term(scaled,co(i)),term(beta,[co(i)[axis]])]);}

  }else if(op==='core:softmax'){
   const axes=x.window!.dimensions,rowDims=[...axes.slice(0,-1),1],sameRow=(a:number[],b:number[])=>a.slice(0,-1).every((v,j)=>v===b[j]),maximum=make(m.id+':maximum','每 query 行最大值',rowDims);
   add(leaf,'row-max','Softmax · 行最大值',[x],maximum,R`m_q=max_k S_{qk}`,i=>x.window!.coordinates.filter(c=>sameRow(c,maximum.window!.coordinates[i])).map(c=>term(x,c)));
   const centered=make(m.id+':centered','稳定平移',axes);add(leaf,'row-center','Softmax · 减去行最大值',[x,maximum],centered,R`C_{qk}=S_{qk}-m_q`,i=>{const c=centered.window!.coordinates[i];return [term(x,c),term(maximum,[...c.slice(0,-1),0])]});
   const exp=unary(leaf,'exp','Softmax · 指数响应',centered,R`E_{qk}=exp(C_{qk})`),sum=make(m.id+':denominator','每行指数和',rowDims);
   add(leaf,'row-sum','Softmax · 沿 key 求和',[exp],sum,R`Z_q=sum_kE_{qk}`,i=>exp.window!.coordinates.filter(c=>sameRow(c,sum.window!.coordinates[i])).map(c=>term(exp,c)));
   add(leaf,'row-normalize',m.title,[exp,sum],out,R`A_{qk}=E_{qk}/Z_q`,i=>[term(exp,co(i)),term(sum,[...co(i).slice(0,-1),0])]);
  }else if(kind==='activation'||/gelu|leaky|softplus/.test(op)){
   const type=/^(exp|exp-negative|reciprocal|clamp|temperature)$/.test(op)?'SourceFunction':/gelu/.test(op)?'GELU':/leaky/.test(op)?'LeakyReLU':/sigmoid/.test(op)?'Sigmoid':/tanh/.test(op)?'Tanh':op==='relu'?'ReLU':/softplus/.test(op)?'Softplus':op==='silu'?'SiLU':'Source';
   add(leaf,'activation',m.title,[x],out,spec.formula||spec.steps[1]?.formula,i=>[term(x,aligned(x,co(i)))],type,{activation:op});
  }else if(kind==='pooling'||/token-mean/.test(op)){
   add(leaf,'pool',m.title,[x],out,spec.formula||R`y_c=\frac1{HW}\sum_{h,w}x_{c,h,w}`,i=>x.window!.coordinates.filter(v=>/token-mean/.test(op)?v.at(-1)===co(i).at(-1):v[0]===co(i)[0]).map(v=>term(x,v)));
  }else if(kind==='concatenation'){
   add(leaf,'concat',m.title,ins,out,R`Y=\operatorname{cat}(X_1,\ldots,X_m;C)`,i=>{const c=[...co(i)],axis=Math.max(0,c.length-3);let channel=c[axis];for(const t of ins){const n=t.shape[axis];if(channel<n){c[axis]=channel;return [term(t,c)]}channel-=n}return []});
  }else if(kind==='reshape'||kind==='broadcast'||kind==='split'||/reshape|detach|split/.test(op)){
   add(leaf,'index',m.title,ins,out,spec.formula||R`Y_i=X_{\pi(i)}`,i=>{const c=co(i);return [term(x,op==='lab:split:3'&&c.length===4&&x.shape.length===2?[c[2],c[0]*(out.shape[1]*out.shape[3])+c[1]*out.shape[3]+c[3]]:op==='slice'?c.map((v,j)=>v+(j===Number(m.settings?.axis??0)?Number(m.settings?.start??0):0)):kind==='broadcast'?aligned(x,c):coords(offset(c,out.shape),x.shape))]});
  }else if(kind==='interpolation'||/bilinear/.test(op)){
   add(leaf,'interpolate',m.title,[x],out,R`Y=\sum_{a,b\in\{0,1\}}w_{ab}X_{u+a,v+b}`,i=>{const c=co(i),h=(c.at(-2)!+.5)*x.shape.at(-2)!/out.shape.at(-2)!-.5,w=(c.at(-1)!+.5)*x.shape.at(-1)!/out.shape.at(-1)!-.5;return [0,1].flatMap(a=>[0,1].map(b=>term(x,[...c.slice(0,-2),Math.max(0,Math.min(x.shape.at(-2)!-1,Math.floor(h)+a)),Math.max(0,Math.min(x.shape.at(-1)!-1,Math.floor(w)+b))],undefined,(a?h-Math.floor(h):1-h+Math.floor(h))*(b?w-Math.floor(w):1-w+Math.floor(w)))))});
  }else if(kind==='padding'){
   add(leaf,'pad',m.title,[x],out,R`Y_{h,w}=X_{\mathrm{reflect}(h-p),\mathrm{reflect}(w-p)}`,i=>{const c=co(i).map((v,j)=>j<out.shape.length-2?v:v-spec.padding);c.forEach((v,j)=>{if(j<c.length-2)return;const period=2*(x.shape[j]-1);if(period){const n=((v%period)+period)%period;c[j]=n>=x.shape[j]?period-n:n}});return [term(x,c)]});
  }else if(kind==='addition'||kind==='multiplication'||kind==='mask'||/add|multiply|subtract|scale|mask/.test(op)){
   add(leaf,'elementwise',m.title,ins,out,spec.formula||spec.steps[1]?.formula,i=>ins.map(t=>term(t,aligned(t,co(i)))));
  }else if(/scores|weighted|fft|ifft/.test(op)){
   // Contractions preserve the full declared reduction axis; only the displayed
   // coordinate window is connected. No fictitious sparse attention is claimed.
   add(leaf,'contract',m.title,ins,out,spec.formula||spec.steps[1]?.formula,i=>{
    const c=co(i);
    if(op==='core:scores'&&x.window!.dimensions.length===4){const [h,q,k]=c;return x.window!.coordinates.filter(v=>v[0]===0&&v[1]===h&&v[2]===q).map(v=>({...term(x,v),factorTensor:x.id,factorIndex:at(x,[1,h,k,v[3]]),label:`Q[${h},${q},${v[3]}] K[${h},${k},${v[3]}] / √d`}))}
    if(op==='core:weighted'){const v=ins.find(t=>t.window!.dimensions.length===4),[h,q,d]=c;if(v)return x.window!.coordinates.filter(a=>a[0]===h&&a[1]===q).map(a=>({...term(x,a),factorTensor:v.id,factorIndex:at(v,[2,h,a[2],d]),label:`A[${a}] V[${h},${a[2]},${d}]`}))}
    return ins.flatMap(t=>t.window!.coordinates.filter(c=>/fft|ifft/.test(op)||c.slice(0,-2).every((v,j)=>v===co(i)[j])).map(c=>term(t,c)))
   }, 'Source');
  }else if(kind==='tensor'&& !op.startsWith('lab:')&&!op.startsWith('core:')){
   add(leaf,'data',m.title,ins,out,m.formula,i=>ins.map(t=>term(t,aligned(t,co(i)))));
  }else{
   // Explicit algebra stages are on the root scaffold, never hidden in a popup.
   // Their equations come from the source audit. Symbolic stages do not pretend
   // to have evaluated weights or flatten an unknown tensor to a numeric scalar.
   let previous=ins;
   for(const [j,stage] of spec.steps.entries()){
    const last=j===spec.steps.length-1,result=last?out:make(`${m.id}:algebra:${j}`,stage.title,dims);
    const inputs=previous;add(leaf,`algebra:${j}`,stage.title,inputs,result,stage.formula,i=>inputs.map(t=>term(t,aligned(t,result.window!.coordinates[i]))));previous=[result];
   }
  }
  outputs.set(m.id,out);
 }
 // Use source module titles for functional regions, while maintaining stable
 // unique IDs for repeated blocks and the full source file navigator.
 const names=new Map(plan.leaves.flatMap(l=>[[l.module.id,l.module.title] as const]));
 for(const s of steps)s.group=`${plan.titles[s.group]??names.get(s.group)??s.group} · ${s.group}`;
 const paths=Object.fromEntries(Object.entries(plan.paths).map(([id,leaves])=>[id,steps.filter(s=>leaves.includes(String(s.settings.sourceId))).map(s=>s.id)]));
 const fallback=make('empty','待指定输入',[null]);
 return {tensors,steps,input:steps[0]?.inputs[0]??steps[0]?.output??fallback,output:steps.at(-1)?.output??fallback,parameters:tensors.filter(t=>t.parameter).reduce((n,t)=>n+product(t.shape),0),scalars:tensors.reduce((n,t)=>n+product(t.shape),0),sourceGraph:{paths,warnings}};
}
export function makeMmhvaeRun(observed:string[]=MODALITIES.map(m=>m.id),page=0,temperature=.5){return compileSource(concreteMmhvaePlan(observed,temperature),page)}
export function makePaperRun(model:PaperModel,page=0){return compileSource(paperSourcePlan(model),page)}
