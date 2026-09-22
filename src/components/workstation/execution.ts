import type {Run,Step,Tensor,Term} from './engine'

export interface ExecutionBlock {source:Step;steps:Step[];select:(index:number)=>void;streamed:boolean}
const R=String.raw
const term=(t:Tensor,i:number,factor?:number):Term=>({tensor:t.id,index:i,value:t.values[i],factor})

/** The same arithmetic DAG is used at every zoom level. Streaming registers
 * contain ALL terms of one output, not sampled or representative neurons. */
export function executionBlock(source:Step):ExecutionBlock {
 const steps:Step[]=[],x=source.inputs[0];let serial=0,select=(_index:number)=>{},streamed=false
 const add=(name:string,inputs:Tensor[],values:number[],shape:number[],formula:string,trace:Step['trace'],output?:Tensor)=>{
  const id=`${source.id}:math:${serial++}`,tensor=output??{id,name,values,shape}
  steps.push({...source,id,title:name,kind:'Arithmetic',inputs,output:tensor,formula,trace,settings:{owner:source.id}});return tensor
 }
 const unary=(name:string,a:Tensor,fn:(v:number)=>number,formula:string)=>add(name,[a],a.values.map(fn),a.shape,formula,i=>[term(a,i)])
 const pair=(name:string,a:Tensor,b:Tensor,fn:(a:number,b:number)=>number,formula:string)=>add(name,[a,b],a.values.map((v,i)=>fn(v,b.values[i])),a.shape,formula,i=>[term(a,i),term(b,i)])
 const finish=(inputs:Tensor[],trace:Step['trace'],title=source.title,formula=source.formula)=>add(title,inputs,source.output.values,source.output.shape,formula,trace,source.output)
 const kind=source.kind,op=source.settings.operation
 if(kind==='Linear'){
  const [input,w,b]=source.inputs,n=input.shape.at(-1)!,out=w.shape[0]
  const products=add('逐项乘积 x · W',[input,w],Array.from({length:source.output.values.length*n},(_,i)=>input.values[Math.floor(i/(out*n))*n+i%n]*w.values[i%(out*n)]),[...input.shape.slice(0,-1),out,n],R`p_{r,o,i}=x_{r,i}W_{o,i}`,i=>[term(input,Math.floor(i/(out*n))*n+i%n,w.values[i%(out*n)]),term(w,i%(out*n))])
  const sum=add('沿输入特征轴累加',[products],source.output.values.map((_,o)=>products.values.slice(o*n,(o+1)*n).reduce((a,b)=>a+b,0)),source.output.shape,R`s_{r,o}=\sum_i p_{r,o,i}`,o=>Array.from({length:n},(_,i)=>term(products,o*n+i,1)))
  finish([sum,...b?[b]:[]],i=>[term(sum,i,1),...b?[term(b,i%out,1)]:[]],b?'广播偏置 · 全连接输出':'全连接输出',b?R`y_{r,o}=s_{r,o}+b_o`:R`y=s`)
 }else if(['ReLU','ReLU6','LeakyReLU'].includes(kind)){
  const predicate=unary('逐元素判断 x > 0',x,v=>v>0?1:0,R`m_i=\mathbf1[x_i>0]`)
  const branch=pair(kind==='LeakyReLU'?'按掩码选择 x 或 0.2x':'按掩码选择 x 或 0',x,predicate,(v,m)=>m?v:kind==='LeakyReLU'?.2*v:0,kind==='LeakyReLU'?R`z_i=m_ix_i+(1-m_i)0.2x_i`:R`z_i=m_ix_i`)
  finish([branch],i=>[term(branch,i)],kind==='ReLU6'?'上限裁剪到 6':'激活输出',kind==='ReLU6'?R`y_i=\min(6,z_i)`:R`y_i=z_i`)
 }else if(['Sigmoid','SiLU'].includes(kind)){
  const abs=unary('取负绝对值',x,v=>-Math.abs(v),R`a=-|x|`),exp=unary('逐元素指数',abs,Math.exp,R`e=\exp(a)`),den=unary('分母加一',exp,v=>1+v,R`d=1+e`)
  const gate=add('按符号选取稳定 Sigmoid',[x,exp,den],x.values.map((v,i)=>(v>=0?1:exp.values[i])/den.values[i]),x.shape,R`g=\begin{cases}1/d&x\ge0\\e/d&x<0\end{cases}`,i=>[term(x,i),term(exp,i),term(den,i)])
  finish(kind==='SiLU'?[x,gate]:[gate],i=>kind==='SiLU'?[term(x,i),term(gate,i)]:[term(gate,i)],kind==='SiLU'?'输入乘以 Sigmoid 门控':'Sigmoid 输出',kind==='SiLU'?R`y=xg`:R`y=g`)
 }else if(kind==='Hardswish'||kind==='Hardsigmoid'){
  const shift=unary('平移 +3',x,v=>v+3,R`a=x+3`),lower=unary('下限裁剪到 0',shift,v=>Math.max(0,v),R`b=\max(0,a)`),upper=unary('上限裁剪到 6',lower,v=>Math.min(6,v),R`c=\min(6,b)`),gate=unary('除以 6',upper,v=>v/6,R`g=c/6`)
  finish(kind==='Hardswish'?[x,gate]:[gate],i=>kind==='Hardswish'?[term(x,i),term(gate,i)]:[term(gate,i)],source.title,kind==='Hardswish'?R`y=xg`:R`y=g`)
 }else if(kind==='Tanh'){
  const abs=unary('绝对值缩放',x,v=>-2*Math.abs(v),R`a=-2|x|`),exp=unary('稳定指数',abs,Math.exp,R`e=\exp(a)`),top=unary('分子 1−e',exp,v=>1-v,R`n=1-e`),den=unary('分母 1+e',exp,v=>1+v,R`d=1+e`),ratio=pair('幅度相除',top,den,(a,b)=>a/b,R`r=n/d`)
  finish([x,ratio],i=>[term(x,i),term(ratio,i)],'恢复输入符号',R`y=\operatorname{sign}(x)r`)
 }else if(kind==='Softplus'){
  const abs=unary('取负绝对值',x,v=>-Math.abs(v),R`a=-|x|`),exp=unary('稳定指数',abs,Math.exp,R`e=\exp(a)`),log=unary('log1p',exp,Math.log1p,R`l=\log(1+e)`),positive=unary('输入的正半轴',x,v=>Math.max(0,v),R`p=\max(0,x)`)
  finish([positive,log],i=>[term(positive,i,1),term(log,i,1)],'两部分相加',R`y=p+l`)
 }else if(kind==='GELU'){
  const z=unary('正态坐标 x / √2',x,v=>v/Math.SQRT2,R`z=x/\sqrt2`),abs=unary('绝对值',z,Math.abs,R`a=|z|`),t=unary('erf 逼近变量',abs,v=>1/(1+.3275911*v),R`t=(1+0.3275911a)^{-1}`)
  let h=unary('Horner · 第 1 项',t,v=>1.061405429*v-1.453152027,R`h_1=1.061405429t-1.453152027`)
  for(const [k,c] of [1.421413741,-.284496736,.254829592].entries())h=pair(`Horner · 第 ${k+2} 项`,h,t,(v,u)=>v*u+c,`h_{${k+2}}=h_{${k+1}}t+(${c})`)
  const poly=pair('多项式乘以 t',h,t,(v,u)=>v*u,R`p=h_4t`),exp=unary('高斯核指数',z,v=>Math.exp(-v*v),R`e=\exp(-z^2)`),erf=add('恢复 erf 的符号',[z,poly,exp],z.values.map((v,i)=>(v<0?-1:1)*(1-poly.values[i]*exp.values[i])),z.shape,R`\operatorname{erf}(z)=\operatorname{sign}(z)(1-pe)`,i=>[term(z,i),term(poly,i),term(exp,i)]),cdf=unary('正态累计概率',erf,v=>.5*(1+v),R`\Phi(x)=\tfrac12(1+\operatorname{erf}(z))`)
  finish([x,cdf],i=>[term(x,i),term(cdf,i)],'输入乘以累计概率',R`y=x\Phi(x)`)
 }else if(kind.includes('Norm')&&source.trace(0).length>=4){
  const traces=source.output.values.map((_,i)=>source.trace(i)),find=(k:number)=>source.inputs.find(t=>t.id===traces[0][k].tensor)!,mu=find(1),variance=find(2),gamma=find(3),beta=traces[0][4]?find(4):undefined,eps=Number(source.settings.eps)
  const centered=add('逐坐标减去所属组均值',[x,mu],traces.map(t=>t[0].value-t[1].value),x.shape,R`c_i=x_i-\mu_{g(i)}`,i=>[traces[i][0],traces[i][1]])
  const adjusted=unary('方差加稳定项 ε',variance,v=>v+eps,`a=v+${eps}`),std=unary('开平方',adjusted,Math.sqrt,R`d=\sqrt{a}`)
  const normalized=add('除以所属组标准差',[centered,std],traces.map((t,i)=>centered.values[i]/std.values[t[2].index]),x.shape,R`n_i=c_i/d_{g(i)}`,i=>[term(centered,i),term(std,traces[i][2].index)])
  const scaled=add('逐特征乘以 γ',[normalized,gamma],traces.map((t,i)=>normalized.values[i]*t[3].value),x.shape,R`s_i=\gamma_i n_i`,i=>[term(normalized,i),traces[i][3]])
  finish([scaled,...beta?[beta]:[]],i=>[term(scaled,i,1),...beta?[traces[i][4]]:[]],'归一化输出 · 加 β',R`y_i=s_i+\beta_i`)
 }else if(kind==='Dropout'){
  const mask=source.inputs[1],masked=pair('逐坐标乘以 Bernoulli 掩码',x,mask,(v,m)=>v*m,R`z_i=x_iM_i`),p=Number(source.settings.p)
  finish([masked],i=>[term(masked,i,1/(1-p))],'保留概率的倒数缩放',`y_i=z_i/(1-${p})`)
 }else if(op==='box-muller'){
  const v=source.inputs[1],log=unary('均匀随机数取对数',x,Math.log,R`l=\log u_1`),radius2=unary('缩放 −2',log,v=>-2*v,R`r^2=-2l`),radius=unary('径向开平方',radius2,Math.sqrt,R`r=\sqrt{-2l}`),angle=unary('均匀角度 2πu₂',v,v=>2*Math.PI*v,R`\theta=2\pi u_2`),cos=unary('角度取余弦',angle,Math.cos,R`c=\cos\theta`)
  finish([radius,cos],i=>[term(radius,i),term(cos,i)],'径向与角向相乘',R`\epsilon=rc`)
 }else if(op==='l2-norm'){
  const square=unary('每个特征平方',x,v=>v*v,R`q_{pd}=h_{pd}^2`),width=x.shape.at(-1)!,sum=add('沿特征维求和',[square],source.output.values.map((_,i)=>square.values.slice(i*width,(i+1)*width).reduce((a,b)=>a+b,0)),source.output.shape,R`s_p=\sum_d q_{pd}`,i=>Array.from({length:width},(_,j)=>term(square,i*width+j,1))),root=unary('范数开平方',sum,Math.sqrt,R`r_p=\sqrt{s_p}`)
  finish([root],i=>[term(root,i)],'范数外加 ε',`n_p=r_p+${source.settings.eps}`)
 }else if(op==='cumprod'){
  // The preceding output is an actual dependency in a scan, not an
  // all-to-all fan. The first element has no predecessor.
  finish([x],i=>[term(x,i),...i?[term(source.output,i-1)]:[]],'逐坐标前缀乘积',R`\bar\alpha_t=\bar\alpha_{t-1}\alpha_t,\quad\bar\alpha_{-1}=1`)
 }else if(op==='time-embedding'){
  const dim=Number(source.settings.dim),half=Math.floor(dim/2),freq=add('指数频率表',[],Array.from({length:half},(_,i)=>Math.exp(-Math.log(10000)*i/(half-1))),[half],R`\omega_j=10000^{-j/(h-1)}`,()=>[]),angle=add('时间乘以每个频率',[x,freq],Array.from({length:x.values.length*half},(_,i)=>x.values[Math.floor(i/half)]*freq.values[i%half]),[x.values.length,half],R`a_{n,j}=t_n\omega_j`,i=>[term(x,Math.floor(i/half)),term(freq,i%half)]),sin=unary('逐坐标正弦',angle,Math.sin,R`s=\sin a`),cos=unary('逐坐标余弦',angle,Math.cos,R`c=\cos a`)
  finish([sin,cos],i=>{const j=i%dim;if(j>=half*2)return [];return [term(j<half?sin:cos,Math.floor(i/dim)*half+j%half)]},'拼接 sin / cos · 奇数末维补零',R`e_t=[s,c,0_{d\bmod2}]`)
 }else if(op==='cross-entropy-zero'){
  const width=x.shape.at(-1)!,rowShape=source.output.shape,max=add('逐行最大 logit',[x],source.output.values.map((_,i)=>Math.max(...x.values.slice(i*width,(i+1)*width))),rowShape,R`m_p=\max_j x_{pj}`,i=>Array.from({length:width},(_,j)=>term(x,i*width+j))),center=add('逐行稳定平移',[x,max],x.values.map((v,i)=>v-max.values[Math.floor(i/width)]),x.shape,R`c_{pj}=x_{pj}-m_p`,i=>[term(x,i),term(max,Math.floor(i/width))]),exp=unary('逐项指数',center,Math.exp,R`u_{pj}=\exp(c_{pj})`),sum=add('逐行指数和',[exp],source.output.values.map((_,i)=>exp.values.slice(i*width,(i+1)*width).reduce((a,b)=>a+b,0)),rowShape,R`Z_p=\sum_j u_{pj}`,i=>Array.from({length:width},(_,j)=>term(exp,i*width+j,1))),log=unary('指数和取对数',sum,Math.log,R`l_p=\log Z_p`)
  finish([max,log,x],i=>[term(max,i,1),term(log,i,1),term(x,i*width,-1)],'减去正样本 logit',R`\ell_p=m_p+l_p-x_{p0}`)
 }else if(op==='row-exp'){
  const maximum=source.inputs[1],width=x.shape.at(-1)!
  const centered=add('减去本行最大值',[x,maximum],x.values.map((v,i)=>v-maximum.values[Math.floor(i/width)]),x.shape,R`c_{ij}=x_{ij}-m_i`,i=>[term(x,i),term(maximum,Math.floor(i/width))])
  finish([centered],i=>[term(centered,i)],'稳定指数',R`u_{ij}=\exp(c_{ij})`)
 }else if(op==='norm-var'&&source.inputs.length){
  const originalTraces=source.output.values.map((_,i)=>source.trace(i)),groupOf=new Map<number,number>()
  originalTraces.forEach((ts,g)=>ts.filter(t=>t.tensor===x.id).forEach(t=>groupOf.set(t.index,g)))
  const mu=source.inputs[1]
  const center=add(mu?'每个元素减去本组均值':'RMS · 保持原始坐标',source.inputs,x.values.map((v,i)=>v-(mu?.values[groupOf.get(i)!]??0)),x.shape,mu?R`c_i=x_i-\mu_{g(i)}`:R`c_i=x_i`,i=>[term(x,i),...mu?[term(mu,groupOf.get(i)!)]:[]])
  const square=unary('每个偏差平方',center,v=>v*v,R`q_i=c_i^2`)
  const reduction:Step={...source,id:`${source.id}:variance`,inputs:[square],trace:i=>{const ids=originalTraces[i].filter(t=>t.tensor===x.id).map(t=>t.index);return ids.map(j=>term(square,j,1/ids.length))},settings:{}}
  const nested=executionBlock(reduction);steps.push(...nested.steps.map(s=>({...s,settings:{...s.settings,owner:source.id}})));select=nested.select;streamed=true
 }else {
  const first=source.trace(0),weighted=first.length>1&&first.every(t=>t.factor!==undefined),max=kind==='MaxPool2d'||op==='row-max'
  if(weighted||max){
   streamed=true;let selected=0,terms=first
   const zero:Tensor={id:`${source.id}:padding`,name:'插零 / 边界填充 0',values:[0],shape:[1],constant:true},padding=source.kind.startsWith('Conv')
   const products=add(max?'当前窗口 · 全部候选值':'当前输出 · 全部逐项乘积',[...source.inputs,...padding?[zero]:[]],terms.map(t=>max?t.value:t.value*t.factor!),[1,terms.length],max?R`v_j=X_{\mathcal P(j)}`:R`p_j=x_jw_j`,i=>[terms[i].index<0?{...terms[i],tensor:zero.id,index:0}:terms[i]])
   let running=max?-Infinity:0
   const prefix=add(max?'顺序比较 · 运行最大值':'顺序累加 · 所有部分和',[products],products.values.map(v=>running=max?Math.max(running,v):running+v),products.shape,max?R`a_j=\max(a_{j-1},v_j)`:R`a_j=a_{j-1}+p_j`,i=>[term(products,i,1),...i?[term(prefix,i-1,1)]:[]])
   finish([prefix],i=>i===selected?[term(prefix,prefix.values.length-1,1)]:source.trace(i),'写回当前输出坐标',source.formula)
   select=(index)=>{selected=index;terms=source.trace(index);products.values=terms.map(t=>max?t.value:t.value*t.factor!);running=max?-Infinity:0;prefix.values=products.values.map(v=>running=max?Math.max(running,v):running+v)}
  }else steps.push({...source,settings:{...source.settings,owner:source.id}})
 }
 return {source,steps,select,streamed}
}

export const activationKinds=new Set(['ReLU','ReLU6','LeakyReLU','Sigmoid','Tanh','SiLU','GELU','Hardswish','Hardsigmoid','Softplus'])
export function executionGraph(run:Run,selected=run.steps,neuralLayers=false){
 const blocks=selected.map(s=>s.settings.symbolic||neuralLayers&&(s.kind==='Linear'||activationKinds.has(s.kind))?{source:s,steps:[{...s,settings:{...s.settings,owner:s.id}}],select:(_index:number)=>{},streamed:false}:executionBlock(s)),steps=blocks.flatMap(b=>b.steps),tensors=[...new Map([...run.tensors,...steps.flatMap(s=>[...s.inputs,s.output])].map(t=>[t.id,t])).values()]
 return {run:{...run,steps,tensors},blocks}
}

/** One output stays in focus through the complete operator, then the sweep
 * advances to the next coordinate on the next network pass. */
export function streamedIndex(count:number,pass:number,manual:number,playing:boolean){return playing?((manual+pass)%count+count)%count:Math.min(manual,count-1)}

/** Back-propagate actual coordinate dependencies to locate the selected
 * output's operands even when statistics have fewer dimensions. */
export function executionCoordinates(block:ExecutionBlock,index:number){
 const byTensor=new Map(block.steps.map(s=>[s.output.id,s])),seen=new Map<string,Set<number>>(),pending:[string,number][]=[[block.source.output.id,index]]
 while(pending.length){const [id,i]=pending.pop()!,step=byTensor.get(id);if(!step||i<0)continue;const indices=seen.get(id)??new Set<number>();if(indices.has(i))continue;indices.add(i);seen.set(id,indices);for(const t of step.trace(i)){if(t.index>=0)pending.push([t.tensor,t.index]);if(t.factorTensor)pending.push([t.factorTensor,t.factorIndex!])}}
 return seen
}
