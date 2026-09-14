/** Explicit eager tensor arithmetic. Every stored scalar has an address and traceable operands. */
export interface Tensor {id:string;name:string;shape:number[];values:number[];parameter?:boolean}
export interface Term {tensor:string;index:number;value:number;factor?:number;label?:string;factorTensor?:string;factorIndex?:number}
export interface Step {id:string;group:string;title:string;kind:string;inputs:Tensor[];output:Tensor;formula:string;detail:string;source:string;trace:(index:number)=>Term[];settings:Record<string,unknown>}
export interface Run {tensors:Tensor[];steps:Step[];output:Tensor;input:Tensor;parameters:number;scalars:number}
export const size=(shape:number[])=>shape.reduce((a,b)=>a*b,1)
export function coords(index:number,shape:number[]){const result=shape.map(()=>0);for(let i=shape.length-1;i>=0;i--){result[i]=index%shape[i];index=Math.floor(index/shape[i])}return result}
export const offset=(c:number[],shape:number[])=>c.reduce((n,x,i)=>n*shape[i]+x,0)
const R=String.raw
export class Engine {
 tensors:Tensor[]=[];steps:Step[]=[];state:number;group='输入';input!:Tensor
 constructor(seed=17){this.state=seed>>>0}
 random(){this.state=(Math.imul(1664525,this.state)+1013904223)>>>0;return this.state/4294967296}
 tensor(name:string,shape:number[],values?:number[],parameter=false):Tensor{if(shape.some(d=>!Number.isInteger(d)||d<1))throw Error('张量各维必须是正整数');if(size(shape)>65536)throw Error('单个张量超过 65,536 个单元。请减小参数；工作站不会抽样或隐藏元素。');if(values&&values.length!==size(shape))throw Error(`${name} 应有 ${size(shape)} 个值，实际为 ${values.length}`);const t={id:`t${this.tensors.length}`,name,shape,values:values??Array.from({length:size(shape)},()=>this.random()*2-1),parameter};if(t.values.some(v=>!Number.isFinite(v)))throw Error('输入必须是有限数值');this.tensors.push(t);return t}
 param(name:string,shape:number[],scale=1,constant?:number){return this.tensor(name,shape,Array.from({length:size(shape)},()=>constant??(this.random()*2-1)*scale),true)}
 term(t:Tensor,i:number,factor?:number,label?:string):Term{return {tensor:t.id,index:i,value:t.values[i],factor,label}}
 record(title:string,kind:string,inputs:Tensor[],out:Tensor,formula:string,detail:string,trace:Step['trace'],settings:Step['settings']={}){this.steps.push({id:`s${this.steps.length}`,group:this.group,title,kind,inputs,output:out,formula,detail,trace,settings,source:`https://docs.pytorch.org/docs/stable/generated/torch.nn.${kind}.html`});return out}
 linear(x:Tensor,out:number,bias=true,title='全连接投影'){
  const n=x.shape.at(-1)!,w=this.param('W',[out,n],1/Math.sqrt(n)),b=bias?this.param('b',[out],1/Math.sqrt(n)):null,shape=[...x.shape.slice(0,-1),out]
  const y=this.tensor(title,shape,Array.from({length:size(shape)},(_,i)=>{const row=Math.floor(i/out),o=i%out;let v=b?.values[o]??0;for(let j=0;j<n;j++)v+=x.values[row*n+j]*w.values[o*n+j];return v}))
  return this.record(title,'Linear',[x,w,...b?[b]:[]],y,R`y_{r,o}=b_o+\sum_i x_{r,i}W_{o,i}`,'对最后一维逐项乘加；全部权重、偏置和输出均可查看。',i=>{const row=Math.floor(i/out),o=i%out;return [...Array.from({length:n},(_,j)=>({...this.term(x,row*n+j,w.values[o*n+j],`W[${o},${j}]`),factorTensor:w.id,factorIndex:o*n+j})),...b?[this.term(b,o,1,'偏置')]:[]]},{bias})
 }
 conv(x:Tensor,out:number,k:number,stride=1,padding=0,groups=1,dilation=1,title='空间卷积',bias=true){
  const [batch,cin,...spatial]=x.shape,rank=spatial.length
  if(cin%groups||out%groups)throw Error('输入和输出通道都必须能被 groups 整除')
  const os=spatial.map(n=>Math.floor((n+2*padding-dilation*(k-1)-1)/stride+1));if(os.some(n=>n<1))throw Error('卷积核大于填充后的输入，请增加输入尺寸或 padding')
  const kernel=Array(rank).fill(k),w=this.param('卷积核',[out,cin/groups,...kernel],1/Math.sqrt(cin/groups*k**rank)),b=bias?this.param('偏置',[out],.1):null,shape=[batch,out,...os]
  const trace=(index:number)=>{const [n,o,...p]=coords(index,shape),g=Math.floor(o/(out/groups)),terms:Term[]=[];for(let ci=0;ci<cin/groups;ci++)for(let q=0;q<k**rank;q++){const kc=coords(q,kernel),where=p.map((v,j)=>v*stride-padding+kc[j]*dilation),wi=offset([o,ci,...kc],w.shape);if(where.every((v,j)=>v>=0&&v<spatial[j]))terms.push({...this.term(x,offset([n,g*cin/groups+ci,...where],x.shape),w.values[wi],`W[${[o,ci,...kc]}]`),factorTensor:w.id,factorIndex:wi});else terms.push({tensor:x.id,index:-1,value:0,factor:w.values[wi],label:'零填充'})}if(b)terms.push(this.term(b,o,1,'偏置'));return terms}
  const y=this.tensor(title,shape,Array.from({length:size(shape)},(_,i)=>trace(i).reduce((s,t)=>s+t.value*(t.factor??1),0)))
  return this.record(title,`Conv${rank}d`,[x,w,...b?[b]:[]],y,R`Y_{n,o,p}=b_o+\sum_{c,u}W_{o,c,u}X_{n,gC+c,\,sp-p_0+du}`,'PyTorch 互相关：核不翻转；groups 划分通道连接，dilation 控制采样间隔。边界零项也列入逐项演算。',trace,{kernel:k,stride,padding,groups,dilation,bias})
 }
 transposeConv(x:Tensor,out:number,k:number,stride=1,padding=0,groups=1,dilation=1,outputPadding=0){
  const [batch,cin,...spatial]=x.shape,rank=spatial.length;if(cin%groups||out%groups)throw Error('转置卷积通道必须能被groups整除');if(outputPadding>=stride)throw Error('当前配置要求 output_padding < stride')
  const os=spatial.map(n=>(n-1)*stride-2*padding+dilation*(k-1)+outputPadding+1),shape=[batch,out,...os],ks=Array(rank).fill(k),w=this.param('转置卷积核',[cin,out/groups,...ks],1/Math.sqrt(out/groups*k**rank)),bias=this.param('偏置',[out],.1)
  const trace=(index:number)=>{const [n,o,...p]=coords(index,shape),group=Math.floor(o/(out/groups)),terms:Term[]=[];for(let ci=0;ci<cin/groups;ci++)for(let q=0;q<k**rank;q++){const kernel=coords(q,ks),source=p.map((v,j)=>(v+padding-kernel[j]*dilation)/stride),wi=offset([group*cin/groups+ci,o%(out/groups),...kernel],w.shape);if(source.every((v,j)=>Number.isInteger(v)&&v>=0&&v<spatial[j]))terms.push({...this.term(x,offset([n,group*cin/groups+ci,...source],x.shape),w.values[wi]),factorTensor:w.id,factorIndex:wi});else terms.push({tensor:x.id,index:-1,value:0,factor:w.values[wi],label:'插零或边界'})}terms.push(this.term(bias,o,1));return terms}
  const output=this.tensor('转置卷积输出',shape,Array.from({length:size(shape)},(_,i)=>trace(i).reduce((s,t)=>s+t.value*t.factor!,0)))
  return this.record('转置卷积',`ConvTranspose${rank}d`,[x,w,bias],output,R`Y_{n,o,p}=b_o+\sum_{c,u}W_{c,o,u}X_{n,c,(p+p_0-du)/s}`,'只读取整除步幅且位于输入内的坐标；其余贡献为零。它是卷积线性算子的转置，不是逆运算。',trace,{kernel:k,stride,padding,groups,dilation,outputPadding})
 }
 interpolate(x:Tensor,mode:'nearest'|'bilinear',scale=2,alignCorners=false){
  const [batch,c,h,w]=x.shape,shape=[batch,c,h*scale,w*scale],trace=(i:number)=>{const [n,ch,y,z]=coords(i,shape);if(mode==='nearest')return [this.term(x,offset([n,ch,Math.floor(y/scale),Math.floor(z/scale)],x.shape),1)];const sy=Math.max(0,Math.min(h-1,alignCorners?y*(h-1)/(h*scale-1):(y+.5)/scale-.5)),sx=Math.max(0,Math.min(w-1,alignCorners?z*(w-1)/(w*scale-1):(z+.5)/scale-.5)),y0=Math.floor(sy),x0=Math.floor(sx),a=sx-x0,b=sy-y0;return [[y0,x0,(1-a)*(1-b)],[y0,Math.min(w-1,x0+1),a*(1-b)],[Math.min(h-1,y0+1),x0,(1-a)*b],[Math.min(h-1,y0+1),Math.min(w-1,x0+1),a*b]].map(([r,s,weight])=>this.term(x,offset([n,ch,r,s],x.shape),weight))};const y=this.tensor('插值输出',shape,Array.from({length:size(shape)},(_,i)=>trace(i).reduce((sum,t)=>sum+t.value*t.factor!,0)))
  return this.record(mode==='nearest'?'最近邻插值':'双线性插值','Upsample',[x],y,mode==='nearest'?R`Y_{y,x}=X_{\lfloor y/s\rfloor,\lfloor x/s\rfloor}`:R`Y=\sum_{i,j\in\{0,1\}}w_{ij}X_{ij},\quad\sum w_{ij}=1`,'依据当前 align_corners 规则反向映射坐标，完整显示各邻点与权重。最近邻保持原值。',trace,{mode,scale,alignCorners})
 }
 activation(x:Tensor,kind:string){
  const erf=(z:number)=>{const sign=z<0?-1:1,t=1/(1+.3275911*Math.abs(z));return sign*(1-(((((1.061405429*t-1.453152027)*t)+1.421413741)*t-.284496736)*t+.254829592)*t*Math.exp(-z*z))}
  const functions:Record<string,(v:number)=>number>={ReLU:v=>Math.max(0,v),ReLU6:v=>Math.min(6,Math.max(0,v)),LeakyReLU:v=>v<0?v*.2:v,Sigmoid:v=>1/(1+Math.exp(-v)),Tanh:Math.tanh,SiLU:v=>v/(1+Math.exp(-v)),GELU:v=>.5*v*(1+erf(v/Math.SQRT2)),Hardswish:v=>v*Math.max(0,Math.min(6,v+3))/6,Hardsigmoid:v=>Math.max(0,Math.min(6,v+3))/6,Softplus:v=>Math.max(0,v)+Math.log1p(Math.exp(-Math.abs(v)))}
  const formulas:Record<string,string>={ReLU:R`y=\max(0,x)`,ReLU6:R`y=\min(6,\max(0,x))`,LeakyReLU:R`y=\max(0,x)+.2\min(0,x)`,Sigmoid:R`y=(1+e^{-x})^{-1}`,Tanh:R`y=\tanh x`,SiLU:R`y=x\sigma(x)`,GELU:R`y=x\Phi(x)`,Hardswish:R`y=x\operatorname{ReLU6}(x+3)/6`,Hardsigmoid:R`y=\operatorname{ReLU6}(x+3)/6`,Softplus:R`y=\log(1+e^x)`}
  return this.record(kind,kind,[x],this.tensor(kind,x.shape,x.values.map(functions[kind])),formulas[kind],'逐元素计算，形状不变。GELU 使用 erf 数值逼近；其余为显式标量公式。',i=>[this.term(x,i)],{})
 }
 binary(a:Tensor,b:Tensor,kind:'add'|'mul',title=kind==='add'?'残差相加':'逐项调制'){
  if(a.shape.length!==b.shape.length||b.shape.some((d,i)=>d!==1&&d!==a.shape[i]))throw Error('广播维度不兼容')
  const bi=(i:number)=>offset(coords(i,a.shape).map((v,j)=>b.shape[j]===1?0:v),b.shape)
  const y=this.tensor(title,a.shape,a.values.map((v,i)=>kind==='add'?v+b.values[bi(i)]:v*b.values[bi(i)]))
  return this.record(title,kind==='add'?'Identity':'Identity',[a,b],y,kind==='add'?R`y_i=a_i+b_i`:R`y_i=a_i b_i`,'沿单例维广播，保持元素真实对应关系。',i=>[this.term(a,i),this.term(b,bi(i))],{operation:kind})
 }
 reshape(x:Tensor,shape:number[],title='重组张量') {if(size(shape)!==x.values.length)throw Error('重排前后元素数必须一致');return this.record(title,'Flatten',[x],this.tensor(title,shape,[...x.values]),R`\operatorname{numel}(Y)=\operatorname{numel}(X)`,'仅重解释连续存储索引，不改变数值或删除元素。',i=>[this.term(x,i)],{shape})}
 permute(x:Tensor,axes:number[],title='维度重排') {const shape=axes.map(i=>x.shape[i]),source=(i:number)=>{const c=coords(i,shape),v=Array(axes.length).fill(0);axes.forEach((a,j)=>v[a]=c[j]);return offset(v,x.shape)};return this.record(title,'Identity',[x],this.tensor(title,shape,Array.from({length:size(shape)},(_,i)=>x.values[source(i)])),R`Y_{i_0,\ldots,i_n}=X_{i_{\pi(0)},\ldots,i_{\pi(n)}}`,'按轴置换实际移动坐标。',i=>[this.term(x,source(i))],{axes})}
 slice(x:Tensor,axis:number,start:number,end:number,title='分离特征组'){const shape=[...x.shape];shape[axis]=end-start;const source=(i:number)=>{const c=coords(i,shape);c[axis]+=start;return offset(c,x.shape)};return this.record(title,'Identity',[x],this.tensor(title,shape,Array.from({length:size(shape)},(_,i)=>x.values[source(i)])),R`Y=X[\ldots,a:b,\ldots]`,'按当前轴范围切片，各元素保留原值。',i=>[this.term(x,source(i))],{axis,start,end})}
 concat(xs:Tensor[],axis:number,title='特征拼接'){const shape=[...xs[0].shape];shape[axis]=xs.reduce((s,x)=>s+x.shape[axis],0);const source=(i:number)=>{const c=coords(i,shape);let k=0;while(c[axis]>=xs[k].shape[axis])c[axis]-=xs[k++].shape[axis];return {t:xs[k],i:offset(c,xs[k].shape)}};return this.record(title,'Identity',xs,this.tensor(title,shape,Array.from({length:size(shape)},(_,i)=>{const s=source(i);return s.t.values[s.i]})),R`Y=\operatorname{cat}(X_1,\ldots,X_m;\mathrm{dim}=d)`,'沿指定轴连接全部元素，不做求和。',i=>{const s=source(i);return [this.term(s.t,s.i)]},{axis})}
 norm(x:Tensor,kind:'BatchNorm2d'|'InstanceNorm2d'|'LayerNorm'|'GroupNorm'|'RMSNorm',groups=1,eps=1e-5,training=true){
  const shape=x.shape,channel=shape[1],dim=shape.at(-1)!,feature=kind==='LayerNorm'||kind==='RMSNorm'?dim:channel
  if(kind==='GroupNorm'&&channel%groups)throw Error('通道数必须能被归一化分组数整除')
  const bucket=(i:number)=>{const c=coords(i,shape);return kind==='LayerNorm'||kind==='RMSNorm'?Math.floor(i/dim):kind==='BatchNorm2d'?c[1]:kind==='GroupNorm'?c[0]*groups+Math.floor(c[1]/(channel/groups)):c[0]*channel+c[1]}
  const rows=new Map<number,number[]>();x.values.forEach((_,i)=>{const key=bucket(i);if(!rows.has(key))rows.set(key,[]);rows.get(key)!.push(i)})
  if(kind==='BatchNorm2d'&&training&&[...rows.values()].some(row=>row.length===1))throw Error('BatchNorm 训练统计每通道至少需要两个值')
  const stats=new Map([...rows].map(([key,ids])=>{const mu=kind==='RMSNorm'?0:ids.reduce((s,i)=>s+x.values[i],0)/ids.length,v=ids.reduce((s,i)=>s+(x.values[i]-mu)**2,0)/ids.length;return [key,kind==='BatchNorm2d'&&!training?{mu:0,v:1}:{mu,v}]}))
  const ordered=[...rows.keys()].sort((a,b)=>a-b),statShape=[ordered.length]
  const mu=this.tensor('归一化均值',statShape,ordered.map(key=>stats.get(key)!.mu)),variance=this.tensor(kind==='RMSNorm'?'均方值':'总体方差',statShape,ordered.map(key=>stats.get(key)!.v))
  this.record('统计归一化均值','Identity',[x],mu,kind==='RMSNorm'?R`\mu=0`:kind==='BatchNorm2d'&&!training?R`\mu=\mathrm{running\_mean}`:R`\mu=|\mathcal I|^{-1}\sum_{i\in\mathcal I}x_i`,'每个统计单元对应一组真实归一化索引。RMSNorm 的中心固定为0；推理态BatchNorm读取初始running_mean。',i=>rows.get(ordered[i])!.map(j=>this.term(x,j)),{operation:'norm-mean',kind,groups,training})
  this.record(kind==='RMSNorm'?'计算均方值':'计算总体方差','Identity',[x,mu],variance,kind==='BatchNorm2d'&&!training?R`v=\mathrm{running\_var}`:R`v=|\mathcal I|^{-1}\sum_{i\in\mathcal I}(x_i-\mu)^2`,'分母是参与统计的元素数，不使用样本方差的N−1。推理态BatchNorm读取初始running_var=1。',i=>rows.get(ordered[i])!.map(j=>this.term(x,j)),{operation:'norm-var',kind,groups,training})
  const gamma=this.param('γ',[feature],1,1),beta=kind==='RMSNorm'?null:this.param('β',[feature],1,0)
  const y=this.tensor(kind,shape,x.values.map((v,i)=>{const s=stats.get(bucket(i))!,c=kind==='LayerNorm'||kind==='RMSNorm'?i%dim:coords(i,shape)[1];return (v-s.mu)/Math.sqrt(s.v+eps)*gamma.values[c]+(beta?.values[c]??0)}))
  return this.record(kind,kind,[x,gamma,...beta?[beta]:[],mu,variance],y,kind==='RMSNorm'?R`y_i=\gamma_i x_i/\sqrt{\operatorname{mean}(x^2)+\epsilon}`:R`y_i=\gamma_i(x_i-\mu)/\sqrt{\operatorname{mean}((x-\mu)^2)+\epsilon}+\beta_i`,`使用实际归一化轴与总体方差。BatchNorm ${training?'使用当前批次统计':'使用初始化 running_mean=0、running_var=1'}；γ=1、β=0为初始化参数。`,i=>{const key=ordered.indexOf(bucket(i)),j=kind==='LayerNorm'||kind==='RMSNorm'?i%dim:coords(i,shape)[1];return [this.term(x,i),this.term(mu,key),this.term(variance,key),this.term(gamma,j),...beta?[this.term(beta,j)]:[]]},{groups,eps,training})
 }
 pool(x:Tensor,kind:'max'|'avg'|'global',k=2,stride=2){const [batch,c,h,w]=x.shape,oh=kind==='global'?1:Math.floor((h-k)/stride)+1,ow=kind==='global'?1:Math.floor((w-k)/stride)+1,shape=[batch,c,oh,ow];const ids=(i:number)=>{const [n,ch,y,z]=coords(i,shape),a:number[]=[];for(let u=0;u<(kind==='global'?h:k);u++)for(let v=0;v<(kind==='global'?w:k);v++)a.push(offset([n,ch,(kind==='global'?0:y*stride)+u,(kind==='global'?0:z*stride)+v],x.shape));return a};const out=this.tensor('池化输出',shape,Array.from({length:size(shape)},(_,i)=>{const a=ids(i).map(j=>x.values[j]);return kind==='max'?Math.max(...a):a.reduce((s,v)=>s+v,0)/a.length}));return this.record(kind==='global'?'全局空间均值':kind==='max'?'最大池化':'平均池化',kind==='global'?'AdaptiveAvgPool2d':kind==='max'?'MaxPool2d':'AvgPool2d',[x],out,kind==='max'?R`Y=\max_{i\in\mathcal P}X_i`:R`Y=|\mathcal P|^{-1}\sum_{i\in\mathcal P}X_i`,'每个输出显式读取所属窗口；最大池化高亮参与比较的所有值。',i=>ids(i).map(j=>this.term(x,j)),{k,stride,kind})}
 softmax(x:Tensor){
  const d=x.shape.at(-1)!,rows=Math.floor(x.values.length/d),shape=[...x.shape.slice(0,-1),1],ids=(i:number)=>Array.from({length:d},(_,j)=>Math.floor(i/d)*d+j)
  const maxima=this.tensor('行最大值',shape,Array.from({length:rows},(_,r)=>Math.max(...x.values.slice(r*d,(r+1)*d))))
  this.record('读取行最大值','Identity',[x],maxima,R`m_i=\max_j S_{ij}`,'减去行最大值避免指数溢出；不会改变Softmax结果。',i=>ids(i*d).map(j=>this.term(x,j)),{operation:'row-max'})
  const exp=this.tensor('稳定指数',x.shape,x.values.map((v,i)=>Math.exp(v-maxima.values[Math.floor(i/d)])))
  this.record('逐元素稳定指数','Identity',[x,maxima],exp,R`u_{ij}=\exp(S_{ij}-m_i)`,'所有指数值都保留为独立的中间张量。',i=>[this.term(x,i),this.term(maxima,Math.floor(i/d))],{operation:'row-exp'})
  const sums=this.tensor('指数行和',shape,Array.from({length:rows},(_,r)=>exp.values.slice(r*d,(r+1)*d).reduce((a,b)=>a+b,0)))
  this.record('沿 key 轴求和','Identity',[exp],sums,R`Z_i=\sum_j u_{ij}`,'完整归约当前行的所有key。',i=>ids(i*d).map(j=>this.term(exp,j,1)),{operation:'row-sum'})
  return this.record('逐行 Softmax','Softmax',[x,exp,sums],this.tensor('注意力概率',x.shape,exp.values.map((v,i)=>v/sums.values[Math.floor(i/d)])),R`p_{ij}=u_{ij}/Z_i`,'使用已显式展开的稳定指数与行和；每行总和为1。',i=>[this.term(exp,i,1/sums.values[Math.floor(i/d)]),this.term(sums,Math.floor(i/d))],{})
 }

 matmul(a:Tensor,b:Tensor,transposeB=false,scale=1){const m=a.shape.at(-2)!,k=a.shape.at(-1)!,n=b.shape.at(transposeB?-2:-1)!,bk=b.shape.at(transposeB?-1:-2)!;if(k!==bk||size(a.shape.slice(0,-2))!==size(b.shape.slice(0,-2)))throw Error('矩阵维度不匹配');const shape=[...a.shape.slice(0,-2),m,n],indices=(i:number,j:number)=>{const batch=Math.floor(i/(m*n)),r=Math.floor(i/n)%m,c=i%n;return [batch*m*k+r*k+j,batch*k*n+(transposeB?c*k+j:j*n+c)]};const y=this.tensor('矩阵乘积',shape,Array.from({length:size(shape)},(_,i)=>{let sum=0;for(let j=0;j<k;j++){const [ai,bi]=indices(i,j);sum+=a.values[ai]*b.values[bi]}return sum*scale}));return this.record(transposeB?'QKᵀ 缩放点积':'注意力加权 V','Identity',[a,b],y,transposeB?R`S_{ij}=\sum_rQ_{ir}K_{jr}/\sqrt{d_h}`:R`O_{ir}=\sum_j A_{ij}V_{jr}`,'每个输出保留完整归约维；动画依次展开全部乘积。',i=>Array.from({length:k},(_,j)=>{const [ai,bi]=indices(i,j);return {...this.term(a,ai,b.values[bi]*scale,`${b.name}[${coords(bi,b.shape)}] × ${scale.toPrecision(3)}`),factorTensor:b.id,factorIndex:bi}}),{transposeB,scale})}
 causal(x:Tensor){const n=x.shape.at(-1)!,y=this.tensor('因果掩码 logits',x.shape,x.values.map((v,i)=>i%n>Math.floor(i/n)%n?-1e30:v));return this.record('禁止读取未来 token','Identity',[x],y,R`S_{ij}\leftarrow\begin{cases}S_{ij}&j\le i\\-\infty&j>i\end{cases}`,'遮挡项用−1e30表示；在当前有限数值下 softmax 精确下溢为0。',i=>[this.term(x,i)],{operation:'causal'})}
 dropout(x:Tensor,p=.1){const mask=this.tensor('Bernoulli 掩码',x.shape,x.values.map(()=>this.random()<p?0:1)),y=this.tensor('Dropout',x.shape,x.values.map((v,i)=>v*mask.values[i]/(1-p)));return this.record(`Dropout p=${p}`,'Dropout',[x,mask],y,R`Y=M\odot X/(1-p)`,'执行固定种子的训练态 Bernoulli 掩码；p=0 时恒等。模型配置可在训练态与推理态之间切换。',i=>[this.term(x,i,mask.values[i]/(1-p)),this.term(mask,i)],{p})}
 run(output:Tensor):Run {const scalars=this.tensors.reduce((s,t)=>s+t.values.length,0);if(scalars>500000)throw Error('本配置超过500,000个显式数值。请减小参数后运行；没有隐藏层或抽样替身。');return {tensors:this.tensors,steps:this.steps,output,input:this.input,parameters:this.tensors.filter(t=>t.parameter).reduce((s,t)=>s+t.values.length,0),scalars}}
}
