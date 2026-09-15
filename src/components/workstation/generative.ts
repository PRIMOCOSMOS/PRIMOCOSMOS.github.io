import {Engine,coords,offset,size,type Tensor,type Run} from './engine'
import type {Config,ModuleDef} from './models'
const R=String.raw
const cut='https://github.com/taesungp/contrastive-unpaired-translation/blob/master/models/'
const ddpm='https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/'
export const GENERATIVE:ModuleDef[]=[
 {id:'cut-generator',name:'CUT · ResNet 生成器',family:'CUT',format:'image',description:'反射填充、下采样、残差堆栈、上采样；no_antialias 路径',source:cut+'networks.py'},
 {id:'cut-patchgan',name:'CUT · PatchGAN 判别器',family:'CUT',format:'image',description:'局部感受野、InstanceNorm 与逐 patch 的真实性分数',source:cut+'networks.py'},
 {id:'cut-projector',name:'CUT · Patch 投影头',family:'CUT',format:'sequence',description:'两层 MLP 与逐 patch L2 归一化，保留全部给定 patch',source:cut+'networks.py'},
 {id:'cut-nce',name:'CUT · PatchNCE',family:'CUT',format:'sequence',description:'对应位置正样本、图内负样本、detach 与温度交叉熵',source:cut+'patchnce.py'},
 {id:'cut-objective',name:'CUT · 对抗训练目标',family:'CUT',format:'sequence',description:'LSGAN 的生成器和判别器损失，分别保留各项贡献',source:cut+'networks.py'},
 {id:'diffusion-forward',name:'Diffusion · 正向加噪',family:'Diffusion',format:'image',description:'完整 β 调度、α 累乘、高斯噪声与闭式重参数化',source:ddpm+'diffusion_utils.py'},
 {id:'diffusion-reverse',name:'DDPM · 反向采样一步',family:'Diffusion',format:'image',description:'预测 x₀、裁剪、后验均值与方差、末步无噪声',source:ddpm+'diffusion_utils.py'},
 {id:'diffusion-ddim',name:'DDIM · 广义采样一步',family:'Diffusion',format:'image',description:'可调 η 与时间跨度，分别显示确定性方向和随机项',source:'https://github.com/ermongroup/ddim/blob/main/functions/denoising.py'},
 {id:'diffusion-time',name:'Diffusion · 时间嵌入',family:'Diffusion',format:'vector',description:'指数频率、sin/cos 与两层时间条件投影',source:ddpm+'nn.py'},
 {id:'diffusion-unet',name:'Diffusion · 条件 U-Net',family:'Diffusion',format:'image',description:'两尺度可配置 U-Net：时间残差、空间注意力与完整跳连',source:ddpm+'models/unet.py'},
 {id:'diffusion-train',name:'DDPM · 噪声预测训练',family:'Diffusion',format:'image',description:'正向加噪 → 条件 U-Net → 逐元素误差 → 均方损失',source:ddpm+'diffusion_utils.py'},
]
export function moduleConfig(id:string,c:Config):Config{return {...c,spatial:Math.min(c.spatial,10),layers:Math.min(c.layers,id==='cut-generator'?9:id==='cut-patchgan'?3:4),...id==='diffusion-ddim'?{clipDenoised:false}:id==='cut-generator'?{spatial:8}:id==='cut-patchgan'?{spatial:16,layers:2}:['diffusion-unet','diffusion-train'].includes(id)?{spatial:8}:{} }}

export function executeGenerative(id:string,c:Config,custom?:number[]):Run{
 const def=GENERATIVE.find(m=>m.id===id)!,e=new Engine(c.seed),shape=id==='diffusion-time'?[c.batch,1]:def.format==='image'?[c.batch,c.channels,c.spatial,c.spatial]:def.format==='sequence'?[c.batch,c.tokens,c.dim]:[c.batch,c.dim]
 let x=e.input=e.tensor(id==='diffusion-time'?'时间步 t':'输入 X',shape,custom??(id==='diffusion-time'?Array(c.batch).fill(Math.min(c.timeStep,c.diffusionSteps-1)):c.pattern==='ramp'?Array.from({length:size(shape)},(_,i)=>2*i/Math.max(1,size(shape)-1)-1):c.pattern==='impulse'?Array.from({length:size(shape)},(_,i)=>i===Math.floor(size(shape)/2)?1:0):undefined))
 const group=(name:string)=>{e.group=name},unary=(t:Tensor,name:string,fn:(v:number)=>number,formula:string,operation:string,settings:Record<string,unknown>={})=>e.record(name,'Identity',[t],e.tensor(name,t.shape,t.values.map(fn)),formula,'完整逐元素运算；当前形状中的每个数值都参与。',i=>[e.term(t,i)],{operation,...settings})
 let scheduleAb:Tensor|undefined,scheduleBeta:Tensor|undefined;const gathered=new Map<string,Tensor>();
 const gather=(table:Tensor,index:number)=>{const key=table.id+':'+index;if(gathered.has(key))return gathered.get(key)!;const result=index<0?e.tensor('初始信号保留率',[1],[1]):e.slice(table,0,index,index+1,table.name+` [${index}]`);gathered.set(key,result);return result}
 const scale=(input:Tensor,k:number,name:string)=>{
  let coefficient:Tensor|undefined;
  if(scheduleAb&&scheduleBeta){const t=Math.min(c.timeStep,c.diffusionSteps-1),earlier=Math.min(c.previousStep,t-1),source={a:()=>gather(scheduleAb!,t),p:()=>gather(scheduleAb!,t-1),b:()=>gather(scheduleBeta!,t),s:()=>gather(scheduleAb!,earlier)};
   const definitions:Record<string,{keys:(keyof typeof source)[];formula:string}>={
    '保留信号':{keys:['a'],formula:R`\sqrt{\bar\alpha_t}`},'注入噪声':{keys:['a'],formula:R`\sqrt{1-\bar\alpha_t}`},
    '缩放带噪输入':{keys:['a'],formula:R`1/\sqrt{\bar\alpha_t}`},'移除预测噪声':{keys:['a'],formula:R`-\sqrt{1/\bar\alpha_t-1}`},
    'x₀ 后验贡献':{keys:['a','p','b'],formula:R`\frac{\beta_t\sqrt{\bar\alpha_{t-1}}}{1-\bar\alpha_t}`},
    'x_t 后验贡献':{keys:['a','p','b'],formula:R`\frac{(1-\bar\alpha_{t-1})\sqrt{1-\beta_t}}{1-\bar\alpha_t}`},
    '后验随机项':{keys:['a','p','b'],formula:R`\mathbf1_{t>0}\sqrt{\frac{\beta_t(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}}`},
    '干净样本方向':{keys:['s'],formula:R`\sqrt{\bar\alpha_s}`},
    '预测噪声方向':{keys:['a','s'],formula:R`\sqrt{1-\bar\alpha_s-\eta^2\frac{1-\bar\alpha_s}{1-\bar\alpha_t}(1-\bar\alpha_t/\bar\alpha_s)}`},
    'η 控制的随机项':{keys:['a','s'],formula:R`\eta\sqrt{\frac{1-\bar\alpha_s}{1-\bar\alpha_t}(1-\bar\alpha_t/\bar\alpha_s)}`},
   };const spec=definitions[name];if(spec){const inputs=spec.keys.map(key=>source[key]());coefficient=e.record(name+'系数','Identity',inputs,e.tensor(name+'系数',[1],[k]),spec.formula,'系数从调度中的真实坐标读取；完整保留当前步和前一步的条件关系。',()=>inputs.map(t=>e.term(t,0)),{operation:'diffusion-coefficient',name,t,eta:c.eta,keys:spec.keys})}
  }
  return e.record(name,'Identity',[input,...coefficient?[coefficient]:[]],e.tensor(name,input.shape,input.values.map(v=>v*k)),R`y_i=a x_i`,'同一系数作用于全部坐标；若来自噪声调度，则保留系数计算的上游连接。',i=>[{...e.term(input,i,k),...coefficient?{factorTensor:coefficient.id,factorIndex:0}:{}}],{operation:'scale',scale:k})
 }

 const sum=(t:Tensor,name:string,mean=false)=>e.record(name,'Identity',[t],e.tensor(name,[1],[t.values.reduce((a,b)=>a+b,0)/(mean?t.values.length:1)]),mean?R`y=N^{-1}\sum_i x_i`:R`y=\sum_i x_i`,'完整归约全部元素。',()=>t.values.map((_,i)=>e.term(t,i,mean?1/t.values.length:1)),{operation:'reduce-all',mean})
 const gaussian=(s:number[],name:string)=>{const u=e.tensor('均匀随机数 u₁',s,Array.from({length:size(s)},()=>Math.max(1e-12,e.random()))),v=e.tensor('均匀随机数 u₂',s,Array.from({length:size(s)},()=>e.random()));return e.record(name,'Identity',[u,v],e.tensor(name,s,u.values.map((a,i)=>Math.sqrt(-2*Math.log(a))*Math.cos(2*Math.PI*v.values[i]))),R`\epsilon=\sqrt{-2\log u_1}\cos(2\pi u_2)`,'Box–Muller 变换。固定种子可重复，输出为标准正态样本。',i=>[e.term(u,i),e.term(v,i)],{operation:'box-muller'})}
 const normalize=(t:Tensor)=>{const d=t.shape.at(-1)!,norm=e.record('逐 patch L2 范数','Identity',[t],e.tensor('L2 范数',[...t.shape.slice(0,-1),1],Array.from({length:t.values.length/d},(_,r)=>Math.sqrt(t.values.slice(r*d,(r+1)*d).reduce((a,b)=>a+b*b,0))+1e-7)),R`n_p=\sqrt{\sum_d h_{pd}^2}+10^{-7}`,'对应 CUT Normalize(2)，epsilon 加在范数外。',i=>Array.from({length:d},(_,j)=>e.term(t,i*d+j)),{operation:'l2-norm',eps:1e-7});return e.record('归一化 patch 特征','Identity',[t,norm],e.tensor('单位 patch 特征',t.shape,t.values.map((v,i)=>v/norm.values[Math.floor(i/d)])),R`z_{pd}=h_{pd}/n_p`,'沿特征维归一化，patch 坐标不变。',i=>[e.term(t,i),e.term(norm,Math.floor(i/d))],{operation:'row-divide'})}
 const reflect=(t:Tensor,p:number)=>{const [b,ch,h,w]=t.shape;if(p>=h||p>=w)throw Error('反射填充必须小于空间边长；请增大输入尺寸');const s=[b,ch,h+2*p,w+2*p],mirror=(n:number,len:number)=>n<0?-n:n>=len?2*len-2-n:n,source=(i:number)=>{const [n,c,y,z]=coords(i,s);return offset([n,c,mirror(y-p,h),mirror(z-p,w)],t.shape)};return e.record('反射边界填充','Identity',[t],e.tensor('反射填充',s,Array.from({length:size(s)},(_,i)=>t.values[source(i)])),R`X_{-i}=X_i,\quad X_{H-1+i}=X_{H-1-i}`,'边界值使用反射坐标，不引入零填充。',i=>[e.term(t,source(i))],{operation:'reflect-pad',padding:p})}
 const instance=(t:Tensor)=>e.norm(t,'InstanceNorm2d',1,1e-5,true,false),relu=(t:Tensor)=>e.activation(t,'ReLU'),conv=(t:Tensor,out:number,k=3,stride=1,pad=1)=>e.conv(t,out,k,stride,pad,1,1,'卷积',true)
 if(id==='cut-generator'){
  if(c.spatial%4)throw Error('CUT 生成器的两个降采样尺度要求边长能被4整除。')
  group('输入映射');x=relu(instance(conv(reflect(x,3),c.out,7,1,0)))
  for(let i=0;i<2;i++){group(`下采样 ${i+1}`);x=relu(instance(conv(x,c.out*2**(i+1),3,2)))}
  for(let i=0;i<c.layers;i++){group(`残差块 ${i+1}`);const skip=x;x=relu(instance(conv(reflect(x,1),x.shape[1],3,1,0)));x=instance(conv(reflect(x,1),x.shape[1],3,1,0));x=e.binary(x,skip,'add')}
  for(let i=0;i<2;i++){group(`上采样 ${i+1}`);x=relu(instance(e.transposeConv(x,c.out*2**(1-i),3,2,1,1,1,1)))}
  group('图像生成');x=e.activation(conv(reflect(x,3),c.channels,7,1,0),'Tanh')
 }else if(id==='cut-patchgan'){
  group('局部判别入口');x=e.activation(conv(x,c.out,4,2,1),'LeakyReLU')
  for(let i=1;i<c.layers;i++){group(`判别尺度 ${i+1}`);x=e.activation(instance(conv(x,c.out*Math.min(2**i,8),4,2,1)),'LeakyReLU')}
  group('局部真实性');x=e.activation(instance(conv(x,c.out*Math.min(2**c.layers,8),4,1,1)),'LeakyReLU');x=conv(x,1,4,1,1)
 }else if(id==='cut-projector'){group('逐 patch 投影');x=normalize(e.linear(relu(e.linear(x,c.hidden)),c.hidden))
 }else if(id==='cut-nce'){
  group('独立 patch 输入');const q=x,k0=e.tensor('对应域 key 特征',shape),k=e.record('停止 key 梯度','Identity',[k0],e.tensor('detach key',shape,[...k0.values]),R`k=\operatorname{stopgrad}(k_0)`,'前向数值相同；训练反传不更新 key 分支。这里显示前向，未运行自动微分。',i=>[e.term(k0,i)],{operation:'detach'})
  const count=c.tokens,batch=c.allNegatives?1:c.batch,n=c.allNegatives?count*c.batch:count,d=c.dim
  group('正样本与图内负样本');const qs=e.reshape(q,[batch,n,d]),ks=e.reshape(k,[batch,n,d]),similarity=e.matmul(qs,ks,true)
  const positives=e.record('提取同位置正样本','Identity',[similarity],e.tensor('正样本相似度',[batch,n,1],Array.from({length:batch*n},(_,i)=>similarity.values[Math.floor(i/n)*n*n+i%n*n+i%n])),R`s_p^+=q_p^\top k_p`,'相同空间 patch 构成正对。',i=>[e.term(similarity,Math.floor(i/n)*n*n+i%n*n+i%n)],{operation:'diagonal'})
  const negatives=e.record('排除负样本对角线','Identity',[similarity],e.tensor('负样本矩阵',similarity.shape,similarity.values.map((v,i)=>i%n===Math.floor(i/n)%n?-10:v)),R`s_{pp}^-=-10`,'严格对应官方实现：对角线设为−10，再除以温度；不是零，也不是未除温度的−∞。',i=>i%n===Math.floor(i/n)%n?[]:[e.term(similarity,i)],{operation:'mask-diagonal',value:-10})
  group('温度与对比损失');const logits=scale(e.concat([positives,negatives],2),1/c.temperature,'温度缩放'),prob=e.softmax(logits),loss=e.record('正样本负对数似然','Identity',[logits,prob],e.tensor('各 patch 对比损失',[batch,n],Array.from({length:batch*n},(_,i)=>(()=>{const row=logits.values.slice(i*(n+1),(i+1)*(n+1)),max=Math.max(...row);return max+Math.log(row.reduce((s,v)=>s+Math.exp(v-max),0))-row[0]})())),R`\ell_p=-\log \frac{\exp(s_p^+/\tau)}{\exp(s_p^+/\tau)+\sum_j\exp(s_{pj}^-/\tau)}`,'保留每个 patch 的损失，再求平均。',i=>Array.from({length:n+1},(_,j)=>e.term(logits,i*(n+1)+j)),{operation:'cross-entropy-zero'});x=sum(loss,'平均 PatchNCE',true)
 }else if(id==='cut-objective'){
  group('LSGAN 判别分数');const fake=x,real=e.tensor('真实域判别分数',shape),sq=(t:Tensor,target:number)=>unary(t,`目标 ${target} 的平方误差`,v=>(v-target)**2,R`(D(x)-a)^2`,'target-square',{target})
  group('生成器对抗损失');const g=sum(sq(fake,1),'L_G',true)
  group('判别器对抗损失');const realLoss=sum(sq(real,1),'真实域损失',true),fakeLoss=sum(sq(fake,0),'生成域损失',true),d=scale(e.binary(realLoss,fakeLoss,'add'),.5,'L_D')
  group('独立优化目标');x=e.concat([g,d],0,'[生成器损失, 判别器损失]')
 }else{
  const total=c.diffusionSteps,t=Math.min(c.timeStep,total-1);if(total<2)throw Error('扩散步数至少为2')
  const schedule=()=>{group('噪声调度');const beta=e.tensor('β 线性调度',[total],Array.from({length:total},(_,i)=>1e-4+(.02-1e-4)*i/(total-1)));scheduleBeta=beta;const alpha=unary(beta,'α = 1 − β',v=>1-v,R`\alpha_t=1-\beta_t`,'one-minus');let acc=1;const result=e.record('累乘信号保留率','Identity',[alpha],e.tensor('ᾱ 调度',[total],alpha.values.map(v=>acc*=v)),R`\bar\alpha_t=\prod_{s=0}^{t}\alpha_s`,'索引 t=0 表示第一次加噪，与 Ho 等的官方实现一致。',i=>Array.from({length:i+1},(_,j)=>e.term(alpha,j)),{operation:'cumprod'});scheduleAb=result;return result}
  const condition=()=>{group('时间条件');const requested=id==='diffusion-time'?c.dim:c.out,dim=Math.max(4,requested+(requested%2)),half=dim/2,tt=id==='diffusion-time'?e.input:e.tensor('时间步 t',[c.batch,1],Array(c.batch).fill(t));const embed=e.record('正弦余弦时间编码','Identity',[tt],e.tensor('时间编码',[c.batch,dim],Array.from({length:c.batch*dim},(_,i)=>{const j=i%dim,f=Math.exp(-Math.log(10000)*(j%half)/(half-1));return j<half?Math.sin(tt.values[Math.floor(i/dim)]*f):Math.cos(tt.values[Math.floor(i/dim)]*f)})),R`\omega_i=10000^{-i/(d/2-1)},\quad e_t=[\sin(t\omega),\cos(t\omega)]`,'指数频率与 sin/cos 两半拼接，时间输入相同的批次共享相同编码。',i=>[e.term(tt,Math.floor(i/dim))],{operation:'time-embedding',dim});return e.linear(e.activation(e.linear(embed,c.out*4),'SiLU'),c.out*4)}
  const denoise=(input:Tensor)=>{
   if(c.spatial%2)throw Error('两尺度 U-Net 要求空间边长为偶数。')
   const temb=condition(),gn=(z:Tensor)=>{let g=Math.min(32,z.shape[1]);while(z.shape[1]%g)g--;return e.norm(z,'GroupNorm',g,1e-6)},sw=(z:Tensor)=>e.activation(z,'SiLU')
   const res=(z:Tensor,width:number)=>{const skip=z;let h=conv(sw(gn(z)),width);const time=e.reshape(e.linear(sw(temb),width),[c.batch,width,1,1],'时间条件广播');h=e.binary(h,time,'add','注入时间条件');h=conv(e.dropout(sw(gn(h)),c.training?.1:0),width);return e.binary(h,skip.shape[1]===width?skip:conv(skip,width,1,1,0),'add','时间残差汇合')}
   const attention=(z:Tensor)=>{const [b,ch,h,w]=z.shape,n=h*w,base=z,norm=gn(z),project=(name:string)=>e.permute(e.reshape(e.conv(norm,ch,1,1,0,1,1,name),[b,ch,n]),[0,2,1],name+' 空间序列');const q=project('Q'),k=project('K'),v=project('V'),a=e.softmax(e.matmul(q,k,true,1/Math.sqrt(ch))),out=e.reshape(e.permute(e.matmul(a,v),[0,2,1]),[b,ch,h,w]);return e.binary(base,conv(out,ch,1,1,0),'add','空间注意力残差')}
   group('U-Net 输入');let h=conv(input,c.out);const skips=[h]
   for(let level=0;level<2;level++){for(let j=0;j<c.layers;j++){group(`编码尺度 ${level+1} / 残差 ${j+1}`);h=res(h,c.out*2**level);if(level===1)h=attention(h);skips.push(h)}if(level===0){group('降采样');const source=h,[b,ch,height,width]=h.shape,ps=[b,ch,height+1,width+1],pad=e.record('SAME 下采样边界','Identity',[source],e.tensor('右下零填充',ps,Array.from({length:size(ps)},(_,i)=>{const [n,c,y,x]=coords(i,ps);return y<height&&x<width?source.values[offset([n,c,y,x],source.shape)]:0})),R`H'=H+1,\quad W'=W+1`,'偶数输入、3×3核、stride=2 的 TensorFlow SAME：前侧不补，右侧和下侧各补1。',i=>{const [n,c,y,x]=coords(i,ps);return y<height&&x<width?[e.term(source,offset([n,c,y,x],source.shape))]:[]},{operation:'pad-right-bottom'});h=conv(pad,h.shape[1],3,2,0);skips.push(h)}}
   group('瓶颈 / 残差—注意力—残差');h=res(h,h.shape[1]);h=attention(h);h=res(h,h.shape[1])
   for(let level=1;level>=0;level--){for(let j=0;j<=c.layers;j++){group(`解码尺度 ${level+1} / 残差 ${j+1}`);h=res(e.concat([h,skips.pop()!],1,'拼接编码跳连'),c.out*2**level);if(level===1)h=attention(h)}if(level===1){group('升采样');h=conv(e.interpolate(h,'nearest',2),h.shape[1])}}
   group('预测噪声 εθ');return conv(sw(gn(h)),c.channels)
  }
  if(id==='diffusion-time')x=condition()
  else if(id==='diffusion-unet')x=denoise(x)
  else{
   const abar=schedule(),ab=abar.values[t],prev=t===0?1:abar.values[t-1],beta=1e-4+(.02-1e-4)*t/(total-1)
   if(id==='diffusion-forward'||id==='diffusion-train'){
    group('正向重参数化');const noise=gaussian(shape,'标准正态 ε'),signal=scale(x,Math.sqrt(ab),'保留信号'),random=scale(noise,Math.sqrt(1-ab),'注入噪声');x=e.binary(signal,random,'add','带噪样本 x_t')
    if(id==='diffusion-train'){const pred=denoise(x);group('噪声预测目标');const diff=e.binary(pred,scale(noise,-1,'负的目标噪声'),'add','预测误差');x=sum(unary(diff,'逐坐标平方误差',v=>v*v,R`(\epsilon_\theta-\epsilon)^2`,'target-square',{target:0}),'噪声预测 MSE',true)}
   }else{
    group('由预测噪声恢复 x₀');const xt=x,eps=e.tensor('外部去噪器预测 εθ',shape);let x0=e.binary(scale(xt,1/Math.sqrt(ab),'缩放带噪输入'),scale(eps,-Math.sqrt(1/ab-1),'移除预测噪声'),'add','预测干净样本');if(c.clipDenoised)x0=unary(x0,'裁剪预测到 [−1,1]',v=>Math.max(-1,Math.min(1,v)),R`\hat x_0\leftarrow\operatorname{clip}(\hat x_0,-1,1)`,'clip')
    if(id==='diffusion-reverse'){group('DDPM 后验分布');const a=beta*Math.sqrt(prev)/(1-ab),b=(1-prev)*Math.sqrt(1-beta)/(1-ab),mean=e.binary(scale(x0,a,'x₀ 后验贡献'),scale(xt,b,'x_t 后验贡献'),'add','后验均值');group('后验采样');const noise=gaussian(shape,'反向标准正态 z'),variance=beta*(1-prev)/(1-ab);x=e.binary(mean,scale(noise,t===0?0:Math.sqrt(variance),'后验随机项'),'add','反向样本 x_(t−1)')}
    else{const earlier=Math.min(c.previousStep,t-1),ap=earlier<0?1:abar.values[earlier],sigma=c.eta*Math.sqrt((1-ap)/(1-ab)*(1-ab/ap));group('DDIM 方向与随机性');const clean=scale(x0,Math.sqrt(ap),'干净样本方向'),direction=scale(eps,Math.sqrt(Math.max(0,1-ap-sigma*sigma)),'预测噪声方向'),z=gaussian(shape,'DDIM 标准正态 z');x=e.binary(e.binary(clean,direction,'add'),scale(z,sigma,'η 控制的随机项'),'add','DDIM 下一样本')}
   }
  }
 }
 for(const step of e.steps)step.source=def.source
 return e.run(x)
}
