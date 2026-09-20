import {Engine,size,type Tensor,type Run} from './engine'
import type {Config,ModuleDef} from './models'
const R=String.raw
const source='https://github.com/pytorch/examples/blob/main/vae/main.py'
export const VAE_MODULES:ModuleDef[]=[
 {id:'vae-head',name:'VAE · 双高斯参数头',family:'VAE',format:'vector',description:'共享特征分别投影为均值 μ 与对数方差 log σ²，保留两套完整权重',source},
 {id:'vae-reparameter',name:'VAE · 重参数化采样',family:'VAE',format:'vector',description:'独立 μ / log σ² 输入，标准差变换、Box–Muller 噪声、逐元素缩放与平移',source},
 {id:'vae-mlp',name:'VAE · MLP 编码器',family:'VAE',format:'vector',description:'可调 MLP → 双高斯头 → 标准差 → 重参数化；与官方示例相同的概率参数化',source},
 {id:'vae-conv',name:'VAE · 卷积编码器',family:'VAE',format:'image',description:'教学型可调卷积编码器：步幅 2 卷积、激活、全局平均池化、双高斯头与采样',source:'https://docs.pytorch.org/docs/stable/distributions.html#torch.distributions.normal.Normal'},
 {id:'vae-kl',name:'VAE · 对角高斯 KL',family:'VAE',format:'vector',description:'逐坐标展开 μ² + exp(log σ²) − 1 − log σ²，保留每个样本的 KL 总和',source},
 {id:'vae-poe',name:'VAE · 高斯专家融合',family:'VAE',format:'vector',description:'两个对角高斯专家与标准正态先验：精度相加、精度加权均值、后验采样',source:'https://arxiv.org/abs/1802.05335'},
]

/** Eager, fully recorded diagonal-Gaussian arithmetic; no operator is a decorative proxy. */
export function executeVAE(id:string,c:Config,custom?:number[]):Run{
 const e=new Engine(c.seed),shape=id==='vae-conv'?[c.batch,c.channels,c.spatial,c.spatial]:[c.batch,c.dim];
 const input=e.input=e.tensor(['vae-reparameter','vae-kl','vae-poe'].includes(id)?'均值 μ₁':'输入 X',shape,custom??(c.pattern==='ramp'?Array.from({length:size(shape)},(_,i)=>2*i/Math.max(1,size(shape)-1)-1):c.pattern==='impulse'?Array.from({length:size(shape)},(_,i)=>i===Math.floor(size(shape)/2)?1:0):undefined));
 const unary=(x:Tensor,title:string,op:string,fn:(v:number)=>number,formula:string,detail:string)=>e.record(title,'Identity',[x],e.tensor(title,x.shape,x.values.map(fn)),formula,detail,i=>[e.term(x,i)],{operation:op});
 const scale=(x:Tensor,k:number,title:string)=>e.record(title,'Identity',[x],e.tensor(title,x.shape,x.values.map(v=>v*k)),R`y_i=a x_i`,'逐元素乘以标注的固定系数；张量形状不变。',i=>[e.term(x,i,k)],{operation:'scale',scale:k});
 const exp=(x:Tensor,title:string)=>unary(x,title,'vae-exp',Math.exp,R`y_i=\exp(x_i)`,'指数映射把实数转换为严格正值。');
 const head=(x:Tensor)=>{e.group='后验参数 / 均值';const mu=e.linear(x,c.out,true,'均值 μ 投影');mu.name='均值 μ';e.group='后验参数 / 对数方差';const logvar=e.linear(x,c.out,true,'对数方差 log σ² 投影');logvar.name='对数方差 log σ²';return {mu,logvar}};
 const noise=(shape:number[])=>{
  e.group='标准正态噪声';
  const u=e.tensor('均匀随机数 u₁',shape,Array.from({length:size(shape)},()=>Math.max(1e-12,e.random()))),v=e.tensor('均匀随机数 u₂',shape,Array.from({length:size(shape)},()=>e.random()));
  return e.record('标准正态噪声 ε','Identity',[u,v],e.tensor('标准正态噪声 ε',shape,u.values.map((x,i)=>Math.sqrt(-2*Math.log(x))*Math.cos(2*Math.PI*v.values[i]))),R`\epsilon_i=\sqrt{-2\log u_{1,i}}\cos(2\pi u_{2,i})`,'用 Box–Muller 显式生成标准正态样本。种子固定可复查；与 torch.randn_like 同分布，但不声称随机序列相同。',i=>[e.term(u,i),e.term(v,i)],{operation:'box-muller'});
 };
 const sample=(mu:Tensor,logvar:Tensor)=>{
  e.group='标准差 / 对数尺度';const half=scale(logvar,.5,'对数标准差 ½ log σ²');
  e.group='标准差 / 指数映射';const std=exp(half,'标准差 σ');
  const eps=noise(mu.shape);e.group='重参数化 / 噪声缩放';const scaled=e.binary(std,eps,'mul','随机偏移 σ ⊙ ε');
  e.group='重参数化 / 均值平移';return e.binary(mu,scaled,'add','潜变量 z = μ + σ ⊙ ε');
 };
 let output:Tensor;
 if(id==='vae-head'){const {mu,logvar}=head(input);e.group='后验参数输出';output=e.concat([mu,logvar],1,'后验参数 [μ | log σ²]')}
 else if(id==='vae-mlp'||id==='vae-conv'){
  let h=input;
  for(let i=0;i<c.layers;i++){e.group=`特征编码 / 层 ${i+1}`;h=id==='vae-conv'?e.conv(h,c.hidden,3,2,1,1,1,'空间特征提取'):e.linear(h,c.hidden,true,'编码特征投影');h=e.activation(h,c.activation)}
  if(id==='vae-conv'){e.group='特征编码 / 空间归约';h=e.pool(h,'global');h=e.reshape(h,[c.batch,c.hidden],'通道特征向量')}
  const {mu,logvar}=head(h);output=sample(mu,logvar);
 }else if(id==='vae-poe'){
  const means=[input,e.tensor('专家 2 · 均值 μ₂',shape)],logs=[e.tensor('专家 1 · 对数方差',shape),e.tensor('专家 2 · 对数方差',shape)];
  const precisions:Tensor[]=[],weighted:Tensor[]=[];
  for(let i=0;i<2;i++){e.group=`专家 ${i+1} / 精度`;precisions.push(exp(scale(logs[i],-1,'负对数方差'),'精度 τ = 1 / σ²'));e.group=`专家 ${i+1} / 均值贡献`;weighted.push(e.binary(means[i],precisions[i],'mul','精度加权均值 τ ⊙ μ'))}
  e.group='融合后验 / 精度相加';const prior=e.tensor('标准正态先验精度 τ₀ = 1',shape,Array(size(shape)).fill(1));prior.constant=true;
  const total=e.binary(e.binary(precisions[0],precisions[1],'add','两专家精度之和'),prior,'add','融合精度 τ₀ + τ₁ + τ₂');
  e.group='融合后验 / 均值';const numerator=e.binary(weighted[0],weighted[1],'add','加权均值之和（先验均值为 0）');
  const mu=e.record('融合均值 μ','Identity',[numerator,total],e.tensor('融合均值 μ',shape,numerator.values.map((v,i)=>v/total.values[i])),R`\mu=\frac{\tau_1\mu_1+\tau_2\mu_2}{1+\tau_1+\tau_2}`,'独立对角高斯专家的乘积，包含均值 0、精度 1 的标准正态先验。',i=>[e.term(numerator,i),e.term(total,i)],{operation:'vae-divide'});
  e.group='融合后验 / 方差';const logvar=unary(total,'融合对数方差','vae-negative-log',v=>-Math.log(v),R`\log\sigma^2=-\log\tau`,'融合精度严格为正；取负对数得到重参数化所需的对数方差。');output=sample(mu,logvar);
 }else{
  const logvar=e.tensor('输入 log σ²',shape);
  if(id==='vae-reparameter')output=sample(input,logvar);
  else{
   e.group='KL / 均值平方';const square=unary(input,'均值平方 μ²','vae-square',v=>v*v,R`a_i=\mu_i^2`,'均值偏离标准正态中心的逐坐标代价。');
   e.group='KL / 方差';const variance=exp(logvar,'方差 σ²');
   e.group='KL / 合并二阶矩';const moment=e.binary(square,variance,'add','二阶矩 μ² + σ²');
   e.group='KL / 减去单位先验';const centered=unary(moment,'μ² + σ² − 1','vae-minus-one',v=>v-1,R`b_i=\mu_i^2+\sigma_i^2-1`,'减去单位方差先验的常数项。');
   e.group='KL / 熵修正';const negative=scale(logvar,-1,'负对数方差 −log σ²'),difference=e.binary(centered,negative,'add','逐维 KL 的两倍');
   e.group='KL / 逐坐标贡献';const per=scale(difference,.5,'逐坐标 KL');
   e.group='KL / 潜维归约';const d=c.dim;output=e.record('每个样本的 KL','Identity',[per],e.tensor('每个样本的 KL',[c.batch,1],Array.from({length:c.batch},(_,b)=>per.values.slice(b*d,(b+1)*d).reduce((a,v)=>a+v,0))),R`D_{KL}(q\|\mathcal N(0,I))=\frac12\sum_i(\mu_i^2+e^{\ell_i}-1-\ell_i)`,'仅对潜变量轴求和，保留 Batch 轴。未混入重建损失或 β 权重。',b=>Array.from({length:d},(_,i)=>e.term(per,b*d+i,1)),{operation:'vae-row-sum'});
  }
 }
 return e.run(output);
}
