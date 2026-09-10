import {mathSpec,type MathSpec,type MathStage,demoValues,mean,normalizeValues} from '../mmhvae/mathematics'
import type {Glyph} from '../mmhvae/anatomy'
export type Op='tensor'|'linear'|'conv'|'pool'|'reshape'|'split'|'leaky01'|'leaky02'|'tanh'|'softplus'|'norm-affine'|'norm'|'adain'|'spectral'|'gaussian'|'sample'|'latent-mask'|'spatial-mask'|'detach'|'poe-precision'|'likelihood'|'kl'|'fft'|'ifft'|'complex-multiply'|'sum'|'add'|'multiply'|'gradient'|'l1'|'lsgan'|'subtract'|'scale'|'magnitude'|'rescale'|'pad'|'unpad'|'bilinear'|'broadcast'
type Step=[string,string,string]
const R=String.raw
const recipes:Partial<Record<Op,Step[]>>={
 leaky01:[['读取独立坐标',R`x_i=X_i`,'不混合位置。此仓库的 nn.LeakyReLU() 使用 PyTorch 默认负斜率 0.01。'],['正负半轴分流',R`f(x)=\max(x,0)+0.01\min(x,0)`,'正值保持，负值乘 0.01；形状和索引均不变。'],['保留小幅负响应',R`f'(x)=\begin{cases}1&x>0\\0.01&x<0\end{cases}`,'共享/私有 VAE 的浅层 MLP 使用此非线性；不是 PnP-CoSMo 的 0.2。']],
 leaky02:[['读取独立响应',R`x_i=X_i`,'卷积或线性层的输出逐项进入非线性。'],['按符号缩放',R`f(x)=\max(x,0)+0.2\min(x,0)`,'layers.py 显式指定 LeakyReLU(0.2)。'],['写回原位置',R`Y_i=f(X_i)`,'负半轴保留梯度，不改变通道或空间尺寸。']],
 softplus:[['读取无约束尺度参数',R`a_i\in\mathbb R`,'变量名 log_sigma 不意味着执行 exp；公开实现使用 softplus。'],['平滑映射到正数',R`s_i=\log(1+e^{a_i})`,'输出是标准差，不能直接解释为方差。'],['构造可用尺度',R`s_i>0,\quad\operatorname{Var}(z_i)=s_i^2`,'先验、后验与观测噪声的角色由上一级区分。']],
 'norm-affine':[['通道内统计',R`\mu_c=\frac1{HW}\sum_{h,w}x_{chw},\quad v_c=\frac1{HW}\sum_{h,w}(x_{chw}-\mu_c)^2`,'每个样本、每个通道独立统计；不跨 batch。'],['标准化空间响应',R`\hat x_{chw}=\frac{x_{chw}-\mu_c}{\sqrt{v_c+10^{-5}}}`,'示例展示中心化后除以标准差的实际结果。'],['可学习仿射恢复',R`y_{chw}=\gamma_c\hat x_{chw}+\beta_c`,'NYU content encoder 的 InstanceNorm2d 使用 affine=True。AdaIN 内部则关闭固定 affine。']],
 adain:[['标准化 content',R`\hat c=(c-\mu(c))/\sqrt{v(c)+10^{-5}}`,'每个内容通道独立消除自身均值和尺度。'],['广播风格条件',R`(\gamma,\beta)=\operatorname{chunk}(\operatorname{LeakyReLU}_{0.2}(Ws+b),2)`,'这里的 fc 是带 LeakyReLU 的 LinearBlock；随后把风格参数广播到全部空间位置。'],['风格重标定',R`y=\beta+(1+\gamma)\odot\hat c`,'忠实对应 torch.addcmul(beta, x, 1 + gamma)，保留加一偏移。']],
 spectral:[['将核视作矩阵',R`W\in\mathbb R^{C_{out}\times(C_{in}k^2)}`,'光谱归一化约束权重矩阵的最大奇异值，不是权重向量 L2 归一化。'],['交替幂迭代',R`v\leftarrow W^Tu/\|W^Tu\|_2,\quad u\leftarrow Wv/\|Wv\|_2`,'u、v 交替更新估计主奇异方向；示例使用固定矩阵。'],['归一化谱尺度',R`\hat\sigma=u^TWv,\quad\bar W=W/\hat\sigma`,'训练时更新幂迭代缓冲；推理使用已有估计。不把它表示成额外的图像层。']],
 'latent-mask':[['读取潜坐标和固定掩码',R`z=[z^{p1},z^{sh},z^{p2}]`,'2+2+2 配置有六维。模态 1 的掩码为 [1,1,1,1,0,0]；模态 2 相反。'],['逐维选择',R`\tilde z_j=s_j\odot z_j`,'不活跃的维度置零；跨视图的两次掩码求交，仅剩共享维度。'],['交给目标解码器',R`z_{j\to m}=s_m\odot s_j\odot z_j`,'目标私有变量在跨视图路径中为零，不采样一个新私有后验。']],
 detach:[['保留同一前向数值',R`y=\operatorname{stopgrad}(z^{sh})=z^{sh}`,'截断不删除共享变量；decoder 仍能读取它的数值。'],['切断反向通路',R`\frac{\partial y}{\partial z^{sh}}=0`,'反向箭头在屏障处停止。只针对同视图重建的共享坐标。'],['保留其余梯度',R`\nabla_{\phi^{sh}}\mathcal L_{cross}\ne0,\quad\nabla_{\phi^{sh}}\mathrm{KL}\ne0`,'跨视图重建和 KL 仍可更新共享后验；私有分支和 decoder 也仍被同视图误差训练。']],
 'poe-precision':[['各坐标的有效专家',R`\tau_j=\frac{s_j}{\sigma_j^2+10^{-8}}`,'这里 s_j 是二值潜维掩码；缺失模态的精度贡献被置零。'],['连同单位先验求和',R`\tau=1+\sum_j\tau_j,\quad\mu=\frac{\sum_j\tau_j\mu_j}{\tau}`,'此仓库按逆方差融合，与 MMHVAE 的逆 scale 实现不同。'],['输出融合标准差',R`\sigma=\sqrt{1/\tau}`,'Normal 的第二参数仍为标准差；潜维无专家时回退标准正态先验。']],
 likelihood:[['均值与噪声',R`\mu_x=G(z),\quad s_x=\operatorname{softplus}(a_x)`,'Gaussian 观测噪声是每模态、每特征的可学习参数，初始化 a_x=-1。'],['逐特征对数密度',R`\log p(x_i|z)=-\frac12\left(\frac{x_i-\mu_i}{s_i}\right)^2-\log s_i-\frac12\log(2\pi)`,'示例显示实际对数似然值；不把密度当成预测置信度。'],['缺失掩码与归约',R`\mathcal L_{rec}=-\sum_m w_m\operatorname{mean}_b\sum_i\log p(x_{bmi}|z)`,'来源或目标缺失的样本对均屏蔽；先按特征求和，再按 batch 平均。']],
 kl:[['读取后验均值和尺度',R`q=\mathcal N(\mu,\operatorname{diag}(s^2)),\quad p=\mathcal N(0,I)`,'后验维度由潜结构掩码控制。'],['逐维解析 KL',R`\mathrm{KL}_i=\tfrac12(\mu_i^2+s_i^2-1-\log s_i^2)`,'每个水晶单元展示一个非负贡献。'],['掩码后归约',R`\mathcal L= -\log p(x|z)+\beta\sum_i\mathrm{KL}_i`,'缺失数据与非活跃潜维不贡献 KL；训练将 beta 从 10⁻⁵ 线性增加至 1。']],
 fft:[['复数图像网格',R`x_{h,w}\in\mathbb C`,'实部和虚部各占一个水平通道层。'],['与复指数基函数相乘',R`F_{u,v}=\frac1{\sqrt{HW}}\sum_{h,w}x_{h,w}e^{-2\pi i(uh/H+vw/W)}`,'同一频率格由全部空间元素贡献；弧线表达复指数相位，不是移动的实物。'],['组织二维频谱',R`k=\operatorname{fftshift}(\mathcal F(\operatorname{ifftshift}(x)))`,'示例使用中心化正交 DFT。原库调用外部 llmr.fft.fft2c；归一化约定需由该依赖确认。']],
 ifft:[['读取复数频率格',R`k_{u,v}\in\mathbb C`,'采样掩码已在上游处理；未采样值不会凭空补齐。'],['逆向基函数叠加',R`x_{h,w}=\frac1{\sqrt{HW}}\sum_{u,v}k_{u,v}e^{+2\pi i(uh/H+vw/W)}`,'逆变换将所有频率贡献叠加到一个空间位置。'],['得到每线圈图像',R`x_c=\mathcal F^{-1}(k_c)`,'后续仍须乘共轭灵敏度并求和；IFFT 本身不完成 SENSE 合并。']],
 'complex-multiply':[['成对读取实部和虚部',R`a=a_r+ia_i,\quad b=b_r+ib_i`,'两层水晶分别表示实数和虚数分量。'],['四次实数乘积',R`ab=(a_rb_r-a_ib_i)+i(a_rb_i+a_ib_r)`,'线圈灵敏度或相位因子逐位置相乘；不跨空间求和。'],['保留复数结构',R`y\in\mathbb C^{C\times H\times W}`,'伴随路径使用共轭灵敏度和相反相位，详见上层节点。']],
 gradient:[['固定网络与风格',R`f(c)=\|A G(c,\operatorname{sg}(s))-y\|_2^2`,'内容修正只令 content 可微，style.detach()；不训练网络权重。'],['链式法则回传',R`\nabla_cf=2J_G(c)^*A^*(AG(c,s)-y)`,'反向箭头穿过固定解码器和测量算子；示例用可核算的小型线性 G 演示链式法则。'],['沿负梯度更新',R`c^{k+1}=c^k-\eta\nabla_cf(c^k)`,'滑杆控制教学步长；真实演示 notebook 设置 cr_step_size=0.1。']],
 l1:[['对齐预测与目标',R`e_i=\hat x_i-x_i`,'图像、content、style 的 L1 目标具有各自的来源和监督范围。'],['逐元素绝对误差',R`a_i=|e_i|`,'水晶块的幅值由真实示例误差计算。'],['对所有元素平均',R`\mathcal L_1=\frac1N\sum_i a_i`,'对应 torch.nn.L1Loss(reduction="mean")。']],
 lsgan:[['读取多尺度 Patch 分数',R`d=\operatorname{cat}(\operatorname{flatten}(D_1),D_2,D_3)`,'多尺度输出展平后拼接，像素多的尺度在均值中贡献更多。'],['平方偏差',R`e_i=(d_i-t)^2`,'生成器与真实样本目标 t=1；伪样本判别器目标 t=0。'],['最小二乘对抗损失',R`\mathcal L_{GAN}=\frac1{2N}\sum_i e_i`,'代码包含 0.5 因子；不是交叉熵或 Wasserstein 目标。']],
 subtract:[['对齐两路数据',R`a_i,\;b_i`,'保持相同通道和空间坐标。'],['逐项相减',R`r_i=a_i-b_i`,'数据一致性中表示预测 k-space 与实际采样之差。'],['保留残差形状',R`r\in\operatorname{shape}(a)`,'残差继续进入伴随算子或损失归约。']],
 scale:[['标量与特征',R`\alpha,\;x_i`,'标量沿张量元素广播。'],['逐项缩放',R`y_i=\alpha x_i`,'数据一致性步长为 1/max_eig。'],['交给下一运算',R`y\in\operatorname{shape}(x)`,'缩放本身不做加法；更新减法在上一级明确列出。']],
 magnitude:[['复数双层表示',R`x=a+ib`,'实部与虚部是同一坐标的两项。'],['计算模长',R`|x|=\sqrt{a^2+b^2}`,'转换到 CoSMo 的第一步丢弃图像相位。'],['得到实数幅值',R`|x|\in\mathbb R_{\ge0}`,'相位图通过测量算子单独建模。']],
 rescale:[['读取幅值与标定范围',R`x\in[a,b]`,'范围由输入 recon_intensity_range 或 ref_intensity_range 指定。'],['映射并裁剪',R`v=\operatorname{clip}(2(x-a)/(b-a)-1,-1,1)`,'公开 MRI→CoSMo 链将幅值映射到 −1 至 1。'],['逆映射回 MRI 范围',R`x=(v+1)(b-a)/2+a`,'反向链先去 padding，再恢复量纲并转换成零虚部复数。']],
 sum:[['收集同一坐标的贡献',R`x_{c,h,w}`,'不同通道或支路按指定轴归约。'],['逐项累加',R`y_{h,w}=\sum_c x_{c,h,w}`,'SENSE 伴随路径在乘共轭灵敏度后沿线圈维求和。'],['保留空间结构',R`[B,C,H,W]\to[B,1,H,W]`,'总和不等于均值；此算子不除以通道数。']],
}
export function operatorSpec(op:Op,detail='',formula=''):MathSpec{
 const standard:Partial<Record<Op,Glyph>>={tensor:'tensor',linear:'linear',conv:'conv',pool:'pool',reshape:'tensor',split:'split',gaussian:'gaussian',sample:'sample',pad:'tensor',unpad:'tensor',bilinear:'up',broadcast:'tensor',tanh:'activation',add:'sum',multiply:'multiply',norm:'norm'}
 if(standard[op]){
  const title=op==='reshape'?'flatten':op==='pad'?'ReflectionPad2d':op==='unpad'?'切片去填充':op==='tanh'?'Tanh':op==='broadcast'?'broadcast':op
  const spec=mathSpec({id:op,title,glyph:standard[op]!,shape:'',detail,sourceName:op},formula)
  if(op==='unpad'){spec.kind='tensor';spec.operation='tensor';spec.steps[1]={title:'保留原始有效区间',formula:R`Y=X[:,:,p_h:p_h+H,p_w:p_w+W]`,explanation:'外部工具 unpad 依据原始形状裁剪；不假定 padding 的具体填充值。'}}
  return spec
 }
 const steps=op==='spatial-mask'?[
 ['读取实际采样网格',R`M_{h,w}\in\{0,1\}`,'二值网格表示 k-space 的测量位置。'],
 ['按位置筛选复数数据',R`k'_{c,h,w}=M_{h,w}k_{c,h,w}`,'同一位置的实部和虚部使用同一个掩码。'],
 ['保留测量支撑集',R`M^2=M`,'未采样位置置零；该步骤不估计缺失频率。']
 ]:recipes[op]??recipes.subtract!
 return {kind:'tensor',operation:`lab:${op}`,steps:steps.map(([title,formula,explanation])=>({title,formula,explanation})),control:op==='gradient'?'教学梯度步长':op==='poe-precision'?'专家标准差':op==='adain'?'风格调制强度':'观察元素位置',legend:'水平网格表示空间或特征坐标，纵向层片区分通道；数据依赖来自当前源码运算。',formula:formula||steps[1][1],source:op,kernel:3,stride:1,padding:1,weightNorm:false,bias:true,reflect:false}
}
export interface NumericExample {a:number[];b:number[];out:number[];index:number;readout:string}
export function scientificExample(spec:MathStage,probe:number):NumericExample{
 const op=spec.operation.replace('lab:','').split(':')[0],a=demoValues.slice(0,16),b=demoValues.slice(16,32),index=Math.min(15,Math.floor(probe*16));let out=[...a]
 if(op==='leaky01'||op==='leaky02')out=a.map(x=>x<0?x*(op==='leaky01'?.01:.2):x)
 else if(op==='softplus')out=a.map(x=>Math.log1p(Math.exp(x)))
 else if(op==='norm-affine'||op==='adain'){const normalized=normalizeValues(a),gamma=op==='adain'?probe:1.2,beta=op==='adain'?probe-.5:.2;out=spec.stage===0?a:spec.stage===1?normalized:normalized.map(x=>op==='adain'?beta+(1+gamma)*x:beta+gamma*x);b.fill(gamma)}
 else if(op==='latent-mask'){const bits=spec.operation.split(':')[2]??'111100';a.splice(6);b.splice(6);for(let i=0;i<6;i++)b[i]=Number(bits[i]);out=a.map((x,i)=>x*b[i])}
 else if(op==='spatial-mask'){for(let i=0;i<16;i++)b[i]=i%4===1||i%4===2?1:0;out=a.map((x,i)=>x*b[i])}
 else if(op==='poe-precision'){const s=.45+probe*1.35,tau=1+1/(s*s+1e-8)+1/(.7**2+1e-8),mu=(-.8/(s*s+1e-8)+1.2/(.7**2+1e-8))/tau;out=[mu,Math.sqrt(1/tau)];return {a:[0,-.8,1.2],b:[1,s,.7],out,index:0,readout:`μ = ${mu.toFixed(3)} · σ = ${out[1].toFixed(3)} · τ = ${tau.toFixed(3)}`}}
 else if(op==='likelihood')out=a.map((x,i)=>-.5*((x-b[i])/.5)**2-Math.log(.5)-.5*Math.log(2*Math.PI))
 else if(op==='kl')out=a.map((mu,i)=>.5*(mu**2+(.6+Math.abs(b[i]))**2-1-Math.log((.6+Math.abs(b[i]))**2)))
 else if(op==='l1')out=a.map((x,i)=>Math.abs(x-b[i]))
 else if(op==='lsgan')out=a.map(x=>.5*(x-1)**2)
 else if(op==='gradient'){const eta=.2*probe;out=a.map((x,i)=>x-eta*2*(2*x-b[i])*2);return {a,b,out,index,readout:`η = ${eta.toFixed(3)} · G(c)=2c · ∂f/∂c = ${(4*(2*a[index]-b[index])).toFixed(3)} · c′ = ${out[index].toFixed(3)}`}}
 else if(op==='magnitude')out=a.map((x,i)=>Math.hypot(x,b[i]))
 else if(op==='complex-multiply'){const real=a.map((x,i)=>x*b[i]-.3*.2),imag=a.map((x,i)=>x*.2+.3*b[i]);return {a:[...a,...a.map(()=>.3)],b:[...b,...b.map(()=>.2)],out:[...real,...imag],index,readout:`(${a[index].toFixed(2)}+0.30i) × (${b[index].toFixed(2)}+0.20i) = ${real[index].toFixed(3)} ${imag[index]<0?'−':'+'} ${Math.abs(imag[index]).toFixed(3)}i`}}
 else if(op==='rescale')out=a.map(x=>Math.max(-1,Math.min(1,x*2-1)))
 else if(op==='subtract')out=a.map((x,i)=>x-b[i])
 else if(op==='scale')out=a.map(x=>x*.5)
 else if(op==='sum')out=Array.from({length:8},(_,i)=>a[i]+a[i+8])
 else if(op==='spectral'){
  const W=Array.from({length:16},(_,i)=>Math.sin(i*.73)*.5);let u=[.5,.5,.5,.5],v=u
  for(let k=0;k<12;k++){v=Array.from({length:4},(_,j)=>u.reduce((s,x,i)=>s+x*W[i*4+j],0));v=v.map(x=>x/Math.hypot(...v));u=Array.from({length:4},(_,i)=>v.reduce((s,x,j)=>s+x*W[i*4+j],0));u=u.map(x=>x/Math.hypot(...u))}
  const sigma=u.reduce((s,x,i)=>s+x*v.reduce((t,y,j)=>t+y*W[i*4+j],0),0);return {a:W,b:[...u,...v],out:W.map(x=>x/sigma),index,readout:`σ̂ = ${sigma.toFixed(4)} · W̄[${index}] = ${(W[index]/sigma).toFixed(4)}`}
 }else if(op==='fft'||op==='ifft'){
  const input=a.map((re,i)=>({re,im:b[i]*.2})),output=complexDFT2(input,op==='ifft')
  return {a:[...input.map(x=>x.re),...input.map(x=>x.im)],b:[],out:[...output.map(x=>x.re),...output.map(x=>x.im)],index,readout:`[${Math.floor(index/4)},${index%4}] = ${output[index].re.toFixed(3)} ${output[index].im<0?'−':'+'} ${Math.abs(output[index].im).toFixed(3)}i · 4×4 中心化正交示例`}
 }
 const scalar=['l1','lsgan'].includes(op)&&spec.stage===2
 const j=Math.min(index,a.length-1)
 return {a,b,out:scalar?[mean(out)]:out,index:scalar?0:Math.min(j,out.length-1),readout:`i = ${j} · x = ${a[j].toFixed(3)} · ${scalar?'均值':'y'} = ${(scalar?mean(out):out[Math.min(j,out.length-1)]).toFixed(3)}${op==='detach'?' · 前向相同，反向导数 = 0':''}`}
}
export function complexDFT2(input:{re:number;im:number}[],inverse=false){
 const n=Math.sqrt(input.length),sign=inverse?1:-1
 return input.map((_,index)=>{const u=Math.floor(index/n)-n/2,v=index%n-n/2;let re=0,im=0;input.forEach((x,i)=>{const h=Math.floor(i/n)-n/2,w=i%n-n/2,angle=sign*2*Math.PI*(u*h+v*w)/n;re+=x.re*Math.cos(angle)-x.im*Math.sin(angle);im+=x.re*Math.sin(angle)+x.im*Math.cos(angle)});return {re:re/n,im:im/n}})
}
