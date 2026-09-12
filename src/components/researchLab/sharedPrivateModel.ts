import {ModelBuilder,type PaperModel} from './catalog'
import {operatorSpec,type Op} from './operators'
import type {Glyph} from '../mmhvae/anatomy'
export type VAEVariant='MMVAE++'|'MMVAE'
const R=String.raw
export function sharedPrivateModel(variant:VAEVariant='MMVAE++'):PaperModel{
 const b=new ModelBuilder(),core='experiments/core.py',code='multimodalVAE/MMVAE.py'
 const compound=(id:string,parent:string|undefined,title:string,shape:string,purpose:string,implementation:string,formula:string,file=code,line=138)=>b.add(id,parent,title,'network',shape,purpose,implementation,formula,file,line)
 const atom=(id:string,parent:string,title:string,op:Op,shape:string,detail:string,formula:string,file=code,line=138,glyph:Glyph='tensor')=>{const spec=operatorSpec(op,detail,formula);return b.add(id,parent,title,glyph,shape,spec.steps[0].explanation,detail,formula||spec.steps[1].formula,file,line,id,spec)}
 compound('root',undefined,`${variant} · 共享与私有潜空间`,'B × [D₁,D₂] → B × [2+2+2]','用跨视图预测识别共享变化，同时保留模态私有信息。','固定于公开实验配置：两个连续模态、6 个潜维、128 隐藏单元、Gaussian 似然。网络为浅层 MLP；这里不引入卷积塔或 MMHVAE 的七层潜变量。',R`z=[z^{p1},z^{sh},z^{p2}],\quad s_1=[1,1,1,1,0,0],\quad s_2=[0,0,1,1,1,1]`,core,22)
 for(const m of [1,2]){
  const x=m===1?-14:14,color=m===1?'#83cce8':'#c0a8e8'
  atom(`input${m}`,'root',`模态 ${m} · 特征向量`,'tensor',`B × D${m}`,'连续组学特征；D 随实验数据而变，不能固定为 MRI 图像尺寸。',R`x_m\in\mathbb R^{B\times D_m}`,core,29).position=[x,48,0]
  const enc=compound(`encoder${m}`,'root',`模态 ${m} · 编码塔`,`B × D${m} → B × 12`,'独立读取各模态，预测完整六维潜空间的均值和尺度参数。','Linear(D,128) → LeakyReLU(0.01) → Linear(128,12)。私有/共享结构来自后续固定掩码，不是三套独立编码网络。',R`h_m=W_{m,2}\operatorname{LReLU}_{.01}(W_{m,1}x_m+b_{m,1})+b_{m,2}`,core,31);enc.overview=true
  atom(`encoder${m}/linear1`,enc.id,'输入特征投影','linear',`B × D${m} → B × 128`,'nn.Linear(data_dims[m],128)，含偏置。',R`h=W_1x+b_1`,core,m===1?33:38,'linear')
  atom(`encoder${m}/activation`,enc.id,'保留弱负响应','leaky01','B × 128','nn.LeakyReLU()，默认负斜率 0.01。',R`f(x)=\max(x,0)+.01\min(x,0)`,core,m===1?34:39,'activation')
  atom(`encoder${m}/head`,enc.id,'均值与尺度联合投影','linear','B × 128 → B × 12','nn.Linear(128,2*z_dim)，输出先均值、后原始尺度参数。',R`h=[\mu,a]\in\mathbb R^{B\times12}`,core,m===1?35:40,'linear');b.chain(enc.id)
  enc.children.forEach((id,i)=>{b.entries[id].position=[x,41-i*5,0];b.entries[id].color=color})
  const q=compound(`posterior${m}`,'root',`模态 ${m} · 六维 Gaussian`,'B × 6','为每个潜坐标构造可重参数化的对角高斯。','torch.split(h,z_dim,dim=1)，然后 Normal(mu,F.softplus(log_sigma))。变量名不是指数参数化的证据。',R`q_m(z|x_m)=\mathcal N(\mu_m,\operatorname{diag}(\operatorname{softplus}(a_m)^2))`,code,59);q.position=[x,23,0];q.glyph='gaussian'
  atom(`${q.id}/split`,q.id,'拆分均值与原始尺度','split','B × 12 → 2 × (B × 6)','torch.split(encoder_output,z_dim,dim=1)。',R`[\mu,a]=\operatorname{split}(h,6)`,code,60,'split')
  atom(`${q.id}/softplus`,q.id,'保证标准差为正','softplus','B × 6','尺度使用 softplus，而非 exp 或平方根。',R`s=\log(1+e^a)`,code,61,'activation')
  atom(`${q.id}/normal`,q.id,'建立对角后验','gaussian','B × 6','Normal(mu,softplus(a))；六个坐标条件独立。',q.formula,code,61,'gaussian');b.chain(q.id)
  const sample=compound(`sample${m}`,'root',`模态 ${m} · 重参数化`,'B × 6','将采样随机性与可微参数分离。','qz_list[j].rsample()，随后乘来源模态的潜维掩码。',R`z_j=\mu_j+s_j\odot\epsilon,\quad\epsilon\sim\mathcal N(0,I)`,code,154);sample.glyph='sample';sample.position=[x,14,0]
  atom(`${sample.id}/noise`,sample.id,'独立标准正态噪声','sample','B × 6','示意正态噪声，Box–Muller 只作数学解释，不假定 PyTorch 内核实现。',R`\epsilon\sim\mathcal N(0,I)`,code,154,'sample')
  atom(`${sample.id}/scale`,sample.id,'按后验标准差缩放','multiply','B × 6','逐坐标乘标准差。',R`\eta=s_j\odot\epsilon`,code,154,'multiply')
  atom(`${sample.id}/mean`,sample.id,'平移到后验中心','add','B × 6','逐位置加均值，不是沿潜维求和。',R`z_j=\mu_j+\eta`,code,154,'sum');b.chain(sample.id)
  const dec=compound(`decoder${m}`,'root',`模态 ${m} · 生成塔`,`B × 6 → B × D${m}`,'共享本模态解码权重，接收来自任一编码器的有效潜维。','Linear(6,128) → LeakyReLU(0.01) → Linear(128,D)。同视图与跨视图调用同一 decoder。',R`\hat x_m=G_m(s_m\odot z)`,core,44);dec.overview=true
  atom(`${dec.id}/linear1`,dec.id,'潜变量投影','linear','B × 6 → B × 128','含偏置的全连接投影。',R`h=Wz+b`,core,m===1?46:51,'linear')
  atom(`${dec.id}/activation`,dec.id,'解码非线性','leaky01','B × 128','负斜率为 0.01。',R`h'=\operatorname{LReLU}_{.01}(h)`,core,m===1?47:52,'activation')
  atom(`${dec.id}/mean`,dec.id,'逐特征均值输出','linear',`B × 128 → B × D${m}`,'最后一层没有 sigmoid；输出 Gaussian 观测均值。',R`\mu_{x_m}=W_2h'+b_2`,core,m===1?48:53,'linear');b.chain(dec.id)
  dec.children.forEach((id,i)=>{b.entries[id].position=[x,-16-i*5,0];b.entries[id].color=color})
  atom(`likelihood${m}`,'root',`模态 ${m} · 观测似然`,'likelihood',`B × D${m}`,'每特征 noise_sd 初始化为 −1，softplus 后作为标准差；输入或目标缺失时屏蔽 log_prob。',R`p(x_m|z)=\mathcal N(G_m(z),\operatorname{diag}(\operatorname{softplus}(a_m)^2))`,code,47,'gaussian').position=[x,-34,0]
  b.link('root',`input${m}`,enc.id);b.link('root',enc.id,q.id);b.link('root',q.id,sample.id);b.link('root',dec.id,`likelihood${m}`)
 }
 const routes=compound('routes','root','四条编码—解码路径','2 × 2 调用对','区分共享表示的训练来源，而不改动前向共享值。',`${variant} 的前向和损失路径。MMVAE++ 仅在 j=m 的重建项中 detach 共享坐标；KL 未被截断。`,R`\mathcal L_{rec}=-\sum_{j=1}^2\sum_{m=1}^2w_m\mathbb E_{q_j}\log p_m(x_m|s_m\odot s_j\odot z_j)`,code,138)
 routes.overview=true
 for(const j of [1,2])for(const m of [1,2]){
  const id=`route${j}${m}`,self=j===m,e=compound(id,routes.id,`${j} → ${m} · ${self?'同视图重建':'跨视图预测'}`,'B × 6',self?'私有与共享共同生成本模态。':'两路潜维掩码求交，仅允许共享因素解释目标。',self&&variant==='MMVAE++'?'共享值仍输入 decoder；只阻断同视图重建误差对这些值的反向梯度。':'此路径正常传递前向数据与反向梯度。',self&&variant==='MMVAE++'?R`z'=[z^{pr},\operatorname{sg}(z^{sh})]`:R`z'=s_m\odot s_j\odot z_j`,code,self?174:166)
  e.position=[(j===1?-1:1)*(self?10:3),3,self?-1:6];e.glyph=self&&variant==='MMVAE++'?'split':'concat'
  atom(`${id}/source-mask`,id,'保留来源潜维','latent-mask','B × 6','根据来源模态固定掩码置零非活跃坐标。',R`z=s_j\odot z_j`,code,157,'multiply').math!.operation=`lab:latent-mask:${j===1?'111100':'001111'}`
  if(self&&variant==='MMVAE++')atom(`${id}/stop`,id,'共享值保留 · 反向截断','detach','B × 2 shared','只 detach which_dims_shared；私有坐标原样复制。',R`\tilde z^{sh}=\operatorname{sg}(z^{sh})`,code,178,'split')
  atom(`${id}/target-mask`,id,'选择目标解码维度','latent-mask','B × 6','再次乘目标模态的潜维掩码。跨视图时只留下共享的两维。',R`z_{active}=s_m\odot z`,code,180,'multiply').math!.operation=`lab:latent-mask:${self?(m===1?'111100':'001111'):'001100'}`;b.chain(id)
  b.link('root',`sample${j}`,id,self?'私有 + 共享':'共享');b.link('root',id,`decoder${m}`,'共享 decoder 权重')
 }
 const loss=compound('objective','root','重建目标与 KL','scalar','平衡可解释的生成和受约束的潜空间。','先求特征和，再求 batch 均值；modality_weights 对不同模态加权。β 从 10⁻⁵ 增至 1。',R`\mathcal L=-\log p(x|z)+\beta\mathrm{KL}`,code,192);loss.position=[0,-45,0];loss.glyph='sum'
 for(const m of [1,2]){atom(`objective/kl${m}`,'objective',`模态 ${m} · 有效潜维 KL`,'kl','B × 6 → scalar','KL × 来源潜维掩码，再屏蔽来源缺失样本，按维求和并对 batch 平均。',R`\mathrm{KL}(q_j\|\mathcal N(0,I))`,code,162,'gaussian');b.link('root',`likelihood${m}`,'objective');b.link('root',`posterior${m}`,'objective','KL',true)}
 atom('objective/sum','objective','合并所有有效损失','sum','scalar','重建项和 beta 加权 KL 的标量和。',loss.formula,code,192,'sum');b.link('objective','objective/kl1','objective/sum');b.link('objective','objective/kl2','objective/sum')
 const inference=compound('inference','root','PoE 与目标族比较','B × 6','显式区分四个模型目标。','MVAE 与 MoPoE-VAE 使用 ProductOfExperts；MMVAE / MMVAE++ 逐单模态采样再解码所有模态，没有一个中央平均 Gaussian 采样节点。',R`q_{PoE}\propto p(z)\prod_{j\in I}q_j(z|x_j)`, 'multimodalVAE/PoE.py',18)
 for(const [key,title] of [['joint','联合观测 PoE'],['one','仅模态 1 PoE'],['two','仅模态 2 PoE']])atom(`inference/${key}`,inference.id,title,'poe-precision','B × 6','含单位先验，按照逆方差精度相加；应用缺失和潜维掩码。',R`\tau=1+\sum_j(\sigma_j^2+10^{-8})^{-1}s_j`, 'multimodalVAE/PoE.py',18,'poe').math!.operation=`lab:poe-precision:${key}`
 const overview:string[]=[]
 for(const id of b.entries.root.children){if(['routes','inference'].includes(id))continue;const e=b.entries[id];if(e.overview)overview.push(...e.children);else overview.push(id)}
 overview.push(...routes.children)
 const edgeEnd=(id:string,from:boolean)=>{const e=b.entries[id];return e.overview&&e.children.length?(from?e.children.at(-1)!:e.children[0]):id}
 const overviewEdges=b.entries.root.edges.map(e=>({...e,from:edgeEnd(e.from,true),to:edgeEnd(e.to,false)}))
 for(const e of Object.values(b.entries))if(e.overview)overviewEdges.push(...e.edges)

 return {id:'mmvae',name:variant,repo:'https://github.com/kasparmartens/shared-private-multimodalVAE',commit:'49d46762b82de89421b70159e2164244d21c8cb6',entries:b.entries,overview,overviewEdges,intro:'从完整潜向量到固定稀疏掩码，再追踪四条前向路径和共享坐标的反向梯度。',config:'experiments/core.py · 隐藏宽度 128 · z=2+2+2 · Gaussian 观测 · supervised=False',equations:[{title:'共享值与梯度路径',formula:R`z_{j\to m}=s_m\odot s_j\odot z_j,\quad j=m:\ z^{sh}\mapsto\operatorname{sg}(z^{sh})`,explanation:'MMVAE++ 只阻断同视图重建项的共享坐标梯度。跨视图重建、KL、私有坐标和 decoder 训练仍保留。'},{title:'PoE 的实际参数化',formula:R`\tau=1+\sum_j\frac{s_j}{\sigma_j^2+10^{-8}},\quad\mu=\frac{\sum_j\tau_j\mu_j}{\tau},\quad\sigma=\sqrt{1/\tau}`,explanation:'PoE 属于 MVAE / MoPoE-VAE；MMVAE 家族则遍历单模态后验。此处 s_j 是潜维掩码，不是 Gaussian 标准差。'},{title:'训练目标与缺失数据',formula:R`\mathcal L=-\sum_{j,m}w_m\,\operatorname{mean}_b\sum_i\log p_m(x_{bi}|z_j)+\beta\sum_j\mathrm{KL}_j`,explanation:'代码对输入缺失与目标缺失取逻辑或，屏蔽该样本对的重建；默认实验缺失率为零。'}],audit:[{title:'不是三套编码头',detail:'每模态一条 D→128→12 的 MLP，随后 split 均值/尺度。共享与私有来自固定潜维掩码。',file:core,line:31},{title:'尺度变量名与实际函数',detail:'log_sigma_z 实际经过 softplus；观测噪声同样如此，初始化参数 −1。',file:code,line:59},{title:'MoPoE 的代码组织',detail:'公开实现沿用 MVAE 类，通过 include_cross_view_terms 增加跨视图项；不是另一个显式混合采样器。',file:core,line:63},{title:'评估辅助函数存在掩码疑点',detail:'MVAE.get_cross_view_preds 中 fill_ 使用私有掩码的补集，与训练 forward_and_loss 的跨视图掩码不同。本展示将训练路径和该评估辅助函数明确区分，不据此生成实验曲线。',file:'multimodalVAE/MVAE.py',line:230},{title:'可选监督头',detail:'core.py 构造额外 Linear(z_dim,1) 与 Bernoulli 似然，但默认 supervised=False，不属于当前双模态无监督计算路径。',file:core,line:55}]}
}
