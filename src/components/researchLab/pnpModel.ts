import {ModelBuilder,type PaperModel,type Entry} from './catalog'
import {operatorSpec,type Op} from './operators'
import type {Glyph} from '../mmhvae/anatomy'
const R=String.raw,ae='pnp_cosmo/cosmo/autoencoders.py',layers='pnp_cosmo/cosmo/layers.py',recon='pnp_cosmo/recon/algorithms.py',system='pnp_cosmo/cosmo/cosmo_systems.py'
export function pnpModel():PaperModel{
 const b=new ModelBuilder()
 const folder=(id:string,parent:string|undefined,title:string,shape:string,purpose:string,detail:string,formula:string,file=ae,line=9)=>b.add(id,parent,title,'network',shape,purpose,detail,formula,file,line)
 const atom=(id:string,parent:string,title:string,op:Op,shape:string,detail:string,formula:string,file=layers,line=40,glyph:Glyph='tensor')=>{const spec=operatorSpec(op,detail,formula);return b.add(id,parent,title,glyph,shape,spec.steps[0].explanation,detail,formula||spec.steps[1].formula,file,line,id,spec)}
 folder('root',undefined,'PnP-CoSMo · 内容引导重建','双域影像 + 复数 k-space','用参考影像的内容引导目标对比重建，同时保留测量数据一致性。','NYU 配置：content 2 通道、style 2 维、基宽 32、4 个残差块、content 不下采样、style 下采样 4 次、style MLP 2 层。两域参数独立。',R`c^0=E^c_r(x_r),\quad x^{k+1}=\mathrm{DC}(G_t(c^k,E^s_t(x^k)))`,recon,47)
 const net=folder('networks','root','双域内容与风格网络','两个独立 AutoEncoder','保留对比特异风格，并把跨对比解剖内容分离为保分辨率的空间特征。','公开 NYU 配置使用 DeterministicContentEncoder；style 网络确定性输出 2×1×1。StochasticContentMUNIT 是另一个类，不是当前默认路径。',R`c_m=E^c_m(x_m),\quad s_m=E^s_m(x_m),\quad\hat x_m=G_m(c,s_m)`);net.overview=true
 function linearBlock(id:string,parent:string,title:string,ci:number,co:number){
  const block=folder(id,parent,title,`${ci} → ${co}`,'将全局风格变换为条件参数。','Linear → Identity → LeakyReLU(0.2)。最后一层同样带非线性。',R`y=\operatorname{LReLU}_{.2}(Wx+b)`,layers,6)
  atom(`${id}/linear`,id,'含偏置线性映射','linear',`${ci} → ${co}`,'nn.Linear(...,bias=True)。',R`h=Wx+b`,layers,17,'linear')
  atom(`${id}/activation`,id,'LeakyReLU · 负斜率 0.2','leaky02',`${co}`,'Identity 不改变数据；随后应用 LeakyReLU(0.2)。',R`y=\max(h,0)+.2\min(h,0)`,layers,31,'activation');b.chain(id);return block
 }
 function adain(id:string,parent:string,c:number){
  const block=folder(id,parent,'自适应实例归一化',`${c} × H × W`,'把风格向量转成每通道的缩放与偏移。','InstanceNorm affine=False；fc 是 256→2C 的 LinearBlock，含 LeakyReLU；unsqueeze、chunk 后执行 beta+(1+gamma)*x。',R`y=\beta(s)+(1+\gamma(s))\odot\operatorname{IN}(x)`,layers,130);block.glyph='norm'
  linearBlock(`${id}/fc`,id,'风格到仿射参数',256,2*c)
  atom(`${id}/broadcast`,id,'补充空间广播维','broadcast',`B × ${2*c} → B × ${2*c} × 1 × 1`,'按 x.dim()-y.dim() 次数在末尾 unsqueeze。',R`y\mapsto y[:,:,None,None]`,layers,145)
  atom(`${id}/split`,id,'拆分缩放与偏移','split',`2 × (B × ${c} × 1 × 1)`,'gamma,beta=y.chunk(2,1)。',R`(\gamma,\beta)=\operatorname{chunk}(y,2)`,layers,146,'split')
  atom(`${id}/normalize`,id,'标准化内容响应','norm',`${c} × H × W`,'InstanceNorm2d(affine=False)；仅沿空间维统计。',R`\hat x=(x-\mu)/\sqrt{v+10^{-5}}`,layers,147,'norm')
  atom(`${id}/affine`,id,'广播后融合风格','adain',`${c} × H × W`,'torch.addcmul(beta,x,1+gamma)。',block.formula,layers,148,'multiply')
  b.link(id,`${id}/fc`,`${id}/broadcast`,'风格参数');b.link(id,`${id}/broadcast`,`${id}/split`);b.link(id,`${id}/split`,`${id}/affine`,'gamma, beta',true);b.link(id,`${id}/normalize`,`${id}/affine`,'标准内容');b.entries[`${id}/normalize`].position=[0,0,0];b.entries[`${id}/affine`].position=[0,-9,0];b.entries[`${id}/fc`].position=[-5,0,2];b.entries[`${id}/broadcast`].position=[-5,-3,2];b.entries[`${id}/split`].position=[-5,-6,2];return block
 }
 function convBlock(id:string,parent:string,title:string,ci:number,co:number,k:number,stride=1,norm:'instance'|'adain'|'none'='none',activation:'leaky02'|'tanh'='leaky02',spectral=true){
  const pad=Math.floor((k-1)/2),block=folder(id,parent,title,`${ci} → ${co} · ${k}×${k} / s${stride}`,'执行局部特征提取，并按公开配置进行归一化和非线性。',`Conv2d(${ci},${co},${k},stride=${stride},padding=${pad},padding_mode=reflect,bias=True) → ${norm==='none'?'Identity':norm==='adain'?'AdaIN':'InstanceNorm2d affine=True'} → ${activation==='tanh'?'Tanh':'LeakyReLU(0.2)'}`,'',layers,40);block.glyph='conv'
  if(spectral)atom(`${id}/spectral`,id,'卷积权重谱归一化','spectral',`${co} × (${ci}·${k}²)`,'spectral_norm 挂在 Conv2d 上，先归一化权重再卷积，不修改输入图像尺寸。',R`\bar W=W/\hat\sigma(W)`,layers,54,'linear')
  atom(`${id}/conv`,id,`${k}×${k} 局部互相关`,'conv',`${ci} → ${co}`,`kernel_size=${k} stride=${stride} padding=${pad} reflect bias=True`,'',layers,52,'conv')
  if(norm==='adain')adain(`${id}/adain`,id,co)
  else if(norm==='instance')atom(`${id}/norm`,id,'按通道归一化并仿射','norm-affine',`${co} × H × W`,'InstanceNorm2d(out_channels,affine=True)。','',layers,59,'norm')
  atom(`${id}/activation`,id,activation==='tanh'?'限定影像动态范围':'保留负半轴响应',activation,`${co} × H × W`,activation==='tanh'?'输出非线性 Tanh。':'LeakyReLU(0.2)。','',layers,67,'activation');b.chain(id);if(spectral){b.entries[id].edges=b.entries[id].edges.map(edge=>edge.from===`${id}/spectral`?{...edge,residual:true,label:'归一化权重'}:edge)};block.formula=R`y=f(\operatorname{Norm}(\bar W\star x+b))`;return block
 }
 function residual(id:string,parent:string,title:string,c:number,norm:'instance'|'adain'){
  const block=folder(id,parent,title,`${c} × H × W`,'保留恒等路径，同时学习内容或风格条件下的局部修正。','两次 Conv2dBlock 后直接与 x 相加。两个卷积块各自带归一化与 LeakyReLU；相加后没有额外激活。',R`y=x+f_1(f_0(x))`,layers,87)
  atom(`${id}/input`,id,'原始输入与恒等旁路','tensor',`${c} × H × W`,'同一 x 同时输入主分支和残差相加端。',R`x`,layers,124)
  convBlock(`${id}/conv0`,id,'第一局部变换',c,c,3,1,norm);convBlock(`${id}/conv1`,id,'第二局部变换',c,c,3,1,norm)
  atom(`${id}/add`,id,'残差与恒等路汇合','add',`${c} × H × W`,'逐元素 x+dx，不跨通道归约。',block.formula,layers,126,'sum');b.chain(id);b.link(id,`${id}/input`,`${id}/add`,'原输入旁路',true);return block
 }
 for(const m of [1,2]){
  const domain=folder(`domain${m}`,'networks',`${m===1?'T1W':'T2W'} · 独立生成域`,'1 × H × W ↔ content + style','每域有独立内容编码器、风格编码器、风格 MLP 与解码器。','参数配置相同但权重不共享。默认路径按 NYU YAML 构造。',R`\mathcal M_m=\{E^c_m,E^s_m,F_m,G_m\}`);domain.overview=true
  atom(`image${m}`,domain.id,`${m===1?'T1W':'T2W'} · 输入影像`,'tensor','B × 1 × H × W','图像域训练数据，映射到 −1 至 1；不是输入 k-space。',R`x_m\in[-1,1]^{B\times1\times H\times W}`,system,193,'image')
  const content=folder(`content${m}`,domain.id,'内容编码塔','B × 1 × H × W → B × 2 × H × W','把解剖内容编码为空间结构；默认保留完整分辨率。','7×7 输入卷积 1→32；4 个 32 通道残差块；1×1 投影 32→2。num_downsamples_content=0。',R`c_m=E^c_m(x_m)`,ae,239);content.overview=true
  convBlock(`${content.id}/stem`,content.id,'输入空间特征',1,32,7,1,'instance')
  for(let r=0;r<4;r++)residual(`${content.id}/res${r}`,content.id,`内容残差块 ${r+1}`,32,'instance')
  convBlock(`${content.id}/projection`,content.id,'两通道内容投影',32,2,1,1,'none');b.chain(content.id)
  const style=folder(`style${m}`,domain.id,'风格编码塔','B × 1 × H × W → B × 2 × 1 × 1','压缩空间因素，保留全局对比风格。','7×7:1→32；4 次 4×4/s2:32→64→128→128→128；全局平均池化；裸 1×1 Conv:128→2。末尾不是 Linear。',R`s_m=E^s_m(x_m)`,ae,338);style.overview=true
  convBlock(`${style.id}/stem`,style.id,'初始风格特征',1,32,7)
  for(let d=0;d<4;d++)convBlock(`${style.id}/down${d}`,style.id,`风格下采样 ${d+1}`,d===0?32:d===1?64:128,d===0?64:128,4,2)
  atom(`${style.id}/pool`,style.id,'全局平均池化','pool','B × 128 × H/16 × W/16 → B × 128 × 1 × 1','AdaptiveAvgPool2d(1)。',R`s_c=\frac1{HW}\sum_{h,w}x_{chw}`,ae,368,'pool')
  atom(`${style.id}/head`,style.id,'两维风格输出','conv','B × 128 × 1 × 1 → B × 2 × 1 × 1','裸 nn.Conv2d(128,2,1,1,0)，无 spectral_norm、无 norm、无激活。kernel_size=1 stride=1 padding=0','',ae,369,'conv');b.chain(style.id)
  const mlp=folder(`mlp${m}`,domain.id,'风格条件 MLP','B × 2 → B × 256','为所有 AdaIN 层提供统一的条件向量。','x.view(B,-1)，2→256→256；两层 LinearBlock 均带 LeakyReLU(0.2)，不使用 spectral_norm。',R`s'=F_m(s_m)`,ae,223)
  atom(`${mlp.id}/flatten`,mlp.id,'展平全局风格','reshape','B × 2 × 1 × 1 → B × 2','view(x.size(0),-1)。',R`s\mapsto\operatorname{view}(s,B,-1)`,ae,236)
  linearBlock(`${mlp.id}/fc0`,mlp.id,'条件映射 1',2,256);linearBlock(`${mlp.id}/fc1`,mlp.id,'条件映射 2',256,256);b.chain(mlp.id)
  const decoder=folder(`decoder${m}`,domain.id,'风格条件生成塔','B × 2 × H × W → B × 1 × H × W','以内容为空间主干，以风格为外围条件生成目标对比。','AdaIN 1×1:2→32；4 个 AdaIN 残差块；7×7:32→1 + Tanh。默认无上采样。最后 7×7 Conv2dBlock 不传 spectral_norm。',R`\hat x_m=G_m(c,s')`,ae,175);decoder.overview=true
  convBlock(`${decoder.id}/lift`,decoder.id,'内容提升到生成通道',2,32,1,1,'adain')
  for(let r=0;r<4;r++)residual(`${decoder.id}/res${r}`,decoder.id,`风格生成残差块 ${r+1}`,32,'adain')
  convBlock(`${decoder.id}/output`,decoder.id,'最终影像映射',32,1,7,1,'none','tanh',false);b.chain(decoder.id)
  b.link(domain.id,`image${m}`,content.id);b.link(domain.id,`image${m}`,style.id);b.link(domain.id,style.id,mlp.id);b.link(domain.id,content.id,decoder.id,'空间内容');b.link(domain.id,mlp.id,decoder.id,'风格 → 全部 AdaIN')
  const x=m===1?-22:22,color=m===1?'#83cce8':'#c4b0ed'
  b.entries[`image${m}`].position=[x,51,0]
  content.children.forEach((id,i)=>{b.entries[id].position=[x-6,43-i*5,0];b.entries[id].color=color})
  style.children.forEach((id,i)=>{b.entries[id].position=[x+6,43-i*4.2,5];b.entries[id].color=color})
  mlp.position=[x+6,8,5]
  decoder.children.forEach((id,i)=>{b.entries[id].position=[x,-2-i*5,0];b.entries[id].color=color})
 }
 const loop=folder('reconstruction','root','在线重建 · CC / DC / CR','复数多线圈 k-space → 目标影像','交替执行网络内容约束、采样一致性和内容梯度修正。','Demo_NYU：200 次迭代，cr_enable=True，cr_step_size=0.1，max_eig=1。网络权重固定；每轮重新估计 style。',R`\tilde x^k=G_t(c^k,E^s_t(x^k)),\quad x^{k+1}=\tilde x^k-\alpha A^*(A\tilde x^k-y)`,recon,47)
 loop.position=[0,28,-10]
 atom('reconstruction/init',loop.id,'伴随初始化与参考内容','tensor','x⁰=Aᴴy · c⁰=Eᶜ(reference)','反投影目标测量，独立提取参考内容并 clone 为可更新内容。',R`x^0=A^*y,\quad c^0=E^c_r(x_r)`,recon,77)
 const cc=folder('reconstruction/cc',loop.id,'内容一致性','xᵏ,cᵏ → x̃ᵏ','以当前内容和目标风格形成网络约束图像。','abs → 强度归一化/补边 → target style encoder → target decode(content,style) → 去补边/恢复量纲/复数。',R`\tilde x^k=G_t(c^k,E^s_t(x^k))`,recon,83)
 atom(`${cc.id}/magnitude`,cc.id,'复数取幅值','magnitude','complex → real','image.abs()。','',recon,159)
 atom(`${cc.id}/rescale`,cc.id,'映射到网络动态范围','rescale','B × 1 × H × W','clip=True，归一化到 [−1,1]。','',recon,162)
 atom(`${cc.id}/pad`,cc.id,'补齐网络输入尺寸','tensor','H,W → 可被 4 整除','外部 pad_to_nearest_divisible_size(...,strict=False)；当前仓库不包含 llmr 工具内部实现，不假定补边取值。',R`H'=4\lceil H/4\rceil,\quad W'=4\lceil W/4\rceil`,recon,163)
 folder(`${cc.id}/style`,cc.id,'调用目标风格塔','B × 2 × 1 × 1','估计当前混叠影像的目标对比风格。','同 target 域 style_encoder，完整网络见双域网络目录；本节点作为调用边界。',R`s^k=E^s_t(x^k)`,recon,85)
 folder(`${cc.id}/decode`,cc.id,'调用固定目标生成器','B × 1 × H′ × W′','用被参考引导的内容替换当前估计内容。','decode(content_estim,style_estim) 内部调用 style MLP 和 AdaIN decoder。',R`\tilde x^k=G_t(c^k,s^k)`,recon,86)
 atom(`${cc.id}/unpad`,cc.id,'恢复图像范围与复数表示','unpad','H′×W′ → H×W complex','unpad → 从 [−1,1] 恢复强度 → 添加零虚部。',R`x=\operatorname{unpad}(\hat x)+i0`,recon,152);b.chain(cc.id)
 function sense(id:string,parent:string,title:string,adjoint=false){
  const e=folder(id,parent,title,adjoint?'B × coils × H × W → B × 1 × H × W':'B × 1 × H × W → B × coils × H × W','实现带相位、多线圈灵敏度与采样掩码的线性测量模型。',adjoint?'mask → centered IFFT → 乘 csm.conj → coil sum → exp(−i phase)。':'exp(i phase) → 乘 csm → centered FFT → mask。',adjoint?R`A^*k=e^{-i\phi}\sum_c\bar S_c\mathcal F^{-1}(Mk_c)`:R`Ax=M\mathcal F(S_ce^{i\phi}x)`,recon,adjoint?108:103)
  const ops: [string,string,Op,Glyph][] = adjoint?[['mask','采样掩码','spatial-mask','multiply'],['ifft','中心化逆傅里叶','ifft','up'],['coils','共轭线圈灵敏度','complex-multiply','multiply'],['sum','线圈维累加','sum','sum'],['phase','消除相位因子','complex-multiply','multiply']]:[['phase','加入相位因子','complex-multiply','multiply'],['coils','线圈灵敏度编码','complex-multiply','multiply'],['fft','中心化傅里叶','fft','up'],['mask','实际采样位置','spatial-mask','multiply']]
  ops.forEach(([key,title,op,glyph])=>atom(`${id}/${key}`,id,title,op,'复数 C × H × W',key==='mask'?'测量采样掩码逐位置相乘；不使用 VAE 潜维掩码。':e.implementation,e.formula,recon,adjoint?109:104,glyph));b.chain(id);return e
 }
 const dc=folder('reconstruction/dc',loop.id,'数据一致性','complex image → complex image','纠正与已测 k-space 不一致的影像分量。','先应用 A 并减去实测 y，再通过 Aᴴ，乘 1/max_eig 后从 x̃ 中减去。',R`x^{k+1}=\tilde x^k-\alpha A^*(A\tilde x^k-y),\quad\alpha=1/\lambda_{max}`,recon,89)
 sense(`${dc.id}/forward`,dc.id,'前向测量算子 A')
 atom(`${dc.id}/residual`,dc.id,'预测减去实测 k-space','subtract','B × coils × H × W','sense2d_forward_op(image)-kspace。',R`r=A\tilde x-y`,recon,89,'sum')
 sense(`${dc.id}/adjoint`,dc.id,'伴随测量算子 Aᴴ',true)
 atom(`${dc.id}/step`,dc.id,'按谱上界缩放','scale','B × 1 × H × W','dc_step_size=1/max_eig。',R`g=\alpha A^*r`,recon,75,'multiply')
 atom(`${dc.id}/subtract`,dc.id,'更新影像估计','subtract','B × 1 × H × W','x̃−g；保持复数测量模型。',dc.formula,recon,89,'sum');b.chain(dc.id)
 const cr=folder('reconstruction/cr',loop.id,'内容梯度修正','content 2 × H′ × W′','允许参考内容适应目标测量中的差异。','重新估计 DC 后的 style；style.detach；content.detach 后 requires_grad=True；对内容求测量残差平方和梯度，再更新 content。',R`c^{k+1}=c^k-\eta\nabla_c\|AG_t(c,\operatorname{sg}(s))-y\|_2^2`,recon,167)
 atom(`${cr.id}/detach`,cr.id,'固定风格与本轮梯度边界','detach','c,s','仅内容成为梯度输入；style.detach()，不累积历史迭代计算图。',R`c\leftarrow\operatorname{sg}(c),\quad c.requires\_grad=True`,recon,170,'split')
 atom(`${cr.id}/gradient`,cr.id,'穿过固定模型的链式梯度','gradient','2 × H′ × W′','loss=||mask*(A G(c,s)-y)||₂²；autograd.grad(outputs=loss,inputs=content)。',cr.formula,recon,176,'network')
 atom(`${cr.id}/update`,cr.id,'内容沿负梯度更新','gradient','2 × H′ × W′','content_estim−weight*content_grad；默认教学对照步长 0.1。',cr.formula,recon,186,'sum');b.entries[`${cr.id}/detach`].math!.steps=[{title:'重置本轮计算图',formula:R`c'=\operatorname{sg}(c)`,explanation:'丢弃历史迭代图，保留当前内容值。'},{title:'内容重新启用梯度',formula:R`c'.requires\_grad=True`,explanation:'此新叶张量参与当前损失；并非阻止本轮内容梯度。'},{title:'风格保持固定',formula:R`s'=\operatorname{sg}(s)`,explanation:'只有风格分支的反向箭头被阻断；梯度继续穿过固定解码器到内容。'}];b.chain(cr.id);b.chain(loop.id);b.link(loop.id,cr.id,cc.id,'下一轮 content',true)
 const train=folder('training','root','离线训练与多尺度判别器','仅图像域监督','先进行无配对建模，再利用配准数据微调。','无 k-space 训练数据。两域生成器与判别器交替优化；body mask 与图像拼接成双通道判别器输入。',R`\mathcal L=\mathcal L_{GAN}+\mathcal L_{image}+\mathcal L_{content}+\mathcal L_{style}`,system,166);train.position=[0,-13,-10]
 for(const m of [1,2]){
  const dis=folder(`training/dis${m}`,train.id,`域 ${m} · 三尺度 PatchGAN`,'2 × H × W → 3 个 Patch 网格','在多个尺度约束合成影像。','每尺度独立权重：2→64/s2→128/s2→256/s2→512/s2→512/s1→1/s1。所有 Conv2dBlock 均保留默认 LeakyReLU，包括最后分数层。尺度间用 bilinear 0.5、align_corners=True。',R`D_m(x,b)=\{D_{m,r}(\operatorname{Down}_{2^r}[x,b])\}_{r=0}^2`,'pnp_cosmo/cosmo/discriminators.py',50)
  for(let s=0;s<3;s++){
   const scale=folder(`${dis.id}/scale${s}`,dis.id,`Patch 判别尺度 ${s+1}`,`H/${2**s} × W/${2**s}`,'独立的局部真实性评分网络。',dis.implementation,dis.formula,'pnp_cosmo/cosmo/discriminators.py',10)
   const channels=[2,64,128,256,512,512,1]
   for(let j=0;j<6;j++)convBlock(`${scale.id}/conv${j}`,scale.id,`判别卷积 ${j+1}`,channels[j],channels[j+1],3,j<4?2:1)
   b.chain(scale.id)
  }
 }
 for(const [key,title,op] of [['gan','最小二乘对抗','lsgan'],['image','同域图像重建','l1'],['content','跨域后内容恢复','l1'],['style','随机风格恢复','l1'],['cross-image','配对跨域图像','l1'],['cross-content','配对内容对齐','l1']] as [string,string,Op][])atom(`training/${key}`,train.id,title,op,'scalar','具体权重由 pretrain / pft YAML 指定；无配对阶段随机采样目标 style，配对阶段采用目标真实编码 style。','',system,230,'sum')
 // Call-site entries reference the actual same-weight network via children, not an unexplained leaf glyph.
 for(const [call,target] of [['reconstruction/cc/style','style2'],['reconstruction/cc/decode','decoder2']] as const){
  const targetEntry=b.entries[target],callEntry=b.entries[call]
  const clone=(original:string,parent:string)=>{const source=b.entries[original],id=`${parent}/${original.split('/').at(-1)}`,copy:Entry={...source,id,parent,children:[] as string[],edges:[],position:undefined};b.entries[id]=copy;b.entries[parent].children.push(id);for(const child of source.children)clone(child,id);copy.edges=source.edges.map(edge=>({...edge,from:edge.from.replace(original,id),to:edge.to.replace(original,id)}));return id}
  if(target==='decoder2')clone('mlp2',callEntry.id);for(const child of targetEntry.children)clone(child,callEntry.id);b.chain(callEntry.id)
 }
 const overview:string[]=[]
 for(const m of [1,2]){overview.push(`image${m}`,...b.entries[`content${m}`].children,...b.entries[`style${m}`].children,`mlp${m}`,...b.entries[`decoder${m}`].children)}
 overview.push(loop.id,train.id)
 const overviewEdges=[] as PaperModel['overviewEdges']
 for(const m of [1,2]){const c=b.entries[`content${m}`],s=b.entries[`style${m}`],d=b.entries[`decoder${m}`];overviewEdges.push(...c.edges,...s.edges,...d.edges,{from:`image${m}`,to:c.children[0]},{from:`image${m}`,to:s.children[0]},{from:c.children.at(-1)!,to:d.children[0],label:'空间 content'},{from:s.children.at(-1)!,to:`mlp${m}`},{from:`mlp${m}`,to:d.children[0],label:'所有 AdaIN 的风格条件'},{from:c.children.at(-1)!,to:loop.id,label:'参考 content'},{from:d.children.at(-1)!,to:train.id,label:'图像域监督'})}
 // Subset folders use exactly these positions; network graph connections are supplied separately by the viewer.
 net.children.forEach(id=>{const e=b.entries[id];e.position=id==='domain1'?[-22,10,0]:[22,10,0]})
 return {id:'pnp',name:'PnP-CoSMo',repo:'https://github.com/cnmy-ro/pnp-cosmo',commit:'01fd04be51b1c7417aebcd5461b87b95b09e6cb2',entries:b.entries,overview,overviewEdges,intro:'双域内容/风格网络与 CC → DC → CR 重建循环，在同一空间逐层拆开。',config:'NYU YAML · content=2 · style=2 · width=32 · 4 ResBlocks · content downsample=0 · MLP=2',equations:[{title:'内容一致性与数据一致性',formula:R`\begin{aligned}s^k&=E^s_t(x^k),&\tilde x^k&=G_t(c^k,s^k)\\x^{k+1}&=\tilde x^k-\alpha A^*(A\tilde x^k-y),&\alpha&=1/\lambda_{max}\end{aligned}`,explanation:'当前估计的 style 与被参考引导的 content 组合；随后用实测数据纠正。相位、线圈灵敏度和采样掩码均属于 A。'},{title:'content refinement',formula:R`c^{k+1}=c^k-\eta\nabla_c\|M(AG_t(c,\operatorname{sg}(s^{k+1}))-y)\|_2^2`,explanation:'A 内已经包含 mask；源码在损失处再次乘 mask。只对 content 求梯度；不更新模型参数。示例 notebook 运行 200 轮，步长 η=0.1。'},{title:'实现中的 AdaIN',formula:R`\begin{aligned}(\gamma,\beta)&=\operatorname{chunk}(\operatorname{LReLU}_{.2}(Ws+b),2)\\\operatorname{AdaIN}(c,s)&=\beta+(1+\gamma)\odot\frac{c-\mu(c)}{\sqrt{v(c)+10^{-5}}}\end{aligned}`,explanation:'风格 fc 自身含 LeakyReLU；内容残差块每次卷积后都有独立 AdaIN。1+gamma 与通常简写的 gamma 有差别。'}],audit:[{title:'默认网络保留内容分辨率',detail:'NYU YAML 的 content_downsamples=0，因此实际无 content 下采样、无 decoder 上采样。content=2；MLP=2，而类构造默认分别是 4 和 8。',file:'pnp_cosmo/configs/nyu_pft.yaml',line:24},{title:'风格头与末层激活',detail:'StyleEncoder 末尾是裸 1×1 Conv；content 投影使用 Conv2dBlock 默认 LeakyReLU，decoder 最后一层为 Tanh。',file:ae,line:368},{title:'公开训练接口存在不一致',detail:'DeterministicContentEncoder 返回单个 Tensor，但 MUNITAutoEncoder.encode 按 mean/logvar 解包，MUNIT 训练调用又按 content/style 解包。本展示按实际层定义和重建中的直接 encoder/decode 调用还原，不声称该训练入口可原样运行。',file:ae,line:81},{title:'配对训练分支中的未定义量',detail:'MUNIT._compute_autoencoder_loss 在配对分支仍提前读取 style_*_rand，content_cross 处使用未定义 content_params_*。应与 YAML/论文定义的训练目标区分。',file:system,line:253},{title:'测量工具是外部依赖',detail:'fft2c、ifft2c、补边和强度变换来自 /path/to/llmr，仓库不包含其实现。数学实验使用明确标注的中心化正交 DFT 示例；不冒充真实 MRI 重建或权重推理。',file:recon,line:8}]}
}
