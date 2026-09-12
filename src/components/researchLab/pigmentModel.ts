import {CoreBuilder} from './coreBuilder'
import type {PaperModel} from './catalog'
const R=String.raw
export function pigmentModel():PaperModel{
 const b=new CoreBuilder('论文 · Methods / Network implementation')
 const unknown='论文未给出此块的通道宽度、卷积核、归一化和激活顺序；当前只展示已披露的功能与数学结构，不假定源码实现。'
 b.folder('root',undefined,'PIGMENT · 物理约束生成先验','3D 参数体 → 离散潜空间 → 个体参数图','用生成先验和实测信号共同约束稀疏采集下的微结构估计。','论文 v1 的代码声明为接收后公开。此处按 Fig.1、Methods 与 Network implementation 展示论文层次；未获得作者源码。',R`\mathrm{VQ}\text{-}\mathrm{AE}+\epsilon_\theta(z_t,t,S_0)+\mathcal L_{consistency}+\lambda\mathcal L_{manifold}`,22)
 b.atom('parameters','root','微结构参数体','tensor','64 × 64 × 64 × channels','训练块为64³；tensor、NODDI、kurtosis使用分别训练的参数框架。',R`x\in\mathbb R^{64\times64\times64\times C}`,29,'image').position=[-18,57,0]
 const ae=b.folder('autoencoder','root','三维 VQ 自编码器','k=7 / 7 / 22 · 16,000 codes','在离散潜空间压缩高维参数体。','三次下采样、各尺度两个残差块；码本大小16,000。嵌入 k 对 DTI/NODDI/DKI 分别为7/7/22，不能当成所有隐藏层通道宽度。',R`z_e=\mathcal E(x),\quad z_q=e_{\arg\min_j\|z_e-e_j\|^2},\quad\hat x=\mathcal D(z_q)`,29)
 ae.overview=true
 const encoder=b.folder('encoder',ae.id,'参数体编码塔','3 次空间下采样','从体积参数逐级获得紧凑空间表示。',unknown,'',29)
 for(let i=0;i<3;i++){
  const scale=b.folder(`encoder/scale${i}`,encoder.id,`编码尺度 ${i+1}`,'空间尺寸与宽度未逐层公开','保留该尺度的空间结构并进行压缩。','每尺度两个残差块；三次下采样。具体倍率未在实现段逐层列出。','',29)
  for(let j=0;j<2;j++)b.atom(`${scale.id}/res${j}`,scale.id,`残差块 ${j+1} · 功能原理`,'add','同形状 3D 张量',unknown,R`y=x+F(x)`,29,'network')
  b.atom(`${scale.id}/down`,scale.id,'空间压缩 · 论文层次','reshape','高分辨率 → 紧凑网格','仅用网格重组说明多尺度压缩概念；不声称真实下采样是 reshape。'+unknown,R`z_{l+1}=\operatorname{Down}_l(z_l)`,29,'pool');b.chain(scale.id)
 }
 b.chain(encoder.id);b.vertical(encoder.id,-18,44,10)
 b.atom('quantizer',ae.id,'最近码字量化','vq','16,000 × k codebook','逐潜向量选择欧氏距离最近码字。示例只显示四个候选；完整码本大小来自论文。','',22,'poe').position=[-18,10,0]
 const decoder=b.folder('decoder',ae.id,'参数体解码器','离散潜表示 → 3D 参数','由码本表示恢复几何与标量参数。',unknown,R`\hat x=\mathcal D(z_q)`,22);decoder.position=[-18,-3,0]
 b.atom('decoder/latent',decoder.id,'接收量化潜表示','tensor','空间网格 × k','所选码字保留空间索引。',R`z_q`,22)
 b.atom('decoder/output',decoder.id,'恢复参数域 · 功能边界','tensor','64³ × 参数通道',unknown,R`\hat x=\mathcal D(z_q)`,22,'image');b.chain(decoder.id)
 b.link(ae.id,encoder.id,'quantizer');b.link(ae.id,'quantizer',decoder.id)
 const loss=b.folder('vq-loss',ae.id,'几何感知 VQ 目标','scalar','分别处理标量与方向几何，并约束离散码本。','Lrec=Lscalar+Langular；量化含码本更新项和β=0.25 commitment项。',R`\mathcal L_{VQ}=\|x_s-\hat x_s\|_1+1-(\bar x_a^T\bar{\hat x}_a)^2+\|\mathrm{sg}(z_e)-e\|_2^2+.25\|z_e-\mathrm{sg}(e)\|_2^2`,22);loss.position=[-28,-18,8]
 b.atom('vq-loss/scalar',loss.id,'标量参数 L1','l1','scalar','均值绝对误差。',R`\mathcal L_s=\mathbb E\|x_s-\hat x_s\|_1`,22,'sum')
 b.atom('vq-loss/angular',loss.id,'轴向方向损失','angular','scalar','单位化后点积平方，平行与反平行等价。','',22,'norm')
 b.atom('vq-loss/codebook',loss.id,'码本与编码承诺','vq','latent × k','sg 两路分别控制码本和编码器梯度。',R`\|\mathrm{sg}(z_e)-e\|^2+.25\|z_e-\mathrm{sg}(e)\|^2`,22,'poe')
 const diffusion=b.folder('diffusion','root','条件潜空间去噪网络','3D latent · 四个尺度','以 S0 提供解剖条件，学习微结构潜变量分布。','attention 3D U-Net，四尺度；每尺度一个自注意力位于两个残差块之间。训练T=1000；源码未公开，条件注入位置与具体宽度不确定。',R`\hat\epsilon=\epsilon_\theta(z_t,t,S_0)`,29)
 for(let i=0;i<4;i++){
  const scale=b.folder(`diffusion/scale${i}`,diffusion.id,`去噪尺度 ${i+1}`,'3D 特征 · 宽度未披露','结合局部体积变换与跨位置关系。','论文明确顺序：Residual → Self-attention → Residual。'+unknown,R`F_l=R_{l,2}\circ A_l\circ R_{l,1}`,29)
  b.atom(scale.id+'/res1',scale.id,'前残差块 · 功能原理','add','3D 特征',unknown,R`y=x+F_1(x)`,29,'network')
  const att=b.folder(scale.id+'/attention',scale.id,'空间自注意力原理','空间位置 × 特征','解释一个位置如何汇聚其余空间位置。','公式为标准自注意力教学分解；论文未公开 head数、投影尺寸或具体实现。',R`A=\operatorname{softmax}(QK^T/\sqrt d)V`,29)
  b.atom(att.id+'/scores',att.id,'位置关系点积','scores','N × N','缩小的4×4数值例；模型真实N未公开。','',29,'linear');b.atom(att.id+'/softmax',att.id,'行权重归一化','softmax','N × N','按key方向归一化。','',29,'activation');b.atom(att.id+'/values',att.id,'空间值汇聚','weighted','N × d','标准注意力原理，不代表已核验源码。','',29,'linear');b.chain(att.id)
  b.atom(scale.id+'/res2',scale.id,'后残差块 · 功能原理','add','3D 特征',unknown,R`z=y+F_2(y)`,29,'network');b.chain(scale.id)
 }
 b.chain(diffusion.id);b.vertical(diffusion.id,18,45,10)
 b.atom('s0','root','非扩散加权条件 S₀','tensor','3D baseline volume','不同采集协议均包含S0，论文以此作为高信噪比解剖条件。',R`S_0`,22,'image').position=[32,57,6]
 b.atom('noise','root','前向潜变量加噪','noise','latent shape','随机t∈1…1000，训练预测注入噪声。','',22,'sample').position=[0,57,-8]
 const sampling=b.folder('sampling','root','DDIM 与双约束推断','500 步采样','用个体实测数据校正生成先验。','从高斯噪声开始；论文实现段：500 DDIM步，第480和500步执行双域优化，学习率10⁻⁵。并非每步均优化。',R`z_T\sim\mathcal N(0,I)\longrightarrow z_0`,30);sampling.position=[18,-6,0]
 b.atom('sampling/ddim',sampling.id,'估计无噪声 latent 并推进','ddim','latent','DDIM中保留δ随机性参数；示例显示δ=0分支。','',23,'sample')
 const dual=b.folder('sampling/dual',sampling.id,'第480 / 500步 · 双域优化','参数域 + latent域','同时约束信号一致性与生成流形。','先解码当前参数，与微结构前向模型的信号比较；再重编码并加噪，约束其去噪一致性。',R`\mathcal L_{dual}=\|S-\mathrm{Model}(\hat x)\|^2+\lambda\mathcal L_{manifold}`,23)
 b.atom(dual.id+'/decode',dual.id,'解码当前微结构估计','tensor','3D 参数','使用固定VQ decoder。',R`\hat x=\mathcal D(\hat z_0)`,23)
 b.atom(dual.id+'/physics',dual.id,'DTI 信号前向模型','tensor-signal','参数 → 已采集方向','这里交互演示DTI；论文还分别使用DKI和NODDI前向模型。','',28,'gaussian')
 b.atom(dual.id+'/data',dual.id,'实测信号一致性','subtract','同一采集方向','预测信号和实测S比较，优化平方误差。',R`\mathcal L_{data}=\|S-\mathrm{Model}(\hat x)\|_2^2`,23,'sum')
 b.atom(dual.id+'/encode',dual.id,'重编码候选参数','tensor','候选参数 → latent','固定编码器映射候选参数。',R`\tilde z=\mathcal E(\hat x_{opt})`,23)
 b.atom(dual.id+'/noise',dual.id,'随机小时间步加噪','noise','latent','论文k∈[0,100]。','',23,'sample')
 b.atom(dual.id+'/manifold',dual.id,'去噪残差近似流形梯度','subtract','latent gradient','式11使用带权噪声残差近似，不反传整个U-Net Jacobian；式10与11噪声记法存在省略，待源码确认。',R`\nabla_z\mathcal L_{manifold}\approx\mathbb E[w(k)(\epsilon_\theta-\epsilon)],\quad w(k)=1-\bar\alpha_k`,23,'sum')
 b.link(dual.id,dual.id+'/decode',dual.id+'/physics');b.link(dual.id,dual.id+'/physics',dual.id+'/data');b.link(dual.id,dual.id+'/decode',dual.id+'/encode','流形分支',true);b.link(dual.id,dual.id+'/encode',dual.id+'/noise');b.link(dual.id,dual.id+'/noise',dual.id+'/manifold')
 b.atom('sampling/resample',sampling.id,'优化表示与先验重新混合','noise','latent','式13：优化参数编码后加到对应时间噪声水平，再与原DDIM轨迹按γ混合。',R`z_{t-1}=(1-\gamma)z'_{t-1}+\gamma(\sqrt{\bar\alpha_{t-1}}\mathcal E(\hat x_{opt})+\sqrt{1-\bar\alpha_{t-1}}\epsilon)`,24,'sum');b.chain(sampling.id)
 b.atom('output','root','个体微结构参数图','tensor','DTI / NODDI / DKI','对应任务单独训练：不是一个检查点同时统一三种参数空间。',R`\hat x=\mathcal D(z_0)`,21,'image').position=[18,-23,0]
 b.link('root','parameters','autoencoder');b.link('root','autoencoder','noise','训练latent');b.link('root','noise','diffusion');b.link('root','s0','diffusion','解剖条件',true);b.link('root','diffusion','sampling');b.link('root','sampling','output')
 // Keep declared multiscale blocks in the overview without inventing an unpublished decoder stage topology.
 const overview=['parameters',...b.entries.encoder.children,'quantizer','decoder','vq-loss','s0','noise',...b.entries.diffusion.children,'sampling','output']
 const overviewEdges=[...b.entries.encoder.edges,...b.entries.diffusion.edges,{from:'parameters',to:'encoder/scale0'},{from:'encoder/scale2',to:'quantizer'},{from:'quantizer',to:'decoder'},{from:'decoder',to:'vq-loss'},{from:'quantizer',to:'noise'},{from:'noise',to:'diffusion/scale0'},{from:'s0',to:'diffusion/scale0',label:'S0 condition'},{from:'diffusion/scale3',to:'sampling'},{from:'sampling',to:'output'}]
 return {id:'pigment',name:'PIGMENT',repo:'https://arxiv.org/abs/2606.00156',paperUrl:'https://arxiv.org/pdf/2606.00156',sourceMode:'paper',commit:'论文 v1 · 2026-05-29',entries:b.entries,overview,overviewEdges,intro:'论文架构复原：离散微结构先验、S₀ 条件去噪，以及数据一致性与流形一致性的双约束采样。',config:'论文依据 · 3D VQ-AE · 16,000 codes · 四尺度 U-Net · 500 DDIM步 · 作者代码尚未确认公开',equations:[{title:'VQ 与参数几何',formula:R`\mathcal L_{VQ}=\mathcal L_{scalar}+\mathcal L_{angular}+\mathcal L_{quant},\quad\mathcal L_{angular}=1-(\bar x^T\bar{\hat x})^2`,explanation:'分别对待标量强度与轴向方向。量化损失的 commitment 权重β=0.25。'},{title:'个体信号与生成流形',formula:R`\mathcal L_{dual}=\mathbb E\|S-\mathrm{Model}(\mathcal D(\hat z_0))\|^2+\lambda\mathcal L_{manifold}`,explanation:'物理前向模型因DTI、DKI和NODDI而异；流形项采用噪声残差梯度近似。第480和500步执行优化。'},{title:'三种微结构前向模型',formula:R`\begin{aligned}S_{DTI}&=S_0e^{-b g^TDg}\\\log(S_{DKI}/S_0)&=-bD_{app}+\tfrac16b^2MD^2K_{app}\\S_{NODDI}&=S_0[(1-f_{iso})(f_{ic}A_{ic}+(1-f_{ic})A_{ec})+f_{iso}A_{iso}]\end{aligned}`,explanation:'前向预测在真实采集的 b-value 和方向上计算，和观测信号比较；不在k-space中做PnP-CoSMo的线圈测量算子。'}],training:[{title:'离散先验学习',detail:'64³块，三次下采样，码本16,000；标量、方向与量化损失。Adam学习率5×10⁻⁵。'},{title:'条件扩散训练',detail:'3D注意力U-Net四尺度，T=1000；S0条件，噪声预测MSE。学习率2.5×10⁻⁵。'},{title:'个体优化采样',detail:'500 DDIM步；480/500步双域优化，学习率10⁻⁵；按γ重新混合轨迹。'}],audit:[{title:'代码尚未确认公开',detail:'v1第36页声明代码和数据将在稿件接收后公开。2026-09-12核验没有找到可确认的作者PIGMENT仓库；链接通向论文，未伪造文件名或提交号。',file:'Code and data',line:36},{title:'已披露与未知边界',detail:'三次下采样、每尺度两个残差块、四尺度去噪U-Net有论文依据；卷积核、通道宽度、归一化和S0注入方式未逐层披露。功能演示不冒充已核验源码。',file:'Network implementation',line:29},{title:'不同模型分别训练',detail:'论文说明参数空间异构，因此目前DTI、DKI、NODDI需分别训练；latent通道7/7/22不是统一多任务输出头。',file:'Limitations / implementation',line:21},{title:'公式记法与数值演示',detail:'流形梯度的噪声/条件记法需要作者实现进一步确认。动画使用标明的DTI和DDIM小型示例，不生成临床结果或虚构性能曲线。',file:'Physics-informed inference',line:23}]}
}
