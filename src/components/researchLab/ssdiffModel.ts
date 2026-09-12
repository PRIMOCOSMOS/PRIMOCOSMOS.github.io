import {CoreBuilder} from './coreBuilder'
import type {PaperModel} from './catalog'
const R=String.raw,F='ssdiff/model.py'
export function ssdiffModel():PaperModel{
 const b=new CoreBuilder(F)
 b.folder('root',undefined,'SSDiff · 以扩散方向为 token','B × V × G → B × 1536','从原始扩散信号学习受试者级表示，不预先拟合微结构标签。','固定 train.py / inference.py：embedding=1536、heads=16、depth=12；12 个输入块 + 1 个中间块，重建端再执行 6 个输出块。skip=False，out_chans=[1]。',R`X\in\mathbb R^{B\times V\times G}\to H\in\mathbb R^{B\times(G+1)\times1536}`,270)
 const prep=b.folder('preprocess','root','空间对齐与采集几何','V × G · V × G × 3','使不同采集方案具有对应的体素和方向解释。','FSL 配准到 MNI 后，加载信号与局部形变。V 由 brain mask 决定；G 随个体改变。','',38,'ssdiff/dataloader.py');prep.position=[-15,60,0]
 b.atom('preprocess/signal',prep.id,'基线归一化与对数信号','log-signal','V × G','b<50 均值为基线；log(abs(S/S0))；NaN 和 Inf 置零。','',46,'tensor','ssdiff/dataloader.py')
 b.atom('preprocess/rotation',prep.id,'局部梯度重定向','rotation','V × G × 3','由 inverse(I+warp) 的极分解获得旋转，b-vector 重新单位化。','',56,'linear','ssdiff/dataloader.py')
 b.atom('preprocess/permute',prep.id,'同步重排扩散方向','reshape','V × G','信号、bvals、bvecs 使用同一 randperm，保持对应。',R`(X,b,g)\mapsto(X_\pi,b_\pi,g_\pi)`,71,'tensor','ssdiff/dataloader.py');b.link(prep.id,'preprocess/signal','preprocess/permute');b.link(prep.id,'preprocess/rotation','preprocess/permute')
 const embed=b.folder('embedding','root','空间投影与物理条件','B × G × 1536','把整个方向图像映射为一个 token。','1×1 Conv1d 的 in_channels 是 V 个体素，序列轴是 G；这不是每体素一个 token。体素位置编码在投影前加入。',R`h_g=P(X_{:,g}+p_{voxel})+\mathrm{LN}(E_g(g))+\mathrm{LN}(E_b(b/1000))`,361);embed.position=[-15,45,0]
 b.atom('embedding/dropout',embed.id,'训练信号遮挡','dropout','B × V × G','F.dropout(data_in,p=.75)，保留值按 1/(1-p) 缩放；推理不执行。','',142,'multiply','train.py')
 b.atom('embedding/position',embed.id,'MNI 坐标嵌入','linear','V × 300 → V × 1','三轴各 100 维正余弦坐标拼接，再 Linear(300,1)，广播加到每个方向。',R`p_v=W[\sin\omega x,\cos\omega x,\ldots]+b`,337,'linear')
 b.atom('embedding/add-position',embed.id,'加入空间位置','add','B × V × G','位置在 patch embedding 之前加入。',R`X'=\tilde X+p_v`,365,'sum')
 b.atom('embedding/project',embed.id,'整个方向图像投影','linear','V → 1536 · 每方向','Conv1d(V,1536,kernel_size=1,stride=1) 数学上是逐方向含偏置全连接。',R`h_g=W X'_{:,g}+b`,262,'linear')
 b.atom('embedding/bvec',embed.id,'方向场投影与轴均值','linear','V → 1536 · 3 坐标均值','Linear(V,1536) 对每个方向的三个坐标场分别投影，再 mean(dim=2)。',R`e_g=\tfrac13\sum_{a=1}^3W_g g_{:,a}+b_g`,397,'linear')
 b.atom('embedding/norm-g',embed.id,'方向嵌入 LayerNorm','layernorm','B × G × 1536','norm01(bvec_embed)。','',398)
 b.atom('embedding/bval',embed.id,'b-value 投影','linear','1 → 1536','bval_mapping，输入 b/1000。',R`e_b=W_b(b/1000)+b_b`,318,'linear')
 b.atom('embedding/norm-b',embed.id,'强度嵌入 LayerNorm','layernorm','B × G × 1536','norm02(bval_mapping(...))。','',398)
 b.atom('embedding/fuse',embed.id,'三路嵌入相加','add','B × G × 1536','图像投影 + 标准化方向编码 + 标准化 b-value 编码。',embed.formula,398,'sum')
 b.link(embed.id,'embedding/dropout','embedding/add-position');b.link(embed.id,'embedding/position','embedding/add-position','位置条件',true);b.link(embed.id,'embedding/add-position','embedding/project');b.link(embed.id,'embedding/project','embedding/fuse');b.link(embed.id,'embedding/bvec','embedding/norm-g');b.link(embed.id,'embedding/norm-g','embedding/fuse','方向',true);b.link(embed.id,'embedding/bval','embedding/norm-b');b.link(embed.id,'embedding/norm-b','embedding/fuse','b-value',true)
 b.atom('tokens','root','前置可学习 token','tensor','B × (G+1) × 1536','out_chans=[1] 仅前置 1 个 learned cls_token；不是默认构造参数的 60 个参数 token。',R`H_0=[c_{learned};h_1;\ldots;h_G]`,400,'concat').position=[-15,32,0]
 const encoder=b.folder('encoder','root','十二层方向注意力塔','B × (G+1) × 1536','沿扩散方向序列学习跨方向关系。','12 个独立 Block，16 heads，head_dim=96，MLP=6144；两个 PreNorm 残差。','',292)
 for(let i=0;i<12;i++)b.transformer(`encoder/block${i}`,encoder.id,`方向 Transformer ${i+1}`,1536,16,6144,228)
 b.chain(encoder.id);b.vertical(encoder.id,-15,23,5)
 b.transformer('middle','root','中间 Transformer',1536,16,6144,299).position=[-15,-40,0]
 b.atom('middle-norm','root','中间特征 LayerNorm','layernorm','B × (G+1) × 1536','两条分支在这里分离：特征提取停止向解码器前进，重建训练继续。','',311).position=[0,-47,0]
 const features=b.folder('features','root','受试者级 1536 维表征','B × 1536','聚合扩散方向，保持输出维度不随 G 改变。','get_features() 对各 token 组分别平均；inference.py 最终只保存 fea[-1]，即所有真实方向 token 的均值。',R`f=\frac1G\sum_{g=1}^GH_g`,438);features.position=[18,-40,8]
 b.atom('features/select',features.id,'去除前置 learned token','tensor','B × G × 1536','x[:,channel_index[-1]:,:]；不平均 cls token。',R`H_{directions}=H[:,1:,:]`,477)
 b.atom('features/mean',features.id,'沿方向维平均','token-mean','B × G × 1536 → B × 1536','沿 dim=1 聚合方向，非空间平均。',R`f_d=\frac1G\sum_gH_{gd}`,477,'pool');b.chain(features.id)
 const decoder=b.folder('decoder','root','六层信号重建塔','B × (G+1) × 1536','训练期从上下文恢复未遮挡原信号。','6 个输出 Block；train.py 指定 skip=False，所以没有 skip_linear，也没有 U 型跳接融合。','',305)
 for(let i=0;i<6;i++)b.transformer(`decoder/block${i}`,decoder.id,`重建 Transformer ${i+1}`,1536,16,6144,228)
 b.chain(decoder.id);b.vertical(decoder.id,16,20,6)
 const readout=b.folder('readout','root','逐方向重建信号','B × G × V','回到输入信号空间，构造自监督目标。','norm1 → Linear(1536,V) → 移除前置 token；不需要 DTI/NODDI 标签。',R`\hat X=\operatorname{Linear}(\mathrm{LN}(H))[:,1:,:]`,423);readout.position=[16,-22,0]
 b.atom('readout/norm',readout.id,'输出 LayerNorm','layernorm','B × (G+1) × 1536','norm1(x)。','',423)
 b.atom('readout/linear',readout.id,'返回全脑体素向量','linear','1536 → V','recon_orig 的权重对各方向共享。','',425,'linear')
 b.atom('readout/slice',readout.id,'只取真实方向 token','tensor','B × G × V','x[:,total_chans:,:]；total_chans=1。',R`\hat X=X[:,1:,:]`,427);b.chain(readout.id)
 b.atom('loss','root','全部原信号的均方误差','lsgan','scalar','训练源码 mean((recon-original)²)，不是仅 mask 位置，也没有 0.5 因子。',R`\mathcal L=\operatorname{mean}(\hat X-X)^2`,148,'sum','train.py').position=[16,-33,0]
 b.entries.loss.math=coreSpecMSE()
 b.link('root',prep.id,embed.id);b.link('root',embed.id,'tokens');b.link('root','tokens',encoder.id);b.link('root',encoder.id,'middle');b.link('root','middle','middle-norm');b.link('root','middle-norm',features.id,'特征提取');b.link('root','middle-norm',decoder.id,'训练重建',true);b.link('root',decoder.id,readout.id);b.link('root',readout.id,'loss')
 const o=b.overview()
 return {id:'ssdiff',name:'SSDiff',repo:'https://github.com/weikanggong1/SSDiff',commit:'56361903d23dd255702a092c6f0ac0e5772db359',entries:b.entries,overview:o.keys,overviewEdges:o.edges,intro:'以一幅扩散方向图像为一个 token，从空间、采集几何和跨方向注意力追踪到 1536 维表征。',config:'train.py · E=1536 · 16 heads · 12+1+6 Blocks · 输入 dropout=0.75 · skip=False',equations:[{title:'信号表示与方向注意力',formula:R`X\in\mathbb R^{V\times G},\quad H=X^TW+E_b(b)+E_g(g),\quad A=\operatorname{softmax}(QK^T/\sqrt{96})`,explanation:'空间体素是投影输入维，扩散方向是 attention 的序列维。V 由配准后的脑掩码决定；可变 G 不要求改变输出维度。'},{title:'自监督训练与特征读取',formula:R`\tilde X=M\odot X/0.25,\quad\mathcal L=\operatorname{mean}(\hat X-X)^2,\quad f=\operatorname{mean}_{g}H_g`,explanation:'训练代码输入 dropout 不移除 token；损失覆盖全部原始信号。推理读取中间 LayerNorm 后的方向 token 均值。'}],training:[{title:'数据对齐',detail:'固定 MNI 体素空间；用局部形变重定向梯度，输入 log(|S/S0|)。'},{title:'自监督重建',detail:'batch=1，累积 64 步；默认 50 epochs、学习率 10⁻⁴；仅原信号 MSE。'},{title:'表征提取',detail:'12+1 个 Transformer 后沿真实方向平均，保存为每受试者 1536 维 CSV；不运行六层重建端。'}],audit:[{title:'公开脚本覆盖的配置',detail:'类默认 embed_dim=256 和 out_chans=[7,45,8] 不等于公开训练脚本。展示固定为 1536 和 [1]，六个输出块无跳接融合。',file:'train.py',line:113},{title:'特征不是逐体素预测',detail:'inference.py 保存 fea[-1]，每行一个受试者；README 的体素级措辞不能替代实际 token 轴和特征读取。',file:'inference.py',line:104},{title:'辅助参数头不构成训练监督',detail:'model_param 和可选 mlp_head 存在，但当前 train.py 只优化重建 MSE，get_features 不走这些预测头。',file:F,line:413},{title:'数值与论文身份',detail:'原子动画为可核算小型示例。SSDiff 是自监督 dMRI Transformer，不是生成式扩散去噪模型；“Diff”不能据此解释为 DDPM。',file:'README.md',line:1}]}
}
import {coreSpec} from './coreOperators'
function coreSpecMSE(){const s=coreSpec('scores');s.operation='core:mse';s.steps=[{title:'预测与原信号',formula:R`e=\hat X-X`,explanation:'两个张量逐位置对齐，损失目标包含遮挡与未遮挡位置。'},{title:'误差平方',formula:R`q_i=e_i^2`,explanation:'正负误差均产生非负代价。'},{title:'全部元素平均',formula:R`\mathcal L=N^{-1}\sum_iq_i`,explanation:'无 0.5 系数。反向传播时再除 accumulation_steps。'}];return s}
