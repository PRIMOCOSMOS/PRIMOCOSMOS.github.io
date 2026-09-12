import {CoreBuilder} from './coreBuilder'
import type {PaperModel} from './catalog'
const R=String.raw
export function metscModel():PaperModel{
 const b=new CoreBuilder('vit_pytorch.py')
 b.folder('root',undefined,'METSC · 注意力与稀疏字典','B × 60 × 3 × 3 → B × 3','从空间邻域和稀疏表示恢复 NODDI 微结构参数。','作者公开的是 NODDI 推理 demo：3×3 输入、60 方向、ViT 512 维/6 层/8 heads；601 维稀疏系数经九次共享字典更新，再生成 Vic、Viso、ODI。','',43,'test.py')
 b.atom('input','root','60 方向 · 3×3 局部影像','tensor','B × 60 × 3 × 3','双壳层，各 30 个方向；输入文件由 Dataform.py 读取。','',23,'image','test.py').position=[-14,60,0]
 const embed=b.folder('embedding','root','局部空间 token','B × 9 × 512','每个空间位置的 60 维信号成为一个 token。','Rearrange 3×3 空间为 9 个位置，再 Linear(60,512)，加可学习位置嵌入。没有独立 cls token。',R`h_i=W x_i+p_i`,109);embed.position=[-14,50,0]
 b.atom('embedding/rearrange',embed.id,'空间邻域展开','reshape','B × 60 × 3 × 3 → B × 9 × 60','patch_size=1，沿空间顺序排列九个位置。',R`[B,C,H,W]\to[B,HW,C]`,110)
 b.atom('embedding/linear',embed.id,'方向向量映射','linear','60 → 512','每个空间位置共享同一投影。','',111,'linear')
 b.atom('embedding/position',embed.id,'加可学习位置嵌入','add','B × 9 × 512','pos_embedding 为 1×9×512；emb_dropout=0.1，eval 时恒等。',R`h_i\leftarrow h_i+p_i`,132,'sum');b.dropout('embedding/dropout',embed.id,.1,133);b.chain(embed.id)
 const enc=b.folder('encoder','root','六层空间 Transformer 塔','B × 9 × 512','让邻域中的九个位置交换扩散信号信息。','heads=8，dim_head=64，MLP 隐藏宽512。每层除两条常规残差外，ff(x) 后再次加层输入 x1。','',82)
 for(let i=0;i<6;i++)b.transformer(`encoder/block${i}`,enc.id,`空间 Transformer ${i+1}`,512,8,512,82,true)
 b.chain(enc.id);b.vertical(enc.id,-14,39,6)
 const head=b.folder('head','root','信号残差预测头','B × 512 → B × 60','读取首位置 token，输出中心信号的学习修正。','x[:,0] → LayerNorm512 → Linear512 → GELU → Linear60。源码没有全局池化，也没有额外 cls token。',R`t=W_2\operatorname{GELU}(W_1\mathrm{LN}(H[:,0])+b_1)+b_2`,121);head.position=[-14,-3,0]
 b.atom('head/first',head.id,'取首位置 token','tensor','B × 512','self.to_latent(x[:,0])；不是空间均值。',R`h=H[:,0]`,135)
 b.atom('head/norm',head.id,'输出 LayerNorm','layernorm','B × 512','沿512维归一化。','',122)
 b.atom('head/linear1',head.id,'信号头隐藏映射','linear','512 → 512','Linear(dim,dim)。','',123,'linear')
 b.atom('head/gelu',head.id,'信号头 GELU','gelu','B × 512','nn.GELU()。','',124,'activation')
 b.atom('head/linear2',head.id,'60 方向信号修正','linear','512 → 60','reshape 成 B×60×1×1。','',125,'linear');b.chain(head.id)
 b.atom('center','root','原始中心体素信号','tensor','B × 60 × 1 × 1','real_A[:,:,1,1]，与首 token 读取是不同操作。',R`s_c=X[:,:,1,1]`,103,'tensor','test.py').position=[0,50,8]
 b.atom('correct','root','与中心信号相加','add','B × 60 × 1 × 1','ViT 输出修正与原中心信号逐元素相加。',R`s=t+s_c`,103,'sum','test.py').position=[0,-13,0]
 const sparse=b.folder('sparse','root','共享稀疏字典塔','60 → 601','将方向信号映射到稀疏系数，并重复应用同一字典算子。','先 W_layer:Conv1×1(60,601)；Dictionary_Block=Threshold(.001,inplace=True)→Conv1×1(601,601)。首次调用+8次循环，总9次；所有更新共享 dcblock1。',R`a=T_{.001}(W_s s+b_s),\quad y_1=a+D(a),\quad y_{k+1}=a+D(T_{.001}(y_k))`,49,'net.py')
 b.atom('sparse/project',sparse.id,'初始字典系数','conv','60 → 601','kernel_size=1 stride=1 padding=0 bias=True；W_layer。',R`x_1=W_s s+b_s`,36,'conv','net.py')
 for(let i=0;i<9;i++){
  const id=`sparse/iteration${i}`,e=b.folder(id,sparse.id,`共享字典更新 ${i+1}`,'B × 601 × 1 × 1','选择有效字典原子，并注入同一初始系数。',i===0?'首次 Threshold 为 inplace，会把 x1 本身也置零；此后的加法基底是已阈值化的 a。':'与首次调用共享同一 Conv 权重；固定 a 沿外围旁路注入。',i===0?R`y_1=T(x_1)+D(T(x_1))`:R`y_{k+1}=a+D(T(y_k))`,i===0?76:82,'net.py')
  b.atom(id+'/threshold',id,'0.001 硬阈值','threshold','601 coefficients','nn.Threshold(.001,0,inplace=True)；不是软阈值。','',21,'activation','net.py')
  b.atom(id+'/dictionary',id,'共享字典线性变换','conv','601 → 601','kernel_size=1 stride=1 padding=0 bias=True。九个调用使用同一权重，不是九套字典。','',23,'conv','net.py')
  b.atom(id+'/add',id,'注入固定初始系数','add','601 coefficients','左侧固定 a 与当前字典输出相加。',e.formula,82,'sum','net.py');b.chain(id);b.link(sparse.id,'sparse/project',id+'/add','固定 a = T(x₁)',true)
 }
 b.atom('sparse/final',sparse.id,'最终非负稀疏筛选','threshold','601 coefficients','self.activ 的 Threshold(.001,0,inplace=True)。','',84,'activation','net.py');b.chain(sparse.id);b.vertical(sparse.id,15,52,5)
 const split=b.folder('fractions','root','分离水与组织系数','601 → 1 + 600','最后一个坐标给出 Viso；其余组成组织字典权重。','Viso=same_B[:,-1]；Vf=same_B[:,:-1]；Vf 加1e-10后 L1归一化。',R`V_{iso}=y_{601},\quad f_i=\frac{y_i+10^{-10}}{\sum_{j=1}^{600}|y_j+10^{-10}|}`,109,'test.py');split.position=[15,-11,0]
 b.atom('fractions/split',split.id,'最后一维与前600维分路','split','601 → 600 + 1','并非均分 chunk；最后一维独立读取。',R`(v,V_{iso})=(y[:,:600],y[:,600])`,109,'split','test.py').math!.operation='lab:split:unequal'
 b.atom('fractions/norm',split.id,'组织系数 L1 归一化','l1norm','600 coefficients','仅组织分支归一化；Viso 不经过 Mapping。','',113,'norm','test.py');b.chain(split.id)
 const map=b.folder('mapping','root','两条独立参数映射','600 → Vic, ODI','从非负组织系数推断胞内体积分数和方向离散程度。','两个独立 Threshold(.0001)→Conv1×1(600,1)；输出直接拼接，源码无 sigmoid/clamp。','',91,'net.py');map.position=[15,-23,0]
 for(const name of ['vic','odi']){const e=b.folder(`mapping/${name}`,map.id,name==='vic'?'胞内体积分数 Vic':'方向离散指数 ODI','600 → 1','学习字典系数到单一微结构指标的映射。','两个分支参数不共享。','',96,'net.py');b.atom(e.id+'/threshold',e.id,'0.0001 硬阈值','threshold','600 coefficients','Mapping 使用更小阈值。','',97,'activation','net.py').math!.operation='core:threshold:0.0001';b.atom(e.id+'/conv',e.id,'单参数线性读出','conv','600 → 1','kernel_size=1 stride=1 padding=0 bias=True。','',99,'conv','net.py');b.chain(e.id)}
 b.atom('output','root','Vic · Viso · ODI 参数图','tensor','B × 3 → volume × 3','按 [Vic,Viso,ODI] 拼接；reshape(order=F)，乘脑 mask 并沿用 NIfTI affine。',R`P=[V_{ic},V_{iso},ODI]`,125,'image','test.py').position=[0,-35,0]
 b.link('root','input',embed.id);b.link('root','input','center');b.link('root',embed.id,enc.id);b.link('root',enc.id,head.id);b.link('root',head.id,'correct');b.link('root','center','correct','中心信号旁路',true);b.link('root','correct',sparse.id);b.link('root',sparse.id,split.id);b.link('root',split.id,map.id,'组织系数');b.link('root',split.id,'output','Viso 独立输出',true);b.link('root',map.id,'output','Vic / ODI')
 const o=b.overview()
 return {id:'metsc',name:'METSC',repo:'https://github.com/Tianshu996/METSC',commit:'414f7d4cb307592b47a534b49b92b9d1388c832e',entries:b.entries,overview:o.keys,overviewEdges:o.edges,intro:'空间注意力预测信号修正，共享字典迭代产生稀疏系数，再分解为三种 NODDI 参数。',config:'test.py · 60×3×3 输入 · ViT 512 / 6 layers / 8 heads · 601 字典系数 · 9 次共享更新',equations:[{title:'公开 Transformer 的实际残差',formula:R`y=x+A(\mathrm{LN}_1x),\quad x'=y+F(\mathrm{LN}_2y)+x`,explanation:'ff 已被 Residual 包装；forward 中再加 x1，因此不能简化成普通两残差 Transformer。'},{title:'稀疏更新与原地阈值',formula:R`a=T_{.001}(W_s s+b_s),\quad y_1=a+D(a),\quad y_{k+1}=a+D(T_{.001}y_k),\quad k=1,\ldots,8`,explanation:'首次 dcblock1 原地阈值化 x1，改变后续使用的固定加法基底。D 是含偏置的共享 1×1 Conv。'},{title:'从字典系数到参数图',formula:R`V_{iso}=y_{601},\quad f_i=\frac{y_i+10^{-10}}{\sum_{j=1}^{600}(y_j+10^{-10})},\quad (V_{ic},ODI)=M(f)`,explanation:'最终 Threshold 保证非负系数。两个 Mapping 分支各自阈值化并线性读出，源码未加入取值范围限制。'}],training:[{title:'论文设计',detail:'Transformer 学习空间关系；稀疏字典结构引入模型归纳偏置。此可视化展示公开 NODDI demo。'},{title:'作者推理配置',detail:'HCP-YA 双壳层，每壳30方向；加载 ViT、SparseReconstruction、Mapping 三份检查点，全部 eval。'},{title:'组织参数输出',detail:'生成 Vic、Viso、ODI；保留 mask 和空间 affine。不以动画数值代替真实评估结果。'}],audit:[{title:'共享字典不是九套参数',detail:'dcblock1 首次调用一次，for range(8) 再调用八次，权重完全共享。Threshold 的 inplace=True 会改变第一次的 x1。',file:'net.py',line:72},{title:'首 token 不是 CLS',detail:'无独立 cls token，直接取 x[:,0]。中心体素 real_A[:,:,1,1] 通过外侧信号旁路加入。',file:'vit_pytorch.py',line:135},{title:'额外残差真实存在',detail:'Transformer.forward 中 ff 是 Residual(PreNorm(...))，然后 ff(x)+x1 又加一次原层输入。',file:'vit_pytorch.py',line:95},{title:'公开代码范围',detail:'仓库提供 NODDI 推理框架和预训练模型链接，未包含完整训练流程；页面不据此宣称所有论文任务均已复现。',file:'README.md',line:5}]}
}
