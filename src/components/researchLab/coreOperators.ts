import type {MathSpec,MathStage} from '../mmhvae/mathematics'
import {demoValues,normalizeValues} from '../mmhvae/mathematics'
import type {NumericExample} from './operators'
const R=String.raw
export type CoreOp='layernorm'|'gelu'|'scores'|'softmax'|'weighted'|'threshold'|'l1norm'|'dropout'|'vq'|'noise'|'ddim'|'tensor-signal'|'angular'|'log-signal'|'rotation'
const data:Record<CoreOp,[string,string,string][]>={
 layernorm:[['读取单个 token',R`x\in\mathbb R^d`,'统计轴是 token 的 embedding 维度，不是空间 H×W。'],['中心化与缩放',R`\hat x_i=(x_i-\mu)/\sqrt{v+10^{-5}}`,'每个 token 独立计算均值和总体方差。'],['学习仿射参数',R`y_i=\gamma_i\hat x_i+\beta_i`,'逐 embedding 坐标具有独立可学习缩放与偏置。']],
 gelu:[['读取输入坐标',R`x_i`,'逐元素非线性，维度保持。'],['正态累积分布门控',R`\operatorname{GELU}(x)=x\Phi(x)`,'曲线平滑保留小负值；不是 ReLU 的硬截断。'],['写回输出',R`y_i=x_i\Phi(x_i)`,'示例采用 erf 数值近似计算标准 GELU。']],
 scores:[['查询与键矩阵',R`Q,K\in\mathbb R^{N\times d_h}`,'水平网格为 token × 特征，两个输入独立投影而来。'],['计算两两点积',R`S_{ij}=\frac{\sum_rQ_{ir}K_{jr}}{\sqrt{d_h}}`,'每个分数连接一个 query 和一个 key；观察位置选中对应的行列。'],['形成关系矩阵',R`S\in\mathbb R^{N\times N}`,'分数尚未归一化；后续 softmax 沿 key 轴进行。']],
 softmax:[['逐行读取 logits',R`s_j=S_{ij}`,'每个 query 的整行 key 共同参与归一化。'],['稳定指数映射',R`u_j=\exp(s_j-\max_k s_k)`,'减去行最大值保持结果不变，并避免指数溢出。'],['行和归一化',R`a_j=u_j/\sum_ku_k,\quad\sum_ja_j=1`,'当前行构成非负权重；这不是独立 sigmoid。']],
 weighted:[['权重与值向量',R`A\in\mathbb R^{N\times N},\quad V\in\mathbb R^{N\times d_h}`,'权重已按行归一化。'],['按 key 汇聚',R`O_{ir}=\sum_j A_{ij}V_{jr}`,'当前输出元素连接权重行及 V 的对应特征列。'],['组织每个 head 输出',R`O\in\mathbb R^{N\times d_h}`,'多个 head 随后拼接并线性投影。']],
 threshold:[['读取稀疏系数',R`z_i\in\mathbb R`,'METSC 使用 nn.Threshold，负数和低于阈值的值均置零。'],['硬阈值筛选',R`T_\lambda(z)=\begin{cases}z&z>\lambda\\0&z\le\lambda\end{cases}`,'不是 soft threshold；保留下来的值不减去 λ。'],['保留活跃原子',R`\operatorname{supp}(T_\lambda z)=\{i:z_i>\lambda\}`,'主稀疏块 λ=0.001；Mapping 分支 λ=0.0001。']],
 l1norm:[['读取非负系数',R`v\in\mathbb R^{600}_{\ge0}`,'最后一个 601 维坐标已单独取出作为 Viso。'],['微小偏置避免零分母',R`a_i=v_i+10^{-10}`,'仅剩余 600 个坐标进入该归一化。'],['沿字典坐标归一化',R`f_i=a_i/\|a\|_1`,'归一化为非负总和一，随后进入两个独立参数映射。']],
 dropout:[['读取完整信号',R`X`,'SSDiff 训练对信号元素应用 dropout，默认丢弃率 0.75。'],['独立 Bernoulli 保留',R`M_i\sim\operatorname{Bernoulli}(1-p)`,'遮挡发生在输入值中，没有从注意力序列移除 token。'],['倒置 dropout 缩放',R`\tilde X=M\odot X/(1-p)`,'源码使用 F.dropout；保留值放大 4 倍，推理不执行此输入遮挡。']],
 vq:[['连续潜向量与码本',R`z_e\in\mathbb R^k,\quad e_j\in\mathbb R^k`,'PIGMENT 论文码本包含 16,000 个 embedding。小型例只画四个候选。'],['选择最近码字',R`j^*=\arg\min_j\|z_e-e_j\|_2^2`,'距离在 latent 特征维计算。'],['替换为离散表示',R`z_q=e_{j^*}`,'码本损失与 commitment 损失分开控制梯度；β=0.25。']],
 noise:[['干净潜变量与噪声',R`z_0,\quad\epsilon\sim\mathcal N(0,I)`,'固定小型噪声示例用于解释前向扩散。'],['按噪声日程混合',R`z_t=\sqrt{\bar\alpha_t}z_0+\sqrt{1-\bar\alpha_t}\epsilon`,'滑杆控制信号保留比例；不是改变模型训练参数。'],['得到指定时间潜变量',R`q(z_t|z_0)=\mathcal N(\sqrt{\bar\alpha_t}z_0,(1-\bar\alpha_t)I)`,'去噪网络读取 z_t、t 与 S0 条件。']],
 ddim:[['当前潜变量与预测噪声',R`z_t,\quad\hat\epsilon=\epsilon_\theta(z_t,t,S_0)`,'网络权重固定，示例噪声预测仅用于计算演示。'],['估计无噪声潜变量',R`\hat z_0=(z_t-\sqrt{1-\bar\alpha_t}\hat\epsilon)/\sqrt{\bar\alpha_t}`,'先去掉噪声再除以信号尺度。'],['沿 DDIM 日程更新',R`z'_{t-1}=\sqrt{\bar\alpha_{t-1}}\hat z_0+\sqrt{1-\bar\alpha_{t-1}-\delta_t^2}\hat\epsilon+\delta_t\epsilon`,'例中 δ=0，展示确定性分支；论文保留随机项。']],
 'tensor-signal':[['对称扩散张量',R`D=D^T`,'一个方向 g 查询三维扩散椭球，而非注意力矩阵。'],['方向上的表观扩散',R`D_{app}=g^TDg`,'滑杆旋转单位方向；示例 D 为固定正定对角矩阵。'],['预测扩散信号',R`S(b,g)=S_0\exp(-b\,g^TDg)`,'这是 DTI 前向模型；DKI 和 NODDI 的扩展另外列出，不混用参数。']],
 angular:[['两条单位方向',R`u=x/\|x\|_2,\quad v=\hat x/\|\hat x\|_2`,'扩散方向的正负具有轴向等价性。'],['平方方向相似度',R`a=(u^Tv)^2`,'平行和反平行均得到相似度 1。'],['角度损失',R`\mathcal L_{angular}=1-(u^Tv)^2`,'相比逐分量 L1，该项尊重方向几何。']],
 'log-signal':[['基线与扩散信号',R`S_0=\operatorname{mean}_{b<50}S_b`,'基线沿 b≈0 的方向平均。'],['相对信号取对数',R`X=\log|S/S_0|`,'SSDiff 数据加载器先取幅值比再取自然对数。'],['清理无效坐标',R`X_{NaN,Inf}\leftarrow0`,'仅保留 b≥50 的扩散观测；bvals 同步筛选。']],
 rotation:[['局部形变矩阵',R`J=(I+\nabla u)^{-1}`,'SSDiff 由预处理保存的形变导数恢复局部映射。'],['取旋转部分',R`R=(JJ^T)^{-1/2}J`,'通过对称特征分解求矩阵平方根；示例展示纯旋转特例。'],['重定向梯度',R`g'=gR/\|gR\|_2`,'每个体素都得到自身的 b-vector，后续送入方向嵌入。']],
}
export function coreSpec(op:CoreOp,formula=''):MathSpec{return {kind:'tensor',operation:`core:${op}`,steps:data[op].map(([title,formula,explanation])=>({title,formula,explanation})),control:op==='noise'?'信号保留比例':op==='angular'||op==='tensor-signal'||op==='rotation'?'观察方向':'观察元素位置',legend:'小型确定性数值演算；水平行列保留矩阵索引，数据元素为等尺寸水晶块。',formula:formula||data[op][1][1],source:op,kernel:3,stride:1,padding:1,weightNorm:false,bias:true,reflect:false}}
export const softmaxRows=(x:number[],n=4)=>x.map((v,i)=>{const row=x.slice(Math.floor(i/n)*n,Math.floor(i/n)*n+n),max=Math.max(...row);return Math.exp(v-max)/row.reduce((s,v)=>s+Math.exp(v-max),0)})
export function coreExample(spec:MathStage,p:number):NumericExample{
 const op=spec.operation.slice(5).split(':')[0],a=demoValues.slice(0,16),b=demoValues.slice(16,32);let out=[...a],index=Math.min(15,Math.floor(p*16)),note=''
 if(op==='layernorm')out=Array.from({length:4},(_,i)=>normalizeValues(a.slice(i*4,i*4+4))).flat()
 if(op==='gelu'){const erf=(x:number)=>{const sign=x<0?-1:1,t=1/(1+.3275911*Math.abs(x));return sign*(1-(((((1.061405429*t-1.453152027)*t)+1.421413741)*t-.284496736)*t+.254829592)*t*Math.exp(-x*x))};out=a.map(x=>x*.5*(1+erf(x/Math.SQRT2)))}
 if(op==='scores')out=a.map((_,i)=>Array.from({length:4},(_,k)=>a[Math.floor(i/4)*4+k]*b[(i%4)*4+k]).reduce((s,v)=>s+v,0)/2)
 if(op==='softmax')out=softmaxRows(a)
 if(op==='weighted'){a.splice(0,16,...softmaxRows(a));out=a.map((_,i)=>Array.from({length:4},(_,k)=>a[Math.floor(i/4)*4+k]*b[k*4+i%4]).reduce((s,v)=>s+v,0))}
 if(op==='threshold'){const threshold=Number(spec.operation.split(':')[2]??.001);out=a.map(x=>x>threshold?x:0);note=`λ=${threshold}`}
 if(op==='l1norm'){a.splice(0,16,...a.map(x=>Math.max(0,x)));const sum=a.reduce((s,x)=>s+x+1e-10,0);out=a.map(x=>(x+1e-10)/sum);note='非负系数归一化'}
 if(op==='dropout'){const prob=Number(spec.operation.split(':')[2]??.75);b.splice(0,16,...b.map((_,i)=>prob===0?1:((i*7)%16+.5)/16<prob?0:1));out=a.map((x,i)=>b[i]*x/(1-prob));note=`p=${prob} · 固定训练掩码示例；eval为恒等`}
 if(op==='vq'){const target=[p*2-1,.25,-.3],codes=[[-.8,.1,-.2],[-.2,.4,-.5],[.4,0,.1],[.9,.5,-.4]],dist=codes.map(c=>c.reduce((s,x,i)=>s+(x-target[i])**2,0)),j=dist.indexOf(Math.min(...dist));return {a:target,b:codes.flat(),out:codes[j],index:0,readout:`最近码字 ${j} · 平方距离 ${dist[j].toFixed(4)} · 教学码本 4×3`}}
 if(op==='noise'){const alpha=.05+p*.9;out=a.map((x,i)=>Math.sqrt(alpha)*x+Math.sqrt(1-alpha)*b[i]);note=`ᾱ=${alpha.toFixed(2)}`}
 if(op==='ddim'){const alpha=.5,prev=.6,z0=a.map((x,i)=>(x-Math.sqrt(1-alpha)*b[i])/Math.sqrt(alpha));out=spec.stage===1?z0:z0.map((x,i)=>Math.sqrt(prev)*x+Math.sqrt(1-prev)*b[i]);note='ᾱt=0.5 · ᾱt−1=0.6 · δ=0'}
 if(op==='tensor-signal'){const angle=p*Math.PI,g=[Math.cos(angle),Math.sin(angle),0],D=[.0017,0,0,0,.0004,0,0,0,.0004],d=g[0]**2*D[0]+g[1]**2*D[4];return {a:D,b:g,out:[Math.exp(-1000*d)],index:0,readout:`b=1000 · gᵀDg=${d.toFixed(6)} · S/S₀=${Math.exp(-1000*d).toFixed(4)}`}}
 if(op==='angular'){const t=p*Math.PI;return {a:[1,0,0],b:[Math.cos(t),Math.sin(t),0],out:[Math.sin(t)**2],index:0,readout:`夹角 ${(p*180).toFixed(0)}° · 损失 ${(Math.sin(t)**2).toFixed(4)}`}}
 if(op==='log-signal'){a.splice(0,16,...a.map(x=>Math.abs(x)+.01));out=a.map(x=>Math.log(x/1.5));note='S₀=1.5'}
 if(op==='rotation'){const t=p*Math.PI;return {a:[1,0,0],b:[Math.cos(t),-Math.sin(t),0,Math.sin(t),Math.cos(t),0,0,0,1],out:[Math.cos(t),-Math.sin(t),0],index:0,readout:`局部纯旋转示例 · 角度 ${(p*180).toFixed(0)}° · ‖g′‖₂=1`}}
 if(op==='mse'){out=a.map((x,i)=>(x-b[i])**2);if(spec.stage===2){const loss=out.reduce((s,x)=>s+x,0)/out.length;return {a,b,out:[loss],index:0,readout:`MSE=${loss.toFixed(4)} · 全部16个位置`}}}
 if(op==='softmax')note=`所选行总和=${out.slice(Math.floor(index/4)*4,Math.floor(index/4)*4+4).reduce((s,x)=>s+x,0).toFixed(3)}`
 return {a,b,out,index,readout:`[${Math.floor(index/4)},${index%4}] · x=${a[index].toFixed(3)} → y=${out[index].toFixed(3)}${note?' · '+note:''}`}
}
