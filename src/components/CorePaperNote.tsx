import {useMemo} from 'react'
import {Github,ArrowUpRight} from 'lucide-react'
import {EditableText} from './EditableContent'
import {SourceLink} from './Primitives'
import ResearchExplorer from './researchLab/ResearchExplorer'
import {ssdiffModel} from './researchLab/ssdiffModel'
import {metscModel} from './researchLab/metscModel'
import {pigmentModel} from './researchLab/pigmentModel'
import {paperById} from '../papers'
type CoreId='ssdiff'|'metsc'|'pigment'
const COPY:Record<CoreId,{summary:string;purpose:string;method:string;data:string;metrics:string;result:string}>={
 ssdiff:{summary:'SSDiff 从原始扩散 MRI 信号进行自监督学习，以信号重建作为预训练任务，获得紧凑的个体表征。它把一个扩散方向对应的整个三维图像视为 token，并纳入 b-value、经过配准修正的 b-vector 与空间坐标信息，学习不同方向与空间位置之间的关系。',purpose:'论文中的“assumption-free”指学习表征不需要先指定 DTI、NODDI 等微结构拟合标签，并不表示预处理、网络或采集几何没有假设。SSDiff 的自监督 Transformer 与生成式扩散模型属于不同机制。',method:'先把扩散数据配准到固定 MNI 体素空间，对信号做 log(|S/S0|) 变换，并调整局部梯度方向。训练通过输入 dropout 隐去部分信号，模型预测完整信号；推理则在中间 Transformer 后取真实方向 token 的均值，形成 1536 维受试者级表征。',data:'多数据集扩散 MRI，经 MNI 空间对齐和梯度重定向。输入方向数可以改变，但体素对应由共同脑掩码确定。',metrics:'信号重建 MSE；论文进一步通过表型预测、疾病相关分析、空间解释与遗传分析评估表示。',result:'论文报告表征可反映传统微结构指标以外的信息。本站核验的是作者预训练与特征提取脚本，不把教学动画当成独立复现实验。'},
 metsc:{summary:'METSC 将 Transformer 的空间特征提取与稀疏表示相结合，用于扩散 MRI 微结构参数估计。稀疏编码为网络引入模型归纳偏置，使数据驱动的空间关系学习与有结构的参数推断共同工作。作者公开仓库提供 NODDI 推理示例。',purpose:'与只用一个回归头预测参数不同，该实现先学习中心扩散信号的修正，再通过重复的共享字典更新得到 601 维系数，最后分离自由水与组织成分。',method:'公开 demo 使用 60 个方向的 3×3 邻域。六层 ViT 读取空间 token，预测 60 维信号修正并与真实中心信号相加。稀疏模块执行九次共享字典更新，之后取最后一维作为 Viso，前 600 维经 L1 归一化与两个独立映射得到 Vic 和 ODI。',data:'作者 demo 采用 HCP-YA，双壳层各 30 个方向，b-value 为 1 和 2 ms/μm²；输入局部窗口 3×3。',metrics:'关注 NODDI 指标 Vic、Viso、ODI 的参数估计误差及图像结构；论文比较采样方案与模型设计。',result:'公开代码可定位到每个 ViT block、共享字典和输出映射。完整训练流程不在该 demo 中，页面保留论文与代码范围差异。'},
 pigment:{summary:'PIGMENT 学习脑微结构参数的生成先验，并在推理时通过受试者真实扩散信号进行物理约束。论文采用三维向量量化自编码器压缩参数体，再训练以 S0 为条件的潜空间去噪模型，并以数据一致性与流形一致性共同优化个体估计。',purpose:'生成先验提供统计上合理的空间结构，前向微结构模型使结果匹配真实采集。DTI、DKI 与 NODDI 的参数空间不同，论文目前分别训练相应框架。',method:'离线阶段先学习 VQ 自编码器，再训练 1000 时间步的条件去噪网络。推理从高斯潜噪声出发运行 500 个 DDIM 步；第 480 与 500 步引入双域优化。优化后的参数重新编码、加噪并与原采样轨迹混合，兼顾生成先验和个体信号。',data:'论文训练队列合计 11,375 次扫描，覆盖多个数据集、设备厂商与场强；评估包含外部采集协议与稀疏方向设置。',metrics:'DTI 与 NODDI 指标使用平均绝对误差；部分 kurtosis 指标采用中位绝对误差，并分析结构连接与下游用途。',result:'当前为 2026-05-29 的 arXiv v1 预印本。作者声明代码与数据将在稿件接收后公开；本站目前展示论文可确认的结构，不声称已完成源码级复现。'},
}
export default function CorePaperNote({id}:{id:CoreId}){
 const paper=paperById(id)!,copy=COPY[id],model=useMemo(()=>id==='ssdiff'?ssdiffModel():id==='metsc'?metscModel():pigmentModel(),[id])
 const heading=(index:string,title:string,caption:string)=><header className="note-block-heading"><span>{index}</span><div><h3>{title}</h3><EditableText textKey={`${id}-caption-${index}`}>{caption}</EditableText></div></header>
 return <article className={`paper-note paper-note--${id}`} id={`panel-${id}`} aria-label={paper.title}>
  <header className="paper-note-title"><div><h2>{paper.title}</h2></div><dl><div><dt>Authors</dt><dd>{paper.authors}</dd></div><div><dt>Venue</dt><dd>{paper.venue}</dd></div><div><dt>栏目</dt><dd>课题核心 / {paper.short}</dd></div></dl></header>
  <div className="archive-jump"><a href={`#${id}-orbit-lab`}>进入三维架构实验台 <ArrowUpRight size={13}/></a><a href={paper.paperUrl} target="_blank" rel="noreferrer">阅读论文 <ArrowUpRight size={13}/></a></div>
  {id==='pigment'&&<EditableText textKey="pigment-source-status" className="core-source-status">论文架构复原 · 截至 2026-09-12 未确认作者公开源码。已披露的层次与未披露的算子参数在目录中分别说明，链接定位到论文页码。</EditableText>}
  <section className="note-block abstract-note">{heading('01','摘要与方法定位','论文内容概述，以及它在课题中的作用')}<div className="abstract-grid"><div className="translation-copy"><h4>内容概述</h4><EditableText textKey={`${id}-summary`}>{copy.summary}</EditableText></div><aside className="abstract-comment"><h4>理解重点</h4><EditableText textKey={`${id}-purpose`}>{copy.purpose}</EditableText></aside></div></section>
  <section className="note-block method-note">{heading('02','Method','从总体结构、逐层模块到原子数学演算')}<div className="method-preface"><EditableText textKey={`${id}-method`}>{copy.method}</EditableText></div><ResearchExplorer key={id} model={model}/></section>
  <section className="note-block evaluation-note">{heading('03','评估与证据边界','数据、指标与可验证的实现范围')}<div className="evaluation-grid">{[['DATA','数据与采集',copy.data],['METRICS','评价指标',copy.metrics],['EVIDENCE','结果与实现范围',copy.result]].map(([tag,title,body],i)=><div key={tag}><span>{tag}</span><EditableText as="strong" textKey={`${id}-evaluation-title-${i}`}>{title}</EditableText><EditableText textKey={`${id}-evaluation-${i}`}>{body}</EditableText></div>)}</div></section>
  <div className="source-row"><SourceLink href={paper.paperUrl}>论文原文</SourceLink>{paper.repo?<SourceLink href={paper.repo}><Github size={14}/>作者代码</SourceLink>:<SourceLink href="https://arxiv.org/pdf/2606.00156#page=36">代码公开声明 · 第36页</SourceLink>}</div>
 </article>
}
