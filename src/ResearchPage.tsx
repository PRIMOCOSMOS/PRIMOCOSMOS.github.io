import { lazy, Suspense, useEffect, type KeyboardEvent } from 'react'
import { Github, ScanLine } from 'lucide-react'
import { MMVAEPlusExplorer } from './components/MMVAEPlusExplorer'
import { PnPCosmoExplorer } from './components/PnPCosmoExplorer'
import { SourceLink } from './components/Primitives'
import { EditableText, NoteEditToolbar } from './components/EditableContent'

const MMHVAEExplorer = lazy(() => import('./components/MMHVAEExplorer').then(module => ({ default: module.MMHVAEExplorer })))

type PaperId = 'mmhvae' | 'pnp' | 'mmvae'

const PAPERS: Array<{ id: PaperId; code: string; short: string; title: string; venue: string }> = [
  {
    id: 'mmhvae',
    code: '01 / H-MOPE',
    short: 'MMHVAE',
    title: 'Unified Cross-Modal Medical Image Synthesis with Hierarchical Mixture of Product-of-Experts',
    venue: 'IEEE TPAMI · Vol. 48(2)',
  },
  {
    id: 'pnp',
    code: '02 / C-S PNP',
    short: 'PnP-CoSMo',
    title: 'A Plug-and-Play Method for Guided Multi-contrast MRI Reconstruction Based on Content/Style Modeling',
    venue: 'Medical Image Analysis · 2026',
  },
  {
    id: 'mmvae',
    code: '03 / S-P VAE',
    short: 'MMVAE++',
    title: 'Disentangling Shared and Private Latent Factors in Multimodal Variational Autoencoders',
    venue: 'PMLR 240 · 2024',
  },
]

function BrandMark() {
  return (
    <svg className="brand-mark" viewBox="0 0 42 42" aria-hidden="true">
      <circle cx="21" cy="21" r="17" />
      <ellipse cx="21" cy="21" rx="17" ry="6.5" transform="rotate(-28 21 21)" />
      <circle cx="21" cy="21" r="3.4" />
      <path d="M6 34L13 27M29 15L36 8" />
    </svg>
  )
}

function NoteHeader({ index, title, caption, captionKey }: { index: string; title: string; caption: string; captionKey: string }) {
  return (
    <header className="note-block-heading">
      <span>{index}</span>
      <div>
        <h3>{title}</h3>
        <EditableText textKey={captionKey}>{caption}</EditableText>
      </div>
    </header>
  )
}

function MMHVAENote() {
  return (
    <article className="paper-note paper-note--mmhvae" id="panel-mmhvae" role="tabpanel" aria-labelledby="tab-mmhvae">
      <header className="paper-note-title">
        <div>
          <span className="paper-kicker">PAPER 01 · MISSING-MODALITY SYNTHESIS</span>
          <h2>Unified Cross-Modal Medical Image Synthesis with Hierarchical Mixture of Product-of-Experts</h2>
        </div>
        <dl>
          <div><dt>Authors</dt><dd>Dorent et al.</dd></div>
          <div><dt>Venue</dt><dd>IEEE TPAMI · Vol. 48(2)</dd></div>
          <div><dt>Model</dt><dd>MMHVAE</dd></div>
        </dl>
      </header>

      <section className="note-block abstract-note">
        <NoteHeader index="01" title="摘要" caption="原文摘要的中文翻译与方法定位" captionKey="mmhvae-abstract-caption" />
        <div className="abstract-grid">
          <div className="translation-copy">
            <h4>摘要翻译</h4>
            <EditableText textKey="mmhvae-abstract-translation">本文提出一种名为 MMHVAE 的深度多模态层次变分自编码器混合模型，根据不同模态中的观测图像合成缺失图像。模型设计围绕四项任务展开：构建能够生成高分辨率图像的复杂多模态潜表示；促使变分分布估计跨模态图像合成所需的缺失信息；在数据缺失条件下学习多模态信息融合；利用数据集层面的信息处理训练阶段的不完整数据集。作者在术前多参数脑磁共振与术中超声的跨模态合成任务上进行了广泛实验。</EditableText>
          </div>
          <aside className="abstract-comment">
            <h4>简要讲解</h4>
            <EditableText textKey="mmhvae-abstract-comment">方法由七层潜变量、观测子集混合后验、各层乘积专家融合和模态判别器共同构成。单个模型可接收 iUS、T2、ceT1、FLAIR 的任意非空观测组合。</EditableText>
          </aside>
        </div>
      </section>

      <section className="note-block method-note">
        <NoteHeader index="02" title="Method" caption="层次生成模型、缺失子集推断、逐层 PoE 与完整训练目标" captionKey="mmhvae-method-caption" />
        <div className="method-preface">
          <EditableText textKey="mmhvae-method-1">完整多模态图像记为 X=(X₁,…,Xₘ)，缺失指示向量 R 标记当前可见模态。潜变量 Z 被分为 Z₁,…,Z₇，从 1×1×256 的全局描述逐步展开到 192×192×8 的像素级描述。每个模态只配置一套单模态编码器；观测组合在每一层通过高斯乘积专家合并。</EditableText>
          <EditableText textKey="mmhvae-method-2">训练时从真实缺失模式 r 中采样子模式 r′，使用同一潜表示重建输入模态并合成未送入编码器的已知模态。KL 项约束层次后验，四个模态判别器利用数据集级样本约束生成分布。</EditableText>
        </div>
        <Suspense fallback={<p role="status">正在加载 MMHVAE 架构实验台…</p>}><MMHVAEExplorer /></Suspense>
      </section>

      <section className="note-block evaluation-note">
        <NoteHeader index="03" title="评估与性能" caption="数据集、指标和论文报告的主要结果" captionKey="mmhvae-evaluation-caption" />
        <div className="evaluation-grid">
          <div><span>DATA</span><EditableText as="strong" textKey="mmhvae-data-title">ReMIND · UPenn-GBM · RESECT-SEG</EditableText><EditableText textKey="mmhvae-data-copy">ReMIND 含 104 名患者，只有 13 名具有完整 MRI 序列；评估同时覆盖 MRI 与 iUS。</EditableText></div>
          <div><span>METRICS</span><EditableText as="strong" textKey="mmhvae-metrics-title">PSNR · SSIM · LPIPS</EditableText><EditableText textKey="mmhvae-metrics-copy">附加脑肿瘤分割 DSC 与 iUS-MR 配准误差，用下游任务检查结构信息。</EditableText></div>
          <div><span>RESULT · ReMIND</span><EditableText as="strong" textKey="mmhvae-result-title">L=7 · T2 28.6 dB · ceT1 29.8 dB</EditableText><EditableText textKey="mmhvae-result-copy">七层消融对所有输入组合取平均；下游 iUS 分割在 RESECT-SEG 与 ReMIND 上的 Dice 分别为 73.6% 与 77.6%。模型含 14M 参数、13G MACs。</EditableText></div>
        </div>
      </section>

      <div className="source-row">
        <SourceLink href="https://doi.org/10.1109/TPAMI.2025.3616632">DOI / TPAMI</SourceLink>
        <SourceLink href="https://pmc.ncbi.nlm.nih.gov/articles/PMC13092166/">开放全文</SourceLink>
        <SourceLink href="https://github.com/ReubenDo/MMHVAE"><Github size={14} /> 作者代码</SourceLink>
      </div>
    </article>
  )
}

function PnPNote() {
  return (
    <article className="paper-note paper-note--pnp" id="panel-pnp" role="tabpanel" aria-labelledby="tab-pnp">
      <header className="paper-note-title">
        <div>
          <span className="paper-kicker">PAPER 02 · GUIDED MRI RECONSTRUCTION</span>
          <h2>A Plug-and-Play Method for Guided Multi-contrast MRI Reconstruction Based on Content/Style Modeling</h2>
        </div>
        <dl>
          <div><dt>Authors</dt><dd>Rao et al.</dd></div>
          <div><dt>Venue</dt><dd>Medical Image Analysis · 2026</dd></div>
          <div><dt>Model</dt><dd>PnP-CoSMo</dd></div>
        </dl>
      </header>

      <section className="note-block abstract-note">
        <NoteHeader index="01" title="摘要" caption="原文摘要的中文翻译与方法定位" captionKey="pnp-abstract-caption" />
        <div className="abstract-grid">
          <div className="translation-copy">
            <h4>摘要翻译</h4>
            <EditableText textKey="pnp-abstract-translation-1">同一解剖结构的不同对比加权 MR 图像包含冗余信息，因此一次扫描会话中已获得的对比可用于引导后续欠采样对比的重建。现有端到端引导重建方法通常需要由原始 k-space 与配准参考图像组成的大规模配对训练集。本文提出模块化的即插即用方法，仅依赖部分配对的图像域数据，无需 k-space 训练数据。</EditableText>
            <EditableText textKey="pnp-abstract-translation-2">方法先从图像数据学习两种对比的 content/style 模型，再将该模型作为迭代重建中的即插即用算子。content 表示对比无关因素，style 表示对比特异因素。重建时用高质量参考扫描的 content 替换当前估计中的混叠 content，再依次执行 MR 数据一致性和 content 修正。作者将该方法命名为 PnP-CoSMo。其设计支持跨对比泛化，并基于两个对比的共享与非共享生成因素提供解释框架。</EditableText>
            <EditableText textKey="pnp-abstract-translation-3">作者通过仿真实验研究方法的可解释性与收敛性。NYU fastMRI DICOM 实验显示，其质量和泛化能力达到或超过端到端方法；两个院内多线圈数据集上的结果显示，在给定 SSIM 时，相对无引导即插即用重建可增加最高 32.6% 的加速。</EditableText>
          </div>
          <aside className="abstract-comment">
            <h4>简要讲解</h4>
            <EditableText textKey="pnp-abstract-comment">训练与逆问题求解分为两个阶段。离线阶段学习双向 content/style 分解；在线阶段把生成器嵌入 ISTA 循环，并使用真实测得的 k-space 修正图像与参考 content。</EditableText>
          </aside>
        </div>
      </section>

      <section className="note-block method-note">
        <NoteHeader index="02" title="Method" caption="MUNIT 预训练、配对微调、content consistency 与迭代重建" captionKey="pnp-method-caption" />
        <div className="method-preface">
          <EditableText textKey="pnp-method-1">两个图像域 X₁ 与 X₂ 共享 content 空间 C，并拥有独立 style 空间 S₁ 与 S₂。每个域包含 content encoder Eᶜ、style encoder Eˢ、decoder G 和 discriminator D。content encoder 使用输入卷积、可选的步长下采样卷积及残差块；style encoder 使用卷积、下采样、全局自适应平均池化和全连接层；decoder 反向恢复空间分辨率，并通过 AdaIN 把 style 注入 content 特征。</EditableText>
          <EditableText textKey="pnp-method-2">无配对 MUNIT 阶段优化 GAN、图像自重建、content 自重建和 style 自重建。少量配准图像进入 PFT 阶段，增加双向跨域图像损失与配对 content 对齐损失。重建阶段从 Aᴴy 初始化目标图像，从参考图像初始化 content，然后循环执行 content consistency、data consistency 和 content refinement。</EditableText>
        </div>
        <PnPCosmoExplorer />
      </section>

      <section className="note-block evaluation-note">
        <NoteHeader index="03" title="评估与性能" caption="仿真、公开数据、院内数据与临床阅片" captionKey="pnp-evaluation-caption" />
        <div className="evaluation-grid">
          <div><span>DATA</span><EditableText as="strong" textKey="pnp-data-title">BrainWeb · NYU fastMRI · LUMC</EditableText><EditableText textKey="pnp-data-copy">BrainWeb 用于超参数、收敛与鲁棒性分析；公开和院内数据覆盖不同对比、线圈与采样设置。</EditableText></div>
          <div><span>METRICS</span><EditableText as="strong" textKey="pnp-metrics-title">SSIM · HaarPSI · DISTS</EditableText><EditableText textKey="pnp-metrics-copy">院内实验另包含两名观察者对锐度、噪声、伪影和组织对比的定性评分。</EditableText></div>
          <div><span>RESULT</span><EditableText as="strong" textKey="pnp-result-title">最高 +32.6% 加速</EditableText><EditableText textKey="pnp-result-copy">在给定 SSIM 下相对 PnP-CNN 获得更高加速；无需 k-space 训练数据。单张 349×284 切片约需 17 秒。</EditableText></div>
        </div>
      </section>

      <div className="source-row">
        <SourceLink href="https://doi.org/10.1016/j.media.2026.104160">DOI / Medical Image Analysis</SourceLink>
        <SourceLink href="https://arxiv.org/abs/2409.13477">arXiv</SourceLink>
        <SourceLink href="https://github.com/cnmy-ro/pnp-cosmo"><Github size={14} /> 作者代码</SourceLink>
      </div>
    </article>
  )
}

function MMVAEPlusNote() {
  return (
    <article className="paper-note paper-note--mmvae" id="panel-mmvae" role="tabpanel" aria-labelledby="tab-mmvae">
      <header className="paper-note-title">
        <div>
          <span className="paper-kicker">PAPER 03 · SHARED / PRIVATE LATENTS</span>
          <h2>Disentangling Shared and Private Latent Factors in Multimodal Variational Autoencoders</h2>
        </div>
        <dl>
          <div><dt>Authors</dt><dd>Märtens & Yau</dd></div>
          <div><dt>Venue</dt><dd>PMLR 240 · 2024</dd></div>
          <div><dt>Model</dt><dd>MMVAE++</dd></div>
        </dl>
      </header>

      <section className="note-block abstract-note">
        <NoteHeader index="01" title="摘要" caption="原文摘要的中文翻译与方法定位" captionKey="mmvae-abstract-caption" />
        <div className="abstract-grid">
          <div className="translation-copy">
            <h4>摘要翻译</h4>
            <EditableText textKey="mmvae-abstract-translation-1">面向多模态数据的生成模型能够识别与观测数据异质性来源相关的潜在因素。共享因素可解释多个模态之间的共同变化，私有因素用于解释单个模态中的变化。MVAE、MMVAE 等多模态变分自编码器适合推断这些潜在因素并分离共享变化与私有变化。本文研究这些模型能否稳定完成这种解耦。</EditableText>
            <EditableText textKey="mmvae-abstract-translation-2">作者重点分析模态特异变化强于共享信号的困难场景，从跨模态预测角度展示现有模型的局限，并提出一项提高模型抗模态特异变化能力的修改。实验覆盖合成数据以及多个真实 multi-omics 数据集。</EditableText>
          </div>
          <aside className="abstract-comment">
            <h4>简要讲解</h4>
            <EditableText textKey="mmvae-abstract-comment">论文把各类多模态 VAE 的目标写成“输入模态集合 I 预测输出模态集合 M”的组合。MMVAE++ 保留同视图生成路径，同时禁止同视图损失更新共享潜变量。</EditableText>
          </aside>
        </div>
      </section>

      <section className="note-block method-note">
        <NoteHeader index="02" title="Method" caption="PoE、MoE、MoPoE 的跨视图目标与 MMVAE++ 梯度路径" captionKey="mmvae-method-caption" />
        <div className="method-preface">
          <EditableText textKey="mmvae-method-1">每个样本由两个模态 x₁ 与 x₂ 构成，结构化潜空间写为 z=[zᵖʳ¹,zˢʰ,zᵖʳ²]。模态编码器估计私有与共享后验，解码器使用本模态私有变量和共享变量完成同视图重建；跨视图预测通过共享变量传递信息。</EditableText>
          <EditableText textKey="mmvae-method-2">MVAE 目标包含联合输入和单模态自重建。MoPoE-VAE 额外包含 x₁→x₂ 与 x₂→x₁ 的跨模态下界。MMVAE 的 MoE 展开后覆盖全部 (m,m′) 编码-解码对。MMVAE++ 在 m=m′ 的路径上对 q(zˢʰ|xₘ) 使用 stop-gradient，仅让 m≠m′ 的误差更新共享后验。</EditableText>
        </div>
        <MMVAEPlusExplorer />
      </section>

      <section className="note-block evaluation-note">
        <NoteHeader index="03" title="评估与性能" caption="共享因素恢复、跨视图预测与线性分类" captionKey="mmvae-evaluation-caption" />
        <div className="evaluation-grid">
          <div><span>DATA</span><EditableText as="strong" textKey="mmvae-data-title">Synthetic GP · CLL · BRCA · single-cell</EditableText><EditableText textKey="mmvae-data-copy">所有实验使用两个连续数据模态；真实任务以 gene expression 与 methylation 等组学视图为主。</EditableText></div>
          <div><span>METRICS</span><EditableText as="strong" textKey="mmvae-metrics-title">Cross-view R² · AUC</EditableText><EditableText textKey="mmvae-metrics-copy">共享特征应得到较高跨视图 R²，私有特征应接近 0；AUC 检查共享潜空间中的类别可分性。</EditableText></div>
          <div><span>RESULT</span><EditableText as="strong" textKey="mmvae-result-title">私有特征增多时保持最稳</EditableText><EditableText textKey="mmvae-result-copy">BRCA 共享特征中位 R² 为 0.26；CLL 的 1000 个 other genes 设置下 AUC 为 0.62，高于其余无监督基线。</EditableText></div>
        </div>
      </section>

      <div className="source-row">
        <SourceLink href="https://proceedings.mlr.press/v240/martens24a.html">PMLR 全文</SourceLink>
        <SourceLink href="https://arxiv.org/abs/2403.06338">arXiv</SourceLink>
        <SourceLink href="https://github.com/kasparmartens/shared-private-multimodalVAE"><Github size={14} /> 作者代码</SourceLink>
      </div>
    </article>
  )
}

function ResearchPage({activePaper,onSelect}:{activePaper:PaperId;onSelect:(paper:PaperId)=>void}) {
  const selectPaper = onSelect
  useEffect(()=>{document.getElementById('main-content')?.focus({preventScroll:true})},[])

  const handleTabKey = (event: KeyboardEvent<HTMLButtonElement>, paper: PaperId) => {
    if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return
    event.preventDefault()
    const current = PAPERS.findIndex((entry) => entry.id === paper)
    const next = event.key === 'Home'
      ? 0
      : event.key === 'End'
        ? PAPERS.length - 1
        : (current + (event.key === 'ArrowRight' ? 1 : -1) + PAPERS.length) % PAPERS.length
    const nextId = PAPERS[next].id
    selectPaper(nextId)
    window.requestAnimationFrame(() => document.getElementById(`tab-${nextId}`)?.focus())
  }

  return (
    <>
      <header className="site-header">
        <a className="brand" href="#/" aria-label="返回 PRIMOCOSMOS 个人主页封面">
          <BrandMark />
          <span>PRIMOCOSMOS</span>
        </a>
        <nav aria-label="主导航">
          <a href="#/">主页</a>
          <a href="#/research/mmhvae">科研笔记</a>
        </nav>
        <div className="link-status"><span /> FIELD ONLINE</div>
      </header>

      <main id="main-content" tabIndex={-1}>
        <section className="research-notebook" id="research" aria-labelledby="research-title">
          <div className="research-intro section-shell">
            <div>
              <span className="section-code">RESEARCH / LITERATURE NOTEBOOK</span>
              <h2 id="research-title">多模态医学影像与潜变量模型</h2>
            </div>
            <div className="research-intro-copy">
              <EditableText textKey="research-introduction">每篇笔记依次整理摘要、Method、评估方式与性能。架构图保留编码器、潜变量、融合算子、解码器、损失项和训练路径，并通过交互显示观测条件与梯度流向。</EditableText>
              <NoteEditToolbar />
            </div>
          </div>

          <div className="paper-tabs section-shell" role="tablist" aria-label="选择文献笔记">
            {PAPERS.map((paper) => (
              <button
                key={paper.id}
                id={`tab-${paper.id}`}
                type="button"
                role="tab"
                aria-selected={activePaper === paper.id}
                aria-controls={`panel-${paper.id}`}
                tabIndex={activePaper === paper.id ? 0 : -1}
                className={activePaper === paper.id ? 'is-active' : ''}
                onClick={() => selectPaper(paper.id)}
                onKeyDown={(event) => handleTabKey(event, paper.id)}
              >
                <span>{paper.code}</span>
                <strong>{paper.short}</strong>
                <small>{paper.title}</small>
                <em>{paper.venue}</em>
              </button>
            ))}
          </div>

          <div className="paper-panel section-shell">
            {activePaper === 'mmhvae' && <MMHVAENote />}
            {activePaper === 'pnp' && <PnPNote />}
            {activePaper === 'mmvae' && <MMVAEPlusNote />}
          </div>
        </section>

        <section className="archive-end section-shell" aria-label="科研笔记结尾">
          <ScanLine aria-hidden="true" />
          <div><span>ARCHIVE STATUS</span><strong>3 PAPER NOTES / ACTIVE</strong></div>
          <a href="#/">返回个人主页</a>
        </section>
      </main>

      <footer>
        <div className="footer-brand"><BrandMark /><span>PRIMOCOSMOS</span></div>
        <p>Personal homepage · research notes · qMRI</p>
        <a href="https://github.com/PRIMOCOSMOS/PRIMOCOSMOS.github.io" target="_blank" rel="noreferrer"><Github size={15} /> GitHub repository</a>
      </footer>
    </>
  )
}

export default ResearchPage
