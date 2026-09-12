export type PaperId='mmhvae'|'pnp'|'mmvae'|'ssdiff'|'metsc'|'pigment'
export type PaperCategory='encoder'|'core'
export interface PaperRecord{id:PaperId;short:string;title:string;venue:string;category:PaperCategory;authors:string;topic:string;paperUrl:string;repo?:string}
export const CATEGORIES=[{id:'encoder',title:'编码器',description:'多模态表征与生成结构'},{id:'core',title:'课题核心',description:'扩散 MRI 与微结构建模'}] as const
export const PAPERS:PaperRecord[]=[
 {id:'mmhvae',short:'MMHVAE',title:'Unified Cross-Modal Medical Image Synthesis with Hierarchical Mixture of Product-of-Experts',venue:'IEEE TPAMI · 48(2)',category:'encoder',authors:'Dorent et al.',topic:'层次潜变量 · 缺失模态合成',paperUrl:'https://doi.org/10.1109/TPAMI.2025.3616632',repo:'https://github.com/ReubenDo/MMHVAE'},
 {id:'pnp',short:'PnP-CoSMo',title:'A Plug-and-Play Method for Guided Multi-contrast MRI Reconstruction Based on Content/Style Modeling',venue:'Medical Image Analysis · 2026',category:'encoder',authors:'Rao et al.',topic:'内容 / 风格 · 引导重建',paperUrl:'https://github.com/cnmy-ro/pnp-cosmo',repo:'https://github.com/cnmy-ro/pnp-cosmo'},
 {id:'mmvae',short:'MMVAE++',title:'Disentangling Shared and Private Latent Factors in Multimodal Variational Autoencoders',venue:'PMLR 240 · 2024',category:'encoder',authors:'Märtens & Yau',topic:'共享 / 私有 · 梯度分离',paperUrl:'https://proceedings.mlr.press/v240/martens24a.html',repo:'https://github.com/kasparmartens/shared-private-multimodalVAE'},
 {id:'ssdiff',short:'SSDiff',title:'A foundation model for efficient and assumption-free characterization of brain microstructure from diffusion MRI',venue:'Research Square · 2026 · 预印本',category:'core',authors:'Weikang Gong et al.',topic:'自监督 · 方向 token · 基础表征',paperUrl:'https://doi.org/10.21203/rs.3.rs-8877816/v1',repo:'https://github.com/weikanggong1/SSDiff'},
 {id:'metsc',short:'METSC',title:'A microstructure estimation Transformer inspired by sparse representation for diffusion MRI',venue:'Medical Image Analysis · 86 · 2023',category:'core',authors:'Tianshu Zheng et al.',topic:'空间注意力 · 稀疏字典 · NODDI',paperUrl:'https://doi.org/10.1016/j.media.2023.102788',repo:'https://github.com/Tianshu996/METSC'},
 {id:'pigment',short:'PIGMENT',title:'A physics-informed foundation model for quantitative diffusion MRI',venue:'arXiv:2606.00156 · 2026 · 预印本',category:'core',authors:'Zihan Li et al.',topic:'离散潜空间 · 物理约束 · 生成先验',paperUrl:'https://arxiv.org/abs/2606.00156'},
]
export const paperById=(id:string)=>PAPERS.find(p=>p.id===id)
