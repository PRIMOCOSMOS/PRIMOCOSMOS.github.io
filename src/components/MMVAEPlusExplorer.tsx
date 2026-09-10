import {useMemo,useState} from 'react'
import ResearchExplorer from './researchLab/ResearchExplorer'
import {sharedPrivateModel,type VAEVariant} from './researchLab/sharedPrivateModel'
export function MMVAEPlusExplorer(){
 const [variant,setVariant]=useState<VAEVariant>('MMVAE++'),model=useMemo(()=>sharedPrivateModel(variant),[variant])
 return <ResearchExplorer key={variant} model={model} variantControl={<div className="rl-variants" aria-label="重建梯度规则">{(['MMVAE++','MMVAE'] as const).map(v=><button key={v} aria-pressed={variant===v} onClick={()=>setVariant(v)}>{v}</button>)}</div>}/>
}
