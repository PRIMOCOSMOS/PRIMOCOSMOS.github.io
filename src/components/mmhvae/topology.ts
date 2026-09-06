import { LEVELS, MODALITIES } from './model'

export type Point3 = [number, number, number]
export interface Link3 { from:string; to:string; label:string; l:number; mod?:string }
export const levelY=(l:number)=>10+(l-1)*16
export function modelTopology(){
  const positions=new Map<string,Point3>(),links:Link3[]=[]
  const at=(id:string,x:number,y:number,z:number)=>positions.set(id,[x,y,z])
  const link=(from:string,to:string,label:string,l:number,mod?:string)=>links.push({from,to,label,l,mod})
  for(const m of MODALITIES){
    const x=m.x*2.6,z=m.z*2.7
    at(`${m.id}-input`,x,3,z);at(`${m.id}-stem`,x,6,z)
    link(`${m.id}-input`,`${m.id}-stem`,'1 × 192²',0,m.id);link(`${m.id}-stem`,`${m.id}-encoder-1`,'16 × 192²',0,m.id)
    for(const v of LEVELS){const l=v.l,y=levelY(l)
      at(`${m.id}-encoder-${l}`,x,y+5,z)
      if(l<7){at(`${m.id}-down-${l}`,x,y+12,z);link(`${m.id}-encoder-${l}`,`${m.id}-down-${l}`,'编码 ↑',l,m.id);link(`${m.id}-down-${l}`,`${m.id}-encoder-${l+1}`,'Conv stride 2',l,m.id)
        at(`${m.id}-concat-${l}`,x*.7,y+3.8,z*.85)
        link(`${m.id}-encoder-${l}`,`${m.id}-concat-${l}`,'skip feature',l,m.id)
        link(`feature-${l}`,`${m.id}-concat-${l}`,'生成特征 gₗ',l,m.id)
        link(`${m.id}-concat-${l}`,`${m.id}-expert-${l}`,'Concat → Qₗ',l,m.id)
      }else link(`${m.id}-encoder-${l}`,`${m.id}-expert-${l}`,'Flatten → Linear',l,m.id)
      at(`${m.id}-expert-${l}`,x*.7,y+1.8,z*.85)
      at(`${m.id}-factor-${l}`,x*.49,y-.2,z*.66)
      link(`${m.id}-expert-${l}`,`${m.id}-factor-${l}`,'μⱼ · log sⱼ',l,m.id)
      link(`${m.id}-factor-${l}`,`poe-${l}`,'观测残差 expert',l,m.id)
    }
    at(`${m.id}-output`,x,0,z);at(`${m.id}-image`,x,-3.8,z+1)
    link('sample-1',`${m.id}-output`,'共享 z₁ → 独立 Decoder',1,m.id)
    link(`${m.id}-output`,`${m.id}-image`,'Tanh · mask → 1 × 192²',1,m.id)
  }
  for(const v of LEVELS){const l=v.l,y=levelY(l)
    if(l<7){at(`up-${l}`,0,y+7.8,0);at(`decoder-${l}`,0,y+5.8,0);at(`feature-${l}`,0,y+3.8,0)
      at(`prior-head-${l}`,-4.5,y+1.8,-7)
      link(l===6?'lift':`sample-${l+1}`,`up-${l}`,'上一级样本',l)
      link(`up-${l}`,`decoder-${l}`,'×2 → Conv3',l);link(`decoder-${l}`,`feature-${l}`,'生成特征',l)
      link(`feature-${l}`,`prior-head-${l}`,'gₗ → Pₗ',l);link(`prior-head-${l}`,`prior-${l}`,'μp · exp(ap)',l)
    }
    at(`prior-${l}`,-4.5,y-.2,-7);at(`poe-${l}`,0,y-2.2,0);at(`posterior-${l}`,0,y-4.2,0);at(`sample-${l}`,0,y-6.2,0)
    link(`prior-${l}`,`poe-${l}`,l===7?'N(0,I)':'条件先验',l)
    link(`poe-${l}`,`posterior-${l}`,'融合 μ · T·s',l);link(`posterior-${l}`,`sample-${l}`,'rsample()',l)
  }
  at('lift',0,levelY(7)-7.2,0);link('sample-7','lift','256 → 128 × 3²',7)
  return {positions,links}
}
