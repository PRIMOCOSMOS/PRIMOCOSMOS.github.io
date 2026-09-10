import {useMemo} from 'react'
import ResearchExplorer from './researchLab/ResearchExplorer'
import {pnpModel} from './researchLab/pnpModel'
export function PnPCosmoExplorer(){const model=useMemo(()=>pnpModel(),[]);return <ResearchExplorer model={model}/>}
