import {useMemo,useState,useEffect} from 'react'
import {atomInfo} from '../mmhvae/navigation'
import {NODE_MAP} from '../mmhvae/model'
import WorkScene from '../workstation/WorkScene'
import {makeMmhvaeRun,makePaperRun} from './sourceExecution'
import type {PaperModel} from './catalog'
import '../workstation/workstation.css'
import './paperWorkScene.css'
interface Props {model?:PaperModel;observed?:string[];windowPage:number;selected:string;playing:boolean;reset:number;zoom:number;view:'orbit'|'front'|'top';labelMode:'hover'|'all'|'none';onSelect:(id:string)=>void}
export default function PaperWorkScene(p:Props){
 const run=useMemo(()=>p.model?makePaperRun(p.model,p.windowPage):makeMmhvaeRun(p.observed,p.windowPage),[p.model,p.observed,p.windowPage]);
 const initial=()=>Math.max(0,run.steps.findIndex(s=>s.settings.sourceConvolution||s.kind==='Linear'))/Math.max(1,run.steps.length);
 const [localScope,setLocalScope]=useState(p.selected);
 const choose=(id:string)=>{setLocalScope(id);if(p.model||id==='root'||NODE_MAP.has(id)||id.startsWith('atom:')&&atomInfo(id,p.observed??[])?.part)p.onSelect(id)};
 useEffect(()=>setLocalScope(p.selected),[p.selected]);
 const [progress,setProgress]=useState(initial),[index,setIndex]=useState(0),[localPlaying,setLocalPlaying]=useState(true);
 useEffect(()=>{setProgress(initial());setLocalPlaying(true)},[run]);
 return <div className="paper-work-scene" data-renderer="workstation" data-source-steps={run.steps.length}>
  <WorkScene run={run} scope={localScope} focusOnly index={index} progress={progress} playing={p.playing&&localPlaying} reset={p.reset} focus="" highlight="" zoom={Math.log(p.zoom)/.23} pan={false} view={p.view} labelMode={p.labelMode} onScope={choose} onIndex={setIndex} onProgress={setProgress} onPlaying={setLocalPlaying}/>
  <div className="paper-coordinate-key"><span>完整计算图 · {run.steps.length} 个计算步骤</span><span>水晶 = 源码坐标 · 冷色数据 / 暖色参数</span><span>自动轮播 · 悬停接管 · 点击聚焦</span></div>
  {run.sourceGraph?.warnings.length?<details className="paper-source-warning"><summary>源码图的待解析关系</summary>{run.sourceGraph.warnings.map(w=><p key={w}>{w}</p>)}</details>:null}
 </div>
}
