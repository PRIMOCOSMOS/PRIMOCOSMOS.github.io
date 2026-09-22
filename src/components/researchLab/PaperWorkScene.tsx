import {useMemo,useState,useEffect} from 'react'
import {atomInfo} from '../mmhvae/navigation'
import {NODE_MAP} from '../mmhvae/model'
import WorkScene from '../workstation/WorkScene'
import {makeMmhvaeRun,makePaperRun} from './sourceExecution'
import {overviewRun,sourceSlice} from './sourceLod'
import type {PaperModel} from './catalog'
import '../workstation/workstation.css'
import './paperWorkScene.css'
interface Props {model?:PaperModel;observed?:string[];temperature?:number;windowPage:number;selected:string;playing:boolean;reset:number;zoom:number;view:'orbit'|'front'|'top';labelMode:'hover'|'all'|'none';onSelect:(id:string)=>void}
export default function PaperWorkScene(p:Props){
 const full=useMemo(()=>p.model?makePaperRun(p.model,p.windowPage):makeMmhvaeRun(p.observed,p.windowPage,p.temperature),[p.model,p.observed,p.windowPage,p.temperature]);
 const [location,setLocation]=useState(p.selected),[focus,setFocus]=useState('root');
 useEffect(()=>{setLocation(p.selected);setFocus('root')},[p.selected]);
 const run=useMemo(()=>{const slice=location==='root'?full:sourceSlice(full,location);return slice.steps.length>180?overviewRun(slice,p.model,location):slice},[full,location,p.model]);
 const overview=!!run.sourceGraph?.overview;
 const initial=()=>Math.max(0,run.steps.findIndex(s=>s.settings.sourceConvolution||s.kind==='Linear'))/Math.max(1,run.steps.length);
 const [progress,setProgress]=useState(initial),[index,setIndex]=useState(0),[localPlaying,setLocalPlaying]=useState(true);
 useEffect(()=>{setProgress(initial());setLocalPlaying(true);setFocus('root')},[run]);
 const choose=(id:string)=>{
  if(overview){setLocation(id);setFocus('root');if(p.model?p.model.entries[id]:id==='root'||NODE_MAP.has(id)||id.startsWith('atom:')&&atomInfo(id,p.observed??[])?.part)p.onSelect(id)}else setFocus(id);
 };
 const home=()=>{setLocation('root');setFocus('root');p.onSelect('root')};
 return <div className="paper-work-scene" data-renderer="workstation" data-source-steps={full.steps.length} data-displayed-steps={run.steps.length} data-detail={location} data-lod={overview?'architecture':'mathematics'}>
  <div className="paper-scene-navigation"><button onClick={home} disabled={location==='root'}>返回架构总览</button><strong>{overview?'架构总览':'数学计算结构'}</strong><span>{overview?'按功能块组织 · 点击查看全部运算':'张量、参数与运算全部展开 · 点击聚焦'}</span><button aria-pressed={p.playing&&localPlaying} onClick={()=>setLocalPlaying(v=>!v)}>{localPlaying?'暂停演算':'播放演算'}</button></div>
  <WorkScene run={run} scope={focus} focusOnly index={index} progress={progress} playing={p.playing&&localPlaying} reset={p.reset} focus="" highlight="" zoom={Math.log(p.zoom)/.23} pan={false} view={p.view} labelMode={p.labelMode} onScope={choose} onIndex={setIndex} onProgress={setProgress} onPlaying={setLocalPlaying}/>
  <div className="paper-coordinate-key"><span>{overview?`${run.steps.length} 个功能块 · ${full.steps.length} 个数学步骤`:`${run.steps.length} 个展开的数学步骤`}</span><span>{overview?'模块轮廓代表功能接口；进入后逐元素显示':'水晶 = 源码坐标 · 冷色数据 / 暖色参数'}</span><span>自动轮播 · 悬停接管</span></div>
  {run.sourceGraph?.warnings.length?<details className="paper-source-warning"><summary>源码图的待解析关系</summary>{run.sourceGraph.warnings.map(w=><p key={w}>{w}</p>)}</details>:null}
 </div>
}
