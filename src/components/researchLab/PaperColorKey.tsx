import {useRef,useState} from 'react'
import type {Run} from '../workstation/engine'
import ColorLegend from '../workstation/ColorLegend'
import {glyphPaths,type OverviewAppearance} from './overviewAppearance'
import {parseSourceValues,sourceValuesTemplate,type SourceValues} from './sourceValues'

export default function PaperColorKey({run,full,model,values,onValues}:{run:Run;full:Run;model:string;values?:SourceValues;onValues:(v?:SourceValues)=>void}){
 const input=useRef<HTMLInputElement>(null),[error,setError]=useState('');
 const appearances=[...new Map(run.steps.map(s=>s.settings.appearance as OverviewAppearance|undefined).filter((a):a is OverviewAppearance=>!!a).map(a=>[a.role,a])).values()];
 const branches=[...new Map(run.steps.map(s=>s.settings.appearance as OverviewAppearance|undefined).filter((a):a is OverviewAppearance=>!!a?.branch).map(a=>[a.branch!,{id:a.branch!,label:a.branch!,color:a.color}])).values()];
 const download=()=>{const blob=new Blob([JSON.stringify(sourceValuesTemplate(run,model),null,2)],{type:'application/json'}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=`${model}-tensor-values.json`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)};
 return <div className="paper-color-key">
  {run.sourceGraph?.overview?<details><summary>模块造型与分支配色</summary><div className="paper-glyph-key">{appearances.map(a=><span key={a.role}><svg viewBox="0 0 24 24" aria-hidden="true"><path d={glyphPaths[a.glyph]??'M3 5H17V19H3Z M7 2H21V16H17'} fill="none" stroke="currentColor" strokeWidth="1.4"/></svg>{a.role}</span>)}</div>{branches.length>0&&<div className="paper-branch-key">{branches.map(m=><span key={m.id}><i style={{background:m.color}}/>{m.label}</span>)}</div>}<p>造型区分功能，颜色识别模态与分支；总览颜色不表示张量数值。</p></details>:<><ColorLegend/><span className="paper-value-status">{values?'已导入坐标数值':'未加载数值用暗灰表示'} · 逐张量窗口归一化</span></>}
  <details className="paper-value-loader"><summary>张量数值</summary><p>进入功能块后下载坐标模板，将 null 替换为相应张量的实际数值再导入。数值只影响颜色与读数，不修改网络结构，也不会自动计算未提供的激活值。文件仅在当前浏览器内读取。</p><div className="paper-value-actions"><button disabled={!!run.sourceGraph?.overview} onClick={download}>下载当前坐标模板</button><button onClick={()=>input.current?.click()}>导入数值</button>{values&&<button onClick={()=>{onValues(undefined);setError('')}}>清除导入</button>}</div><input ref={input} hidden type="file" accept=".json,application/json" aria-label="导入文献张量数值" onChange={async e=>{const file=e.target.files?.[0];e.target.value='';if(!file)return;try{if(file.size>8*1024*1024)throw Error('文件超过 8 MB，请按模块拆分坐标窗口。');const data=parseSourceValues(await file.text(),full,model);onValues(data);setError('')}catch(error){setError(error instanceof Error?error.message:'无法读取文件，请检查 JSON 格式。')}}}/>{error&&<p role="alert">{error}</p>}{values&&<p role="status">已载入 {values.tensors.length} 个张量的坐标记录；缺失值保持未知。</p>}</details>
 </div>;
}
