import { lazy, Suspense, useEffect, useState } from 'react'
import Home from './components/home/Home'
const ResearchPage=lazy(()=>import('./ResearchPage'))
type PaperId='mmhvae'|'pnp'|'mmvae'
type Route={page:'home'|'research';paper:PaperId;anchor?:string}
function readRoute(hash:string,previous?:Route):Route{
 const path=hash.replace(/^#/, '')
 if(!path||path==='/'||path==='home')return {page:'home',paper:'mmhvae'}
 if(path.startsWith('/research')){const candidate=path.split('/')[2];return {page:'research',paper:candidate==='pnp'||candidate==='mmvae'?candidate:'mmhvae'}}
 if(path==='research'||path==='mmhvae-orbit-lab'||path==='mm-module-directory'||/^(panel|tab)-(mmhvae|pnp|mmvae)$/.test(path))return {page:'research',paper:path.includes('pnp')?'pnp':path.includes('mmvae')?'mmvae':'mmhvae',anchor:path}
 return previous??{page:'home',paper:'mmhvae'}
}
export default function App(){
 const [route,setRoute]=useState(()=>readRoute(location.hash))
 useEffect(()=>{const change=()=>setRoute(previous=>readRoute(location.hash,previous));window.addEventListener('hashchange',change);return()=>window.removeEventListener('hashchange',change)},[])
 useEffect(()=>{
  document.title=route.page==='home'?'PRIMOCOSMOS：SILICONDEVINE':`${route.paper==='mmhvae'?'MMHVAE':route.paper==='pnp'?'PnP-CoSMo':'MMVAE++'} · PRIMOCOSMOS 文献档案`
  if(!route.anchor){window.scrollTo({top:0,behavior:'instant'});return}
  const scroll=()=>{const element=document.getElementById(route.anchor!);if(element){element.scrollIntoView({block:'start',behavior:'instant'});return true}return false}
  if(scroll())return
  const observer=new MutationObserver(()=>{if(scroll())observer.disconnect()});observer.observe(document.body,{childList:true,subtree:true});const timeout=window.setTimeout(()=>observer.disconnect(),8000)
  return()=>{observer.disconnect();clearTimeout(timeout)}
 },[route])
 return route.page==='home'?<Home/>:<Suspense fallback={<main id="main-content" className="archive-loading" role="status">正在打开文献档案…</main>}><ResearchPage activePaper={route.paper} onSelect={id=>{location.hash=`/research/${id}`}}/></Suspense>
}
