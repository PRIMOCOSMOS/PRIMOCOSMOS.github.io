import { lazy, Suspense, useEffect, useState } from 'react'
import Home from './components/home/Home'
const ResearchPage=lazy(()=>import('./ResearchPage'))
import {paperById,type PaperId} from './papers'
type Route={page:'home'|'research';paper:PaperId;anchor?:string}
function readRoute(hash:string,previous?:Route):Route{
 const path=hash.replace(/^#/, '')
 if(!path||path==='/'||path==='home')return {page:'home',paper:'mmhvae'}
 if(path.startsWith('/research')){const candidate=path.split('/')[2];return {page:'research',paper:paperById(candidate)?.id??'mmhvae'}}
 if(path==='mm-module-directory')return {page:'research',paper:'mmhvae',anchor:path}
 if(path==='research'||/^(mmhvae|pnp|mmvae|ssdiff|metsc|pigment)-orbit-lab$/.test(path)||/^(panel|tab)-(mmhvae|pnp|mmvae|ssdiff|metsc|pigment)$/.test(path)){const id=path.replace(/^(panel|tab)-/,'').replace(/-orbit-lab$/,'');return {page:'research',paper:paperById(id)?.id??'mmhvae',anchor:path}}
 return previous??{page:'home',paper:'mmhvae'}
}
export default function App(){
 const [route,setRoute]=useState(()=>readRoute(location.hash))
 useEffect(()=>{const change=()=>setRoute(previous=>readRoute(location.hash,previous));window.addEventListener('hashchange',change);return()=>window.removeEventListener('hashchange',change)},[])
 useEffect(()=>{
  document.title=route.page==='home'?'PRIMOCOSMOS：SILICONDEVINE':`${paperById(route.paper)!.short} · PRIMOCOSMOS 文献档案`
  if(!route.anchor){window.scrollTo({top:0,behavior:'instant'});return}
  const scroll=()=>{const element=document.getElementById(route.anchor!);if(element){element.scrollIntoView({block:'start',behavior:'instant'});return true}return false}
  if(scroll())return
  const observer=new MutationObserver(()=>{if(scroll())observer.disconnect()});observer.observe(document.body,{childList:true,subtree:true});const timeout=window.setTimeout(()=>observer.disconnect(),8000)
  return()=>{observer.disconnect();clearTimeout(timeout)}
 },[route])
 return route.page==='home'?<Home/>:<Suspense fallback={<main id="main-content" className="archive-loading" role="status">正在打开文献档案…</main>}><ResearchPage activePaper={route.paper} onSelect={id=>{location.hash=`/research/${id}`}}/></Suspense>
}
