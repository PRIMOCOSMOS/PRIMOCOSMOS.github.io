import { useEffect, useRef, useState, type MouseEvent } from 'react'
import { ArrowUpRight, Pause, Play, Github } from 'lucide-react'
import { EditableText, NoteEditToolbar } from '../EditableContent'
import { Sigil, Wordmark } from './Wordmark'
import './home.css'

const papers=[['mmhvae','MMHVAE','层次潜变量 · 跨模态合成'],['pnp','PnP-CoSMo','内容 / 风格 · MRI 重建'],['mmvae','MMVAE++','共享 / 私有 · 多模态学习']]
export default function Home(){
 const root=useRef<HTMLDivElement>(null),[motion,setMotion]=useState(()=>!matchMedia('(prefers-reduced-motion: reduce)').matches),[leaving,setLeaving]=useState(false)
 const departure=useRef<number>()
 useEffect(()=>{const media=matchMedia('(prefers-reduced-motion: reduce)'),changed=()=>setMotion(!media.matches);media.addEventListener('change',changed);return()=>{media.removeEventListener('change',changed);clearTimeout(departure.current)}},[])
 useEffect(()=>{
  const el=root.current!;let frame=0,px=0,py=0,x=0,y=0,last=0
  const move=(e:PointerEvent)=>{if(e.pointerType==='touch')return;px=(e.clientX/innerWidth-.5)*2;py=(e.clientY/innerHeight-.5)*2}
  const reset=()=>{px=0;py=0}
  const tick=(time:number)=>{frame=requestAnimationFrame(tick);if(document.hidden||time-last<32)return;last=time;x+=(px-x)*.07;y+=(py-y)*.07;el.style.setProperty('--aim-x',`${x*14}px`);el.style.setProperty('--aim-y',`${y*9}px`)}
  const visibility=()=>el.classList.toggle('is-background',document.hidden)
  if(motion){window.addEventListener('pointermove',move,{passive:true});document.addEventListener('pointerleave',reset);document.addEventListener('visibilitychange',visibility);frame=requestAnimationFrame(tick)}else{el.style.setProperty('--aim-x','0px');el.style.setProperty('--aim-y','0px')}
  return()=>{cancelAnimationFrame(frame);window.removeEventListener('pointermove',move);document.removeEventListener('pointerleave',reset);document.removeEventListener('visibilitychange',visibility)}
 },[motion])
 const enter=(e:MouseEvent<HTMLAnchorElement>)=>{if(e.metaKey||e.ctrlKey||e.shiftKey||e.altKey||e.button!==0||!motion)return;e.preventDefault();if(leaving)return;setLeaving(true);const url=e.currentTarget.hash;departure.current=window.setTimeout(()=>{window.location.hash=url},400)}
 return <div ref={root} className={`silicon-home${motion?' is-animated':''}${leaving?' is-departing':''}`}>
  <div className="silicon-atmosphere" aria-hidden="true"><img className="silicon-colossus" src="/assets/silicon-colossus.png" alt="" fetchPriority="high"/><div className="silicon-grazing-light"/><div className="silicon-haze"/></div>
  <header className="silicon-header"><a className="silicon-signature" href="#/" aria-label="PRIMOCOSMOS 主页"><Sigil/></a><nav aria-label="主页导航"><a href="#/research/mmhvae" onClick={enter}>文献档案 <ArrowUpRight size={14}/></a><a href="https://github.com/PRIMOCOSMOS/PRIMOCOSMOS.github.io" target="_blank" rel="noreferrer">GitHub <Github size={14}/></a></nav></header>
  <main id="main-content" className="silicon-main" tabIndex={-1}>
   <section className="silicon-identity" aria-label="PRIMOCOSMOS：SILICONDEVINE">
    <h1><span className="silicon-sr-only">PRIMOCOSMOS：SILICONDEVINE</span><Wordmark/></h1>
    <div className="silicon-entry"><a className="silicon-enter" href="#/research/mmhvae" onClick={enter}><span>进入文献档案</span><ArrowUpRight size={26}/></a><EditableText textKey="silicon-home-description" className="silicon-description">qMRI · 多模态建模 · 交互式文献分析</EditableText></div>
   </section>
   <nav className="silicon-archives" aria-label="直接打开文献">{papers.map(([id,title,topic])=><a key={id} href={`#/research/${id}`} onClick={enter}><span>{title}</span><small>{topic}</small><ArrowUpRight size={19}/></a>)}</nav>
  </main>
  <footer className="silicon-footer"><span>PRIMOCOSMOS</span><div><button onClick={()=>setMotion(v=>!v)} aria-pressed={motion}>{motion?<Pause size={13}/>:<Play size={13}/>}<span>{motion?'暂停动效':'开启动效'}</span></button><details><summary>编辑主页文字</summary><NoteEditToolbar compact/></details></div></footer>
  <div className="silicon-transition" aria-hidden="true"/>
 </div>
}
