import { useEffect, useRef, useState } from 'react'
import * as T from 'three'
import { OrbitControls } from 'three/addons/controls/OrbitControls.js'
import { MODALITIES } from './model'
import { type AnatomyGraph } from './anatomy'
import { makeGlyph, disposeGroup } from './glyphs'

interface Props { graph:AnatomyGraph; active:string|null; motion:boolean; zoom:number; onSelect:(id:string)=>void; reducedMotion:boolean }
export default function AnatomyScene(props:Props){
  const host=useRef<HTMLDivElement>(null),latest=useRef(props);latest.current=props
  const [failed,setFailed]=useState(false)
  useEffect(()=>{
    const container=host.current!;let renderer:T.WebGLRenderer
    try{renderer=new T.WebGLRenderer({antialias:true,alpha:true,powerPreference:'low-power'})}catch{setFailed(true);return}
    renderer.setPixelRatio(Math.min(devicePixelRatio,1.6));renderer.setClearColor('#071119',0);container.appendChild(renderer.domElement)
    renderer.domElement.setAttribute('aria-label',`${props.graph.title}：可拖动旋转，点击算子查看张量和参数。下方同序按钮支持键盘操作。`)
    renderer.domElement.setAttribute('role','img');renderer.domElement.tabIndex=0
    const scene=new T.Scene(),content=new T.Group();scene.add(content)
    scene.add(new T.AmbientLight('#d1e4eb',1.7));const light=new T.DirectionalLight('#d8edff',2);light.position.set(0,8,15);scene.add(light)
    const camera=new T.OrthographicCamera(-12,12,9,-9,.1,120);camera.position.set(1.8,3.8,32)
    const controls=new OrbitControls(camera,renderer.domElement);controls.enableZoom=false;controls.enableDamping=true;controls.minPolarAngle=.3;controls.maxPolarAngle=2.6
    const labels:T.Sprite[]=[],hits:T.Object3D[]=[],parts:{id:string;group:T.Group;dest:T.Vector3;label:T.Sprite;dim:T.Sprite;major:boolean}[]=[]
    const text=(value:string,color:string,width=3,height=.32)=>{
      const canvas=document.createElement('canvas'),ctx=canvas.getContext('2d')!;ctx.font='500 36px "Segoe UI", sans-serif';canvas.width=Math.ceil(ctx.measureText(value).width+18);canvas.height=52
      ctx.fillStyle='#071119e8';ctx.fillRect(0,0,canvas.width,52);ctx.font='500 36px "Segoe UI", sans-serif';ctx.fillStyle=color;ctx.textAlign='center';ctx.textBaseline='middle';ctx.fillText(value,canvas.width/2,26)
      const map=new T.CanvasTexture(canvas);map.colorSpace=T.SRGBColorSpace;const s=new T.Sprite(new T.SpriteMaterial({map,depthTest:false,transparent:true}));const h=Math.min(height,width/(canvas.width/52));s.scale.set(h*canvas.width/52,h,1);s.renderOrder=5;labels.push(s);return s
    }
    const graph=props.graph,columns=container.clientWidth<500?3:4,rows=Math.ceil(graph.parts.length/columns)
    const positions=new Map<string,T.Vector3>()
    const compact=container.clientWidth<500,experts=graph.parts.filter(p=>p.id.endsWith('-q'))
    const compactBottom=-2.5-(Math.ceil(experts.length/2)-1)*5.4
    const compactPosition=(id:string):[number,number]|undefined=>{
      const hasLift=graph.parts.some(p=>p.id==='lift')
      const top=hasLift?8:5.5
      const fixed:Record<string,[number,number]>={previous:[-3.2,top],lift:[0,top],up:[hasLift?3.2:0,top],decoder:[3.2,5.5],prior:[-3.2,0],poe:[-3.2,compactBottom],posterior:[0,compactBottom],sample:[3.2,compactBottom],next:[3.2,compactBottom-2.7]}
      if(fixed[id])return fixed[id]
      const index=experts.findIndex(p=>p.id===id.replace('-skip','-q'))
      if(index>=0)return [(index%2)*3.2,(id.endsWith('-skip')?2.5:0)-Math.floor(index/2)*5.4]
      const outputIndex=MODALITIES.findIndex(m=>id===`${m.id}-output`||id===`${m.id}-image`)
      if(outputIndex>=0)return [(outputIndex-1.5)*3.2,compactBottom-(id.endsWith('-image')?5.4:2.8)]
      return undefined
    }
    graph.parts.forEach((part,i)=>{
      const row=Math.floor(i/columns),column=row%2?columns-1-i%columns:i%columns
      const position=part.position&&compact?compactPosition(part.id)??part.position:part.position
      const pos=position?new T.Vector3(position[0],position[1],0):new T.Vector3((column-(columns-1)/2)*3.25,((rows-1)/2-row)*2.75,0)
      positions.set(part.id,pos)
      const color=part.color??MODALITIES.find(m=>m.id===part.mod)?.color??(['gaussian','linear','up'].includes(part.glyph)?'#78b8ff':['poe','sum'].includes(part.glyph)?'#ebdfb9':['sample','image'].includes(part.glyph)?'#d8ff45':'#a7c9c9')
      const group=new T.Group(),glyph=makeGlyph(part.glyph,color);glyph.scale.setScalar(.7);group.add(glyph)
      const shortName=part.position?({previous:part.title.split(' · ')[0],decoder:'BlockDecoder',prior:'先验 pₗ',poe:'PoE',posterior:'后验 qₗ',sample:part.title,next:'下一级 ↑'} as Record<string,string>)[part.id]??(part.id.endsWith('-q')?`${MODALITIES.find(m=>m.id===part.mod)?.label} / Qₗ`:part.title):part.title
      const label=text(`${part.position?'':String(i+1).padStart(2,'0')+'  '}${shortName}${part.child?' ↗':''}`,color,3.3,.44);label.position.set(0,.94,.1);group.add(label)
      const dim=text(part.shape,'#afc4cc',3.1,.32);dim.position.set(0,-.87,.1);group.add(dim)
      const major=!!part.child||['image','gaussian','poe','sample','tensor'].includes(part.glyph)
      // Broad transparent hit area includes wire-only operators without changing their silhouette.
      const hit=new T.Mesh(new T.PlaneGeometry(2.6,2.25),new T.MeshBasicMaterial({transparent:true,opacity:0,depthWrite:false,side:T.DoubleSide}));hit.userData.part=part.id;group.add(hit);hits.push(hit)
      content.add(group);parts.push({id:part.id,group,dest:pos,label,dim,major})
    })
    const bounds=new T.Box3().setFromPoints([...positions.values()]);const center=bounds.getCenter(new T.Vector3());content.position.sub(center)
    const shell=new T.LineSegments(new T.EdgesGeometry(new T.BoxGeometry(bounds.max.x-bounds.min.x+3.2,bounds.max.y-bounds.min.y+2.65,2)),new T.LineBasicMaterial({color:'#8eb5c9',transparent:true,opacity:.13}));scene.add(shell)
    const flows:{curve:T.CatmullRomCurve3;line:T.Line;dot:T.Mesh;arrow:T.Mesh;from:string;to:string;label?:T.Sprite}[]=[]
    graph.edges.forEach(edge=>{
      const a=positions.get(edge.from)!,b=positions.get(edge.to)!,delta=b.clone().sub(a),start=a.clone().addScaledVector(delta.clone().normalize(),.72),end=b.clone().addScaledVector(delta.clone().normalize(),-.83)
      const middle=start.clone().lerp(end,.5);middle.z=edge.residual?-1.6:-.18
      if(edge.residual)middle.x=Math.min(bounds.min.x-1.5,middle.x-2)
      const curve=new T.CatmullRomCurve3([start,middle,end]),color=edge.residual?'#e3aa82':'#5e919e'
      const line=new T.Line(new T.BufferGeometry().setFromPoints(curve.getPoints(35)),new T.LineBasicMaterial({color,transparent:true,opacity:.42}));content.add(line)
      const dot=new T.Mesh(new T.SphereGeometry(.067,6,6),new T.MeshBasicMaterial({color:edge.residual?'#ffb58b':'#d8ff45'}));content.add(dot)
      const arrow=new T.Mesh(new T.ConeGeometry(.09,.24,8),new T.MeshBasicMaterial({color}));arrow.position.copy(end);arrow.quaternion.setFromUnitVectors(new T.Vector3(0,1,0),curve.getTangent(1).normalize());content.add(arrow)
      let label:T.Sprite|undefined;if(edge.label){label=text(edge.label,edge.residual?'#d8a383':'#9bb9c2',2.5,.24);label.position.copy(middle).add(new T.Vector3(0,.22,0));content.add(label)}
      flows.push({curve,line,dot,arrow,from:edge.from,to:edge.to,label})
    })
    let frame=0,startTime=performance.now(),last=0,visible=true,hover='',previous='',dirty=true
    const resize=()=>{const w=container.clientWidth,h=container.clientHeight;renderer.setSize(w,h);const half=Math.max((bounds.max.y-bounds.min.y+4)/2,(bounds.max.x-bounds.min.x+5)/2/(w/h));camera.left=-half*w/h;camera.right=half*w/h;camera.top=half;camera.bottom=-half;camera.updateProjectionMatrix();dirty=true}
    const resizeObserver=new ResizeObserver(resize);resizeObserver.observe(container);resize()
    const observer=new IntersectionObserver(([e])=>{visible=e.isIntersecting});observer.observe(container)
    const ray=new T.Raycaster(),pointer=new T.Vector2();let down=[0,0]
    const pick=(e:PointerEvent)=>{const r=container.getBoundingClientRect();pointer.set((e.clientX-r.left)/r.width*2-1,1-(e.clientY-r.top)/r.height*2);ray.setFromCamera(pointer,camera);return ray.intersectObjects(hits)[0]?.object.userData.part as string|undefined}
    const pointerdown=(e:PointerEvent)=>{down=[e.clientX,e.clientY]}
    const pointerup=(e:PointerEvent)=>{if(Math.hypot(e.clientX-down[0],e.clientY-down[1])<5){const id=pick(e);if(id)latest.current.onSelect(id)}}
    const move=(e:PointerEvent)=>{if(e.buttons)return;hover=pick(e)??'';renderer.domElement.style.cursor=hover?'pointer':'grab';dirty=true}
    const leave=()=>{hover='';dirty=true}
    const change=()=>{dirty=true};controls.addEventListener('change',change)
    renderer.domElement.addEventListener('pointerdown',pointerdown);renderer.domElement.addEventListener('pointerup',pointerup);renderer.domElement.addEventListener('pointermove',move);renderer.domElement.addEventListener('pointerleave',leave)
    const lost=(e:Event)=>{e.preventDefault();setFailed(true)};renderer.domElement.addEventListener('webglcontextlost',lost)
    const animate=(now:number)=>{
      frame=requestAnimationFrame(animate);if(!visible||document.hidden)return
      const p=latest.current,t=p.reducedMotion?1:Math.min((now-startTime)/900,1),ease=1-(1-t)**3,key=`${p.active}|${hover}|${p.motion}|${p.zoom}`
      if(t<1||key!==previous||dirty){
        parts.forEach((part,i)=>{const progress=p.reducedMotion?1:Math.max(0,Math.min((now-startTime-i*8)/650,1));part.group.position.copy(part.dest).multiplyScalar(1-(1-progress)**3);const active=part.id===(hover||p.active);part.label.visible=part.major||active;part.dim.visible=part.major||active;part.group.scale.setScalar(active?1.1:1)})
        camera.zoom=(.78+.22*ease)*p.zoom;camera.updateProjectionMatrix()
        flows.forEach(f=>{const active=!p.active||f.from===p.active||f.to===p.active;(f.line.material as T.LineBasicMaterial).opacity=(active?.65:.18)*ease;f.arrow.visible=t>.8;f.dot.visible=p.motion&&!p.reducedMotion&&active&&t>.8;if(f.label)f.label.visible=active&&t>.8})
        previous=key;dirty=true
      }
      if(p.motion&&!p.reducedMotion&&now-last>24){flows.forEach((f,i)=>{if(f.dot.visible)f.dot.position.copy(f.curve.getPointAt((now*.00024+i*.17)%1))});dirty=true;last=now}
      controls.update();if(dirty){renderer.render(scene,camera);dirty=false}
    };frame=requestAnimationFrame(animate)
    return()=>{cancelAnimationFrame(frame);resizeObserver.disconnect();observer.disconnect();controls.dispose();disposeGroup(scene);renderer.dispose();renderer.domElement.remove()}
  },[props.graph])
  return <div className="mm-anatomy-canvas" ref={host}>{failed&&<p className="mm-anatomy-fallback">三维视图暂不可用。下方算子序列仍可逐项展开、查看张量尺寸和参数。</p>}</div>
}
