import { useEffect, useRef, useState } from 'react'
import * as THREE from 'three'
import { OrbitControls } from 'three/addons/controls/OrbitControls.js'
import { LEVELS, MODALITIES, NODE_MAP, type ModelNode } from './model'

export interface SceneProps {
  selected: string; observed: string[]; target: string; level: number
  isolate: boolean; spread: number; playing: boolean; stage: number
  temperature: number; view: 'orbit' | 'front' | 'top'; reset: number; zoom: number; focus: number
  reducedMotion: boolean; onSelect: (id: string) => void
}
type Visual = { group: THREE.Group; node: ModelNode; materials: THREE.Material[]; edge: THREE.LineBasicMaterial }
type Route = { curve: THREE.CatmullRomCurve3; line: THREE.Line; particle: THREE.Mesh; mod?: string; l: number; stage: number }

export default function OrbitScene(props: SceneProps) {
  const host = useRef<HTMLDivElement>(null)
  const latest = useRef(props)
  latest.current = props
  const [error,setError] = useState(false)
  const [hover,setHover] = useState('')
  useEffect(()=>{
    const container = host.current!
    let renderer: THREE.WebGLRenderer
    try { renderer = new THREE.WebGLRenderer({antialias:true,alpha:true,powerPreference:'low-power'}) }
    catch { setError(true); return }
    renderer.setPixelRatio(Math.min(window.devicePixelRatio,1.75))
    renderer.setClearColor('#060d13',1)
    renderer.outputColorSpace=THREE.SRGBColorSpace
    container.appendChild(renderer.domElement)
    renderer.domElement.setAttribute('aria-label','MMHVAE 三维模型，可拖动旋转；键盘左右旋转、上下倾斜，加减号缩放。模块也可通过下方目录选择。')
    renderer.domElement.setAttribute('role','img')
    renderer.domElement.tabIndex=0
    const scene=new THREE.Scene()
    const camera=new THREE.OrthographicCamera(-15,15,10,-10,0.1,180)
    camera.position.set(0,15,36)
    const controls=new OrbitControls(camera,renderer.domElement)
    controls.target.set(0,7,0)
    controls.enableDamping=true; controls.dampingFactor=0.09
    controls.minPolarAngle=0.15; controls.maxPolarAngle=Math.PI/2-0.04
    controls.minZoom=0.65; controls.maxZoom=2.8
    controls.enablePan=true; controls.enableZoom=false
    // Leave wheel scrolling to the research notebook; explicit controls handle zoom.
    scene.add(new THREE.AmbientLight('#b4cbe1',1.2))
    const light=new THREE.DirectionalLight('#e4f3ff',1.5);light.position.set(-6,20,15);scene.add(light)
    const rim=new THREE.DirectionalLight('#6e9dba',1.4);rim.position.set(8,10,-10);scene.add(rim)
    const content=new THREE.Group();scene.add(content)
    const visuals:Visual[]=[]; const hitObjects:THREE.Object3D[]=[];const textures:THREE.Texture[]=[];const routes:Route[]=[]
    const decorations:{object:THREE.Object3D;mod?:string;l?:number}[]=[]
    const label=(text:string,color:string,width:number,height=0.34)=>{
      const canvas=document.createElement('canvas');canvas.height=72
      const ctx=canvas.getContext('2d')!;ctx.font='500 54px "Segoe UI", sans-serif'
      canvas.width=Math.ceil(ctx.measureText(text).width+24)
      ctx.fillStyle='rgba(6,14,20,0.92)';ctx.fillRect(0,0,canvas.width,72)
      ctx.fillStyle=color;ctx.font='500 54px "Segoe UI", sans-serif';ctx.textAlign='center';ctx.textBaseline='middle';ctx.fillText(text,canvas.width/2,36)
      const texture=new THREE.CanvasTexture(canvas);texture.colorSpace=THREE.SRGBColorSpace;textures.push(texture)
      const sprite=new THREE.Sprite(new THREE.SpriteMaterial({map:texture,transparent:true,depthTest:false}))
      const aspect=canvas.width/72,labelHeight=Math.min(height*1.3,width/aspect)
      sprite.scale.set(labelHeight*aspect,labelHeight,1);sprite.renderOrder=5;return sprite
    }
    const yFor=(l:number)=>2.55+(l-1)*1.67
    const addNode=(id:string,pos:THREE.Vector3,size:[number,number,number],color:string,text?:string)=>{
      const node=NODE_MAP.get(id)!; const group=new THREE.Group();group.position.copy(pos)
      const material=new THREE.MeshStandardMaterial({color:new THREE.Color(color).multiplyScalar(0.25),roughness:0.48,metalness:0.45,transparent:true})
      const box=new THREE.Mesh(new THREE.BoxGeometry(...size),material);group.add(box)
      const edge=new THREE.LineBasicMaterial({color,transparent:true,opacity:0.7})
      group.add(new THREE.LineSegments(new THREE.EdgesGeometry(box.geometry),edge))
      if(text){const sprite=label(text,color,Math.max(size[0],1.65),0.32);sprite.position.set(0,0.02,size[2]/2+0.18);group.add(sprite)}
      group.traverse(obj=>{obj.userData.nodeId=id});hitObjects.push(box)
      const materials:THREE.Material[]=[];group.traverse(obj=>{if('material'in obj)materials.push((obj as THREE.Mesh).material as THREE.Material)})
      content.add(group);visuals.push({group,node,materials,edge});return group
    }
    const addRoute=(points:THREE.Vector3[],color:string,l:number,stage:number,mod?:string)=>{
      const curve=new THREE.CatmullRomCurve3(points)
      const line=new THREE.Line(new THREE.BufferGeometry().setFromPoints(curve.getPoints(36)),new THREE.LineBasicMaterial({color,transparent:true,opacity:0.22}))
      const particle=new THREE.Mesh(new THREE.SphereGeometry(0.065,6,6),new THREE.MeshBasicMaterial({color}))
      content.add(line,particle);routes.push({curve,line,particle,l,stage,mod})
    }
    const v=(x:number,y:number,z:number)=>new THREE.Vector3(x,y,z)
    MODALITIES.forEach(m=>{
      const x=m.x,z=m.z,c=m.color
      const railGeometry=new THREE.BufferGeometry().setFromPoints([v(x,1.9,z),v(x,14.5,z)])
      const rail=new THREE.Line(railGeometry,new THREE.LineBasicMaterial({color:c,transparent:true,opacity:0.2}));content.add(rail);decorations.push({object:rail,mod:m.id})
      const title=label(`${m.label} / ENCODER`,c,3,0.42);title.position.set(x,14.35,z);content.add(title);decorations.push({object:title,mod:m.id})
      addNode(`${m.id}-input`,v(x,1.4,z),[2.45,0.12,1.35],c,`${m.label} · 1 × 192²`)
      addNode(`${m.id}-stem`,v(x,1.95,z),[2.05,0.15,0.9],c)
      addRoute([v(x,1.5,z),v(x,7,z),v(x,13.3,z)],c,0,0,m.id)
      LEVELS.forEach(level=>{
        const y=yFor(level.l),width=2.25-(level.l-1)*0.09
        addNode(`${m.id}-encoder-${level.l}`,v(x,y,z),[width,0.28,0.95],c,`E${level.l} · ${level.feature}ch`)
        // Feature slabs reveal each residual cell's depth without tilting its tower.
        const slab=new THREE.Mesh(new THREE.BoxGeometry(width,0.035,0.95),new THREE.MeshBasicMaterial({color:c,transparent:true,opacity:0.15}));slab.position.set(0,0.23,0)
        visuals[visuals.length-1].group.add(slab);visuals[visuals.length-1].materials.push(slab.material)
        addNode(`${m.id}-expert-${level.l}`,v(x,y-0.5,z+0.65),[1.75,0.16,0.36],c,level.l===7?'FC · μ, a':`Q${level.l} · μ, a`)
        if(level.l<7)addNode(`${m.id}-down-${level.l}`,v(x,y+0.77,z),[0.35,0.15,0.35],c)
        addRoute([v(x,y,z),v(x,y-0.35,z+0.45),v(x,y-0.5,z+0.65)],c,level.l,2,m.id)
        addRoute([v(x,y-0.5,z+0.65),v(x*0.55,y-0.5,z*0.4+0.3),v(0,y-0.12,0.65)],c,level.l,level.l===7?1:2,m.id)
        if(level.l<7)addRoute([v(0,y+0.7,-0.1),v(x*0.55,y+0.48,z-0.8),v(x,y-0.5,z+0.65)],'#769bad',level.l,2,m.id)
      })
      const output=addNode(`${m.id}-output`,v(x,-0.3,z+1),[2.75,0.13,1.9],c,`${m.label} / DECODER`)
      output.children.filter(o=>o instanceof THREE.Sprite).forEach(o=>o.position.z=1.06)
      for(let i=1;i<=6;i++)addNode(`${m.id}-resnet-${i}`,v(x-1.02+(i-1)*0.408,0.05,z+0.7),[0.29,0.3,0.65],c)
      addNode(`${m.id}-image`,v(x,-0.85,z+1.9),[1.6,0.08,0.65],c,'7×7 → 7×7 → x̂')
      addRoute([v(0,yFor(1)-0.35,0.6),v(x*0.45,0.8,z+1),v(x,0.15,z+1)],c,1,3,m.id)
    })
    LEVELS.forEach(level=>{
      const y=yFor(level.l),l=level.l
      addNode(`poe-${l}`,v(0,y,0.5),[2.15,0.3,1.22],'#ebdfb9',`PoE · z${l}`)
      const dims=label(`${level.channels} × ${level.size}²`,'#adc0c5',1.75,0.26);dims.position.set(0,y-0.4,1.05);content.add(dims);decorations.push({object:dims,l})
      addNode(`prior-${l}`,v(-1.5,y,0),[0.52,0.32,0.62],'#78b8ff',l===7?'N(0,I)':`P${l}`)
      addNode(`sample-${l}`,v(1.5,y,0),[0.45,0.4,0.45],'#d8ff45',`z${l}`)
      addRoute([v(-1.5,y,0),v(-0.75,y,0.4),v(0,y,0.5),v(1.5,y,0)],'#d8ff45',l,l===7?1:2)
      if(l<7){
        addNode(`up-${l}`,v(-0.62,y+0.95,-0.5),[0.85,0.16,0.58],'#78b8ff')
        addNode(`decoder-${l}`,v(0.55,y+0.76,-0.22),[1.2,0.2,0.76],'#78b8ff',`D${l} · ×6`)
        addRoute([l===6?v(0,13.65,-0.5):v(1.5,yFor(l+1),0),v(1.85,y+1.25,-0.8),v(-0.62,y+0.95,-0.5),v(0.55,y+0.76,-0.22),v(-1.5,y,0)],'#78b8ff',l,2)
      }
    })
    addNode('lift',v(0,13.65,-0.5),[1.9,0.15,0.9],'#78b8ff','FC → 128 × 3²')
    addRoute([v(1.5,yFor(7),0),v(1.6,13.5,-0.4),v(0,13.65,-0.5)],'#78b8ff',7,1)
    const coreLabel=label('HIERARCHICAL FUSION','#f2e9d8',4.5,0.38);coreLabel.position.set(0,14.5,0);content.add(coreLabel)
    // Orbital calibration follows the topology; the network remains the focal object.
    const floor=new THREE.Group();floor.position.y=-1.5;scene.add(floor)
    ;[4,7.5,11.7].forEach(radius=>{
      const points=Array.from({length:129},(_,i)=>v(Math.cos(i/128*Math.PI*2)*radius,0,Math.sin(i/128*Math.PI*2)*radius*0.57))
      floor.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(points),new THREE.LineBasicMaterial({color:'#274454',transparent:true,opacity:0.5})))
    })
    for(let i=0;i<64;i++){
      const a=i/64*Math.PI*2,r=11.7
      floor.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints([v(Math.cos(a)*r,0,Math.sin(a)*r*0.57),v(Math.cos(a)*(r+0.15),0,Math.sin(a)*(r+0.15)*0.57)]),new THREE.LineBasicMaterial({color:'#476270',transparent:true,opacity:i%4===0?0.65:0.3})))
    }
    let visible=true,dirty=true,frame=0,lastTime=0,elapsed=0,lastSignature='',lastView='',lastReset=-1,lastZoom=1,lastFocus=0
    const draw=()=>{dirty=true}
    controls.addEventListener('change',draw)
    const resize=()=>{
      const w=container.clientWidth,h=container.clientHeight
      renderer.setSize(w,h); const halfHeight=Math.max(9.4,12.1/(w/h))
      camera.left=-halfHeight*w/h;camera.right=halfHeight*w/h;camera.top=halfHeight;camera.bottom=-halfHeight;camera.updateProjectionMatrix();dirty=true
    }
    const observer=new ResizeObserver(resize);observer.observe(container);resize()
    const intersection=new IntersectionObserver(([entry])=>{visible=entry.isIntersecting;dirty=true},{rootMargin:'80px'});intersection.observe(container)
    const raycaster=new THREE.Raycaster();const pointer=new THREE.Vector2();let downX=0,downY=0
    const pick=(event:PointerEvent)=>{const rect=renderer.domElement.getBoundingClientRect();pointer.set((event.clientX-rect.left)/rect.width*2-1,-(event.clientY-rect.top)/rect.height*2+1);raycaster.setFromCamera(pointer,camera);return raycaster.intersectObjects(hitObjects).find(hit=>hit.object.parent?.visible)?.object.userData.nodeId as string|undefined}
    const pointerdown=(e:PointerEvent)=>{downX=e.clientX;downY=e.clientY}
    const pointerup=(e:PointerEvent)=>{if(Math.hypot(e.clientX-downX,e.clientY-downY)<5){const id=pick(e);if(id)latest.current.onSelect(id)}}
    const pointermove=(e:PointerEvent)=>{if(e.buttons)return;const id=pick(e);setHover(id?NODE_MAP.get(id)!.title:'');renderer.domElement.style.cursor=id?'pointer':'grab'}
    const leave=()=>setHover('')
    const keydown=(e:KeyboardEvent)=>{
      const offset=camera.position.clone().sub(controls.target)
      const spherical=new THREE.Spherical().setFromVector3(offset)
      if(e.key==='ArrowLeft')spherical.theta-=0.12
      else if(e.key==='ArrowRight')spherical.theta+=0.12
      else if(e.key==='ArrowUp')spherical.phi=Math.max(0.2,spherical.phi-0.1)
      else if(e.key==='ArrowDown')spherical.phi=Math.min(1.5,spherical.phi+0.1)
      else if(e.key==='+'||e.key==='=')camera.zoom=Math.min(2.8,camera.zoom*1.1)
      else if(e.key==='-')camera.zoom=Math.max(0.65,camera.zoom/1.1)
      else return
      e.preventDefault();camera.position.copy(controls.target).add(new THREE.Vector3().setFromSpherical(spherical));camera.updateProjectionMatrix();controls.update();dirty=true
    }
    renderer.domElement.addEventListener('pointerdown',pointerdown);renderer.domElement.addEventListener('pointerup',pointerup);renderer.domElement.addEventListener('pointermove',pointermove);renderer.domElement.addEventListener('pointerleave',leave);renderer.domElement.addEventListener('keydown',keydown)
    const lost=(e:Event)=>{e.preventDefault();setError(true)};renderer.domElement.addEventListener('webglcontextlost',lost)
    const animate=(now:number)=>{
      frame=requestAnimationFrame(animate)
      if(!visible||document.hidden){lastTime=now;return}
      const p=latest.current,dt=Math.min((now-lastTime)/1000,0.05);lastTime=now
      if(p.playing&&!p.reducedMotion)elapsed+=dt
      const signature=[p.selected,p.observed.join(),p.target,p.level,p.isolate,p.spread,p.stage,p.playing,p.temperature,p.view,p.reset].join('|')
      if(signature!==lastSignature){
        lastSignature=signature;dirty=true
        const selectedNode=NODE_MAP.get(p.selected)
        for(const visual of visuals){
          const {node,group,edge}=visual
          const active=!node.mod||p.observed.includes(node.mod)||['output','resnet','image'].includes(node.kind)
          const focused=!p.isolate||node.l===p.level||node.kind==='input'
          group.visible=focused
          const selected=node.id===p.selected
          const opacity=active?1:0.2
          for(const mat of visual.materials)mat.opacity=mat instanceof THREE.SpriteMaterial?(active?1:0.72):mat instanceof THREE.LineBasicMaterial?(selected?1:0.55)*opacity:opacity
          edge.color.set(selected?'#ffffff':node.mod?MODALITIES.find(m=>m.id===node.mod)!.color:['poe'].includes(node.kind)?'#ebdfb9':['sample'].includes(node.kind)?'#d8ff45':'#78b8ff')
          group.scale.setScalar(selected?1.09:1)
          if(node.mod){const modality=MODALITIES.find(m=>m.id===node.mod)!;group.position.x=modality.x*(1+p.spread*0.12)+(node.kind==='resnet'?(-1.02+(Number(node.id.split('-').at(-1))-1)*0.408):0)}
        }
        for(const decoration of decorations){
          const {object,mod,l}=decoration
          object.visible=!p.isolate||l===undefined||l===p.level
          if(mod){const m=MODALITIES.find(m=>m.id===mod)!;object.position.x=object instanceof THREE.Sprite?m.x*(1+p.spread*0.12):m.x*p.spread*0.12}
        }
        for(const route of routes){
          const active=(!route.mod||p.observed.includes(route.mod)||route.stage===3)&&(!p.isolate||route.l===p.level||route.l===0)
          const highlighted=route.l===p.level||route.stage===p.stage&&p.playing&&(p.stage===0||p.stage===3)
          route.line.visible=active&&p.spread===0
          ;(route.line.material as THREE.LineBasicMaterial).opacity=highlighted?0.5:0.055
          route.particle.visible=active&&p.playing&&route.stage===p.stage&&(route.l===p.level||p.stage===0||p.stage===3)&&!p.reducedMotion&&p.spread===0
        }
        // Exploded towers expose modules; connections return in the aligned view.
        if(selectedNode&&selectedNode.mod&&['output','resnet','image'].includes(selectedNode.kind))dirty=true
      }
      if(lastView!==p.view||lastReset!==p.reset){
        lastView=p.view;lastReset=p.reset;camera.zoom=1;controls.target.set(0,7,0)
        camera.position.set(...(p.view==='front'?[0,7,38]:p.view==='top'?[0,35,12]:[0,15,36]) as [number,number,number]);camera.updateProjectionMatrix();controls.update();dirty=true
      }
      controls.update()
      if(lastZoom!==p.zoom){lastZoom=p.zoom;camera.zoom=p.zoom;camera.updateProjectionMatrix();dirty=true}
      if(lastFocus!==p.focus){lastFocus=p.focus;const visual=visuals.find(v=>v.node.id===p.selected);if(visual){const offset=camera.position.clone().sub(controls.target);controls.target.copy(visual.group.position);camera.position.copy(controls.target).add(offset);camera.zoom=2.4;camera.updateProjectionMatrix();controls.update();dirty=true}}
      if(p.playing&&!p.reducedMotion){routes.forEach((route,i)=>{if(route.particle.visible)route.particle.position.copy(route.curve.getPointAt((elapsed*(route.stage===0?0.16:0.4)+i*0.113)%1))});dirty=true}
      if(dirty){renderer.render(scene,camera);dirty=false}
    }
    frame=requestAnimationFrame(animate)
    return()=>{
      cancelAnimationFrame(frame);observer.disconnect();intersection.disconnect();controls.dispose()
      renderer.domElement.removeEventListener('pointerdown',pointerdown);renderer.domElement.removeEventListener('pointerup',pointerup);renderer.domElement.removeEventListener('pointermove',pointermove);renderer.domElement.removeEventListener('pointerleave',leave);renderer.domElement.removeEventListener('keydown',keydown);renderer.domElement.removeEventListener('webglcontextlost',lost)
      const geometries=new Set<THREE.BufferGeometry>(),materials=new Set<THREE.Material>()
      scene.traverse(obj=>{if('geometry'in obj)geometries.add((obj as THREE.Mesh).geometry);if('material'in obj){const mat=(obj as THREE.Mesh).material;(Array.isArray(mat)?mat:[mat]).forEach(m=>materials.add(m))}})
      geometries.forEach(g=>g.dispose());materials.forEach(m=>m.dispose());textures.forEach(t=>t.dispose());renderer.dispose();renderer.domElement.remove()
    }
  },[])
  return <div className="mm-orbit-host" ref={host}>
    {error&&<div className="mm-webgl-fallback"><h4>当前浏览器无法显示三维场景</h4><p>完整架构仍可阅读：使用下方「模块目录」逐层查看所有 Encoder、Gaussian head、生成块和公式。</p><a href="#mm-module-directory">打开模块目录</a></div>}
    {hover&&!error&&<div className="mm-hover-label">{hover}<span>点击查看内部算子</span></div>}
  </div>
}
