import type { AnatomyGraph } from './anatomy'
import type { Point3 } from './topology'

/** Containment bounds exclude context references. Ports occupy separate outside rails. */
export function detailLayout(graph:AnatomyGraph,compact=false){
  const internal=graph.parts.filter(p=>!p.role),ids=new Set(internal.map(p=>p.id))
  const depth=new Map(internal.map(p=>[p.id,0])),pending=new Set(ids)
  for(let pass=0;pass<internal.length;pass++)for(const p of internal){
    if(!pending.has(p.id))continue
    const incoming=graph.edges.filter(e=>e.to===p.id&&!e.residual&&ids.has(e.from))
    if(incoming.every(e=>!pending.has(e.from))){depth.set(p.id,incoming.length?1+Math.max(...incoming.map(e=>depth.get(e.from)!)):0);pending.delete(p.id)}
  }
  // The deepest dependency path is the trunk. Sibling computations surround it.
  const downstream=new Map<string,number>()
  const ordered=[...internal].sort((a,b)=>depth.get(b.id)!-depth.get(a.id)!)
  for(const p of ordered){const next=graph.edges.filter(e=>e.from===p.id&&!e.residual&&ids.has(e.to));downstream.set(p.id,next.length?1+Math.max(...next.map(e=>downstream.get(e.to)??0)):0)}
  const spine=new Set<string>();let cursor=[...internal].sort((a,b)=>(downstream.get(b.id)!-downstream.get(a.id)!)||Number(b.id==='input')-Number(a.id==='input'))[0]?.id
  while(cursor&&!spine.has(cursor)){spine.add(cursor);cursor=graph.edges.filter(e=>e.from===cursor&&!e.residual&&ids.has(e.to)).sort((a,b)=>(downstream.get(b.to)??0)-(downstream.get(a.to)??0))[0]?.to}
  const positions=new Map<string,Point3>()
  for(const p of internal){
    const d=depth.get(p.id)!,branches=internal.filter(v=>depth.get(v.id)===d&&!spine.has(v.id)),index=branches.indexOf(p),angle=index*Math.PI*.75+Math.PI/4
    positions.set(p.id,graph.layout==='overview'&&p.position?[...p.position]:p.position?[p.position[0]*1.7,p.position[1]*1.7,p.position[2]*1.7]:spine.has(p.id)?[0,-d*(graph.mathematics?11:4.6),0]:[Math.cos(angle)*7,-d*4.6,Math.sin(angle)*7])
  }
  const values=[...positions.values()]
  const margin=graph.mathematics?[7,5,5]:[3.6,2.6,3.6]
  const min=([0,1,2] as const).map(i=>Math.min(...values.map(v=>v[i]))-margin[i]) as Point3
  const max=([0,1,2] as const).map(i=>Math.max(...values.map(v=>v[i]))+margin[i]) as Point3
  for(const role of ['input','output'] as const){
    const ports=graph.parts.filter(p=>p.role===role)
    const used:number[]=[]
    ports.forEach((p,i)=>{
      const endpoints=graph.edges.filter(e=>role==='input'?e.from===p.id:e.to===p.id).map(e=>positions.get(role==='input'?e.to:e.from)).filter((v):v is Point3=>!!v)
      let y=endpoints.length?endpoints.reduce((sum,v)=>sum+v[1],0)/endpoints.length:(min[1]+max[1])/2
      while(used.some(v=>Math.abs(v-y)<4.6))y-=4.6
      used.push(y)
      positions.set(p.id,[role==='input'?min[0]-(compact?4:6):max[0]+(compact?4:6),y,(min[2]+max[2])/2+(i%2?2:-2)])
    })
  }
  return {positions,min,max,spine:[...spine]}
}
