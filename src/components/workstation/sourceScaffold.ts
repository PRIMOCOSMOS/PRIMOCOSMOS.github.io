import type {Run} from './engine'
import {stratifyScaffold,type ScaffoldNode,type ScaffoldEdge} from './scaffold'
/** Lay out source functions as complete vertical calculation trunks. Independent
 * modality/block lanes share rows; unlike a single flattened list, a branch is
 * not pushed thousands of rows away by another branch's arithmetic details. */
export function layoutSourceScaffold(graph:{nodes:ScaffoldNode[];edges:ScaffoldEdge[];levels:number},run:Run){
 const byId=new Map(graph.nodes.map(n=>[n.tensor.id,n])),unit=new Map<string,string>(),groups=new Map<string,ScaffoldNode[]>();
 const sourceUnit=(n:ScaffoldNode)=>{const s=n.step??run.steps.find(s=>s.inputs.some(t=>t.id===n.tensor.id));const path=s?.settings.ancestors as string[]|undefined;
  if(path&&['encoder','decoder'].includes(path[1])&&path[2]?.includes('/block'))return path[2];
  return String(s?.settings.lane??path?.[1]??'input');};
 for(const n of graph.nodes){const id=sourceUnit(n);unit.set(n.tensor.id,id);groups.set(id,[...groups.get(id)??[],n])}
 const links=[...new Set(graph.edges.map(e=>`${unit.get(e.from)}\n${unit.get(e.to)}`).filter(k=>k.split('\n')[0]!==k.split('\n')[1]))].map(k=>k.split('\n'));
 const depths=new Map([...groups.keys()].map(id=>[id,0]));
 for(let pass=0;pass<groups.size;pass++){let changed=false;for(const [a,b] of links){const d=depths.get(a)!+1;if(depths.get(b)!<d){depths.set(b,d);changed=true}}if(!changed)break}
 const extents=new Map<string,{width:number;height:number;depth:number}>();
 for(const [id,nodes] of groups){const internal=graph.edges.filter(e=>unit.get(e.from)===id&&unit.get(e.to)===id);nodes.forEach(n=>n.level=0);
  for(let pass=0;pass<nodes.length;pass++){let changed=false;for(const e of internal){const a=byId.get(e.from)!,b=byId.get(e.to)!;if(b.level<=a.level){b.level=a.level+1;changed=true}}if(!changed)break}
  const local={nodes,edges:internal,levels:0};stratifyScaffold(local);
  const left=Math.min(...nodes.map(n=>n.position[0]-n.width/2)),right=Math.max(...nodes.map(n=>n.position[0]+n.width/2)),bottom=Math.min(...nodes.map(n=>n.position[1]-(n.height??1)/2)),top=Math.max(...nodes.map(n=>n.position[1]+(n.height??1)/2));
  extents.set(id,{width:right-left,height:top-bottom,depth:Math.max(...nodes.map(n=>n.depth))});
 }
 // Modules remain arranged by dependency depth. A stable per-modality order
 // puts homologous encoders side by side without mixing their tensor elements.
 const levels=[...new Set(depths.values())].sort((a,b)=>a-b);let y=0;
 for(const depth of levels){const ids=[...groups.keys()].filter(id=>depths.get(id)===depth),width=ids.reduce((n,id)=>n+extents.get(id)!.width,0)+(ids.length-1)*12;let x=-width/2;
  for(const id of ids){const size=extents.get(id)!;for(const n of groups.get(id)!){n.position[0]+=x+size.width/2;n.position[1]+=y;n.level+=depth*1000}x+=size.width+12}
  y-=Math.max(...ids.map(id=>extents.get(id)!.height))+9;
 }
 graph.levels=levels.length;
 return graph;
}
