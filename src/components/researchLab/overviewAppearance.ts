import {MODALITIES,NODE_MAP} from '../mmhvae/model'
import {glyphFor,type Glyph} from '../mmhvae/anatomy'
import type {PaperModel} from './catalog'
import type {Step} from '../workstation/engine'

export type OverviewGlyph=Glyph|'encoder'|'decoder'|'attention'|'fourier'|'loss'
export interface OverviewAppearance {glyph:OverviewGlyph;color:string;role:string;branch?:string}
const roles:Record<string,[OverviewGlyph,string,string]>={
 input:['image','#8ddbe7','输入 / 输出'],image:['image','#8ddbe7','输入 / 输出'],
 encoder:['encoder','#70bddf','编码 / 残差'],resnet:['encoder','#70bddf','编码 / 残差'],network:['encoder','#70bddf','编码 / 残差'],
 decoder:['decoder','#9cb2ef','生成解码'],output:['decoder','#9cb2ef','生成解码'],
 expert:['gaussian','#c4a3ed','概率分布'],prior:['gaussian','#c4a3ed','概率分布'],gaussian:['gaussian','#c4a3ed','概率分布'],
 poe:['poe','#f0d18f','专家融合'],sample:['sample','#d8ff45','潜变量采样'],
 up:['up','#78d5c2','尺度变换'],pool:['pool','#78d5c2','尺度变换'],down:['pool','#78d5c2','尺度变换'],
 attention:['attention','#edacb9','注意力'],fourier:['fourier','#b9acf3','频域变换'],loss:['loss','#e8a581','约束 / 损失'],
 conv:['conv','#81bed6','卷积'],norm:['norm','#92d3bc','归一化'],linear:['linear','#a2bcf2','线性映射'],
 activation:['activation','#c4acee','非线性'],concat:['concat','#e4bd8d','分流 / 汇合'],split:['split','#e4bd8d','分流 / 汇合'],
 sum:['sum','#e2ce98','逐元素运算'],multiply:['multiply','#e2ce98','逐元素运算'],se:['se','#edacb9','通道注意力'],
 tensor:['tensor','#acc4d6','特征张量'],
}
export function overviewAppearance(id:string,steps:Step[],model?:PaperModel):OverviewAppearance{
 const key=id.replace(/^direct:/,''),node=!model?NODE_MAP.get(key):undefined,entry=model?.entries[key];
 let kind:string=node?.kind??entry?.glyph??'tensor';
 if(!node){
  const hint=`${key} ${entry?.title??steps[0]?.title??''}`.toLowerCase();
  if(/attention|注意力|transformer/.test(hint))kind='attention';
  else if(/fft|fourier|频域/.test(hint))kind='fourier';
  else if(/loss|training|constraint|consistency|约束|一致性|损失/.test(hint))kind='loss';
  else if(/sample|sampling|重参数|采样/.test(hint))kind='sample';
  else if(/poe|专家融合/.test(hint))kind='poe';
  else if(/prior|posterior|gaussian|先验|后验|概率/.test(hint))kind='gaussian';
  else if(/mlp|linear|全连接/.test(hint))kind='linear';
  else if(/decoder|decode|生成|解码/.test(hint))kind='decoder';
  else if(/encoder|content|style|resblock|编码|残差/.test(hint))kind='encoder';
 }
 const [glyph,color,role]=roles[kind]??roles[glyphFor(kind)]??roles.tensor;
 const modality=MODALITIES.find(m=>m.id===node?.mod);
 let domain=entry;while(domain?.parent&&!/^domain[12]$/.test(domain.id))domain=model?.entries[domain.parent];
 const pnpDomain=model?.id==='pnp'&&/^domain[12]$/.test(domain?.id??'')?domain!.id:undefined;
 return {glyph,color:modality?.color??entry?.color??(pnpDomain?pnpDomain==='domain1'?'#83cce8':'#c4b0ed':color),role,branch:modality?.label??(pnpDomain?pnpDomain==='domain1'?'T1W':'T2W':undefined)};
}

/** Same silhouettes in the compact HTML key and the 3D scene. */
export const glyphPaths:Partial<Record<OverviewGlyph,string>>={
 image:'M3 4H21V20H3Z M6 16L10 11L14 15L17 10L20 16',
 encoder:'M7 4H19V8H7Z M7 11H19V15H7Z M7 18H19V22H7Z M4 6H2V20H4',
 decoder:'M10 3H14V7H10Z M7 10H17V14H7Z M3 17H21V21H3Z M12 7V10 M12 14V17',
 gaussian:'M2 20C7 20 7 3 12 3S17 20 22 20 M2 21H22',
 poe:'M4 4L12 10L20 4 M12 16V22 M7 13H17 M12 8V18',
 sample:'M12 3L20 12L12 21L4 12Z M2 5C8 1 22 4 22 12',
 up:'M9 3H15V6H9Z M3 18H21V21H3Z M9 8L3 16 M15 8L21 16',
 pool:'M3 3H21V6H3Z M9 18H15V21H9Z M3 8L9 16 M21 8L15 16',
 attention:'M4 5H10V11H4Z M14 13H20V19H14Z M7 11L17 13 M10 8L20 8V13',
 conv:'M3 3H17V17H3Z M7 7H21V21H7Z M7 12H21 M12 7V21',
 linear:'M4 4L20 8 M4 4L20 16 M4 12L20 8 M4 12L20 16 M4 20L20 8 M4 20L20 16',
 norm:'M3 8H21 M3 16H21 M6 4V20 M12 4V20 M18 4V20',
 activation:'M3 3V21H22 M3 16H10L20 5',
 fourier:'M2 12C6 -3 6 27 12 12S18 -3 22 12',
 loss:'M5 3H19V21H5Z M8 12L11 15L17 8',
}
