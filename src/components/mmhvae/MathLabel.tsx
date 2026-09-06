import katex from 'katex'
import { useMemo } from 'react'

const escapeHTML=(text:string)=>text.replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]!))
const symbols:Record<string,string>={'μ':'\\mu ','σ':'\\sigma ','ε':'\\epsilon ','Σ':'\\sum ','Λ':'\\Lambda ','⊙':'\\odot ','×':'\\times ','→':'\\to ','←':'\\leftarrow ','↑':'\\uparrow ','↓':'\\downarrow ','−':'-','²':'^{2}','³':'^{3}','ₗ':'_{l}','ⱼ':'_{j}','ₚ':'_{p}','₁':'_{1}','₂':'_{2}','₃':'_{3}','₄':'_{4}','₅':'_{5}','₆':'_{6}','₇':'_{7}','₀':'_{0}','∈':'\\in ','·':'\\cdot ','∞':'\\infty '}
/** Labels contain trusted model notation, but escape all text and disable KaTeX trust. */
export function labelHTML(value:string,forceMath=false):string {
  return value.split(/([\u3400-\u9fff，。；：、]+)/).map(fragment=>{
    if(!fragment||/[\u3400-\u9fff]/.test(fragment))return escapeHTML(fragment)
    if(!forceMath&&!/[μσεΣΛ⊙×→←²³ₗⱼₚ₀-₇=]|\b[zgpqEPQ][1-7]\b/.test(fragment))return escapeHTML(fragment)
    const tex=fragment.replace(/μp/g,'μₚ').replace(/sp\b/g,'sₚ').replace(/\\/g,'').replace(/[{}$%&#]/g,c=>`\\${c}`).replace(/\b([zgpqEPQ])([1-7])\b/g,'$1_{$2}').replace(/[A-Za-z]{2,}/g,word=>`\\text{${word}}`).replace(/[μσεΣΛ⊙×→←↑↓−²³ₗⱼₚ₀-₇∈·∞]/g,c=>symbols[c]??c)
    return katex.renderToString(tex,{displayMode:false,throwOnError:false,strict:'ignore',trust:false,output:'htmlAndMathml'})
  }).join('')
}
export function MathLabel({value,math=false}:{value:string;math?:boolean}){const html=useMemo(()=>labelHTML(value,math),[value,math]);return <span className="mm-inline-math-label" dangerouslySetInnerHTML={{__html:html}}/>}
