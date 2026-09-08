// Bespoke planar glyphs: the wordmark remains sharp at any screen size.
const glyphs:Record<string,{w:number;d:string}>={
 P:{w:62,d:'M0 80V12L12 0H49L62 13V34L49 47H17V80ZM17 15V32H42L46 28V19L42 15Z'},
 R:{w:64,d:'M0 80V12L12 0H49L62 13V34L47 47L64 80H44L28 47H17V80ZM17 15V32H42L46 28V19L42 15Z'},
 I:{w:22,d:'M2 0H20V80H2Z'},
 M:{w:76,d:'M0 80V0H16L38 30L60 0H76V80H59V28L38 55L17 28V80Z'},
 O:{w:66,d:'M13 0H53L66 13V67L53 80H13L0 67V13ZM18 16V64H48V16Z'},
 C:{w:63,d:'M14 0H63L53 16H18V64H53L63 80H14L0 66V14Z'},
 S:{w:64,d:'M13 0H64L52 16H18V30H49L64 44V66L50 80H0L11 64H46V49H14L0 35V13Z'},
 L:{w:60,d:'M0 0H18V63H60L49 80H0Z'},
 N:{w:67,d:'M0 80V0H16L49 49V0H67V80H51L18 31V80Z'},
 D:{w:66,d:'M0 0H45L66 20V60L45 80H0ZM18 16V64H39L48 55V25L39 16Z'},
 E:{w:61,d:'M0 0H61L49 16H18V31H51V47H18V64H61L49 80H0Z'},
 V:{w:68,d:'M0 0H19L34 54L49 0H68L44 80H24Z'},
}
function LetterLine({word,y,x,width}:{word:string;y:number;x:number;width:number}){
 const total=[...word].reduce((n,c)=>n+glyphs[c].w+9,0)-9;let cursor=0
 return <g transform={`translate(${x} ${y}) scale(${width/total} 1)`}>{[...word].map((c,i)=>{const pos=cursor;cursor+=glyphs[c].w+9;return <path key={i} d={glyphs[c].d} fillRule="evenodd" transform={`translate(${pos} 0)`}/>})}</g>
}
export function Sigil(){return <svg viewBox="0 0 100 108" aria-hidden="true"><path fill="currentColor" d="M7 19 36 47 54 0 61 42 94 16 71 79 41 108 26 74ZM28 48 39 75 48 59 54 22 46 57 40 64ZM59 64 48 88 67 71 77 44Z" fillRule="evenodd"/></svg>}
export function Wordmark(){return <svg className="silicon-wordmark" viewBox="-10 -95 1110 330" aria-hidden="true">
 <defs><mask id="silicon-cuts"><rect x="-30" y="-110" width="1150" height="360" fill="white"/><path d="M137 20 168 54M408 123 438 155M725 20 755 52M949 144 980 175" stroke="black" strokeWidth="4"/></mask></defs>
 <g fill="currentColor" mask="url(#silicon-cuts)"><LetterLine word="PRIMOCOSMOS" x={90} y={24} width={950}/><LetterLine word="SILICONDEVINE" x={42} y={140} width={1030}/><path className="silicon-blade" d="M0 142 21 -41 65 15 100 -95 107 28 87 16 89 -10 64 60 38 26 29 121 65 98 45 136Z"/><path d="M0 166 86 104 67 140 28 169 36 190 10 225 13 183Z"/><path d="M1048 39H1067V55H1048ZM1048 74H1067V90H1048Z"/></g>
 <path d="M998 228H1100L1075 234H985Z" fill="currentColor"/>
</svg>}
