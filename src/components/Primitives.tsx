import type { ReactNode } from 'react'
import { ChevronDown, ExternalLink } from 'lucide-react'
import { EditableText } from './EditableContent'

interface EquationBlockProps {
  id: string
  title: string
  formula: string
  children: ReactNode
  defaultOpen?: boolean
}

export function EquationBlock({ id, title, formula, children, defaultOpen = false }: EquationBlockProps) {
  return (
    <details className="equation-block" id={id} open={defaultOpen}>
      <summary>
        <span>{title}</span>
        <ChevronDown aria-hidden="true" />
      </summary>
      <div className="equation-content">
        <code>{formula}</code>
        {typeof children === 'string'
          ? <EditableText textKey={`${id}-explanation`}>{children}</EditableText>
          : <p>{children}</p>}
      </div>
    </details>
  )
}

interface SourceLinkProps {
  href: string
  children: ReactNode
}

export function SourceLink({ href, children }: SourceLinkProps) {
  return (
    <a className="source-link" href={href} target="_blank" rel="noreferrer">
      {children}
      <ExternalLink size={14} aria-hidden="true" />
    </a>
  )
}

export function SignalKey({ color, children }: { color: string; children: ReactNode }) {
  return (
    <span className="signal-key">
      <span className="signal-key__mark" style={{ '--signal-color': color } as React.CSSProperties} />
      {children}
    </span>
  )
}
