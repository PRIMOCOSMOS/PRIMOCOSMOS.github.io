import { createContext, useContext, useMemo, useState, type ElementType, type ReactNode } from 'react'
import { Check, Pencil, RotateCcw } from 'lucide-react'

const STORAGE_KEY = 'primocosmos-note-copy-v1'

type StoredCopy = Record<string, string>

interface EditableContentContextValue {
  editing: boolean
  savedCopy: StoredCopy
  beginEditing: () => void
  finishEditing: () => void
  resetCopy: () => void
  updateCopy: (key: string, value: string) => void
}

const EditableContentContext = createContext<EditableContentContextValue | null>(null)

function readStoredCopy(): StoredCopy {
  if (typeof window === 'undefined') return {}
  try {
    const saved = window.localStorage.getItem(STORAGE_KEY)
    return saved ? JSON.parse(saved) as StoredCopy : {}
  } catch {
    return {}
  }
}

export function EditableContentProvider({ children }: { children: ReactNode }) {
  const [editing, setEditing] = useState(false)
  const [savedCopy, setSavedCopy] = useState<StoredCopy>(readStoredCopy)

  const updateCopy = (key: string, value: string) => {
    const normalized = value.replace(/\s+/g, ' ').trim()
    setSavedCopy((current) => {
      const next = { ...current, [key]: normalized }
      window.localStorage.setItem(STORAGE_KEY, JSON.stringify(next))
      return next
    })
  }

  const resetCopy = () => {
    if (!window.confirm('恢复所有默认文本？当前浏览器中保存的修改将被清除。')) return
    window.localStorage.removeItem(STORAGE_KEY)
    setSavedCopy({})
    setEditing(false)
  }

  const value = useMemo(() => ({
    editing,
    savedCopy,
    beginEditing: () => setEditing(true),
    finishEditing: () => setEditing(false),
    resetCopy,
    updateCopy,
  }), [editing, savedCopy])

  return <EditableContentContext.Provider value={value}>{children}</EditableContentContext.Provider>
}

function useEditableContent() {
  const context = useContext(EditableContentContext)
  if (!context) throw new Error('Editable content must be rendered inside EditableContentProvider')
  return context
}

interface EditableTextProps {
  textKey: string
  children: string
  as?: ElementType
  className?: string
}

export function EditableText({ textKey, children, as: Tag = 'p', className }: EditableTextProps) {
  const { editing, savedCopy, updateCopy } = useEditableContent()
  const text = savedCopy[textKey] ?? children

  return (
    <Tag
      className={className}
      contentEditable={editing}
      suppressContentEditableWarning
      data-editable={editing ? 'active' : undefined}
      spellCheck={editing}
      onBlur={(event: React.FocusEvent<HTMLElement>) => updateCopy(textKey, event.currentTarget.innerText)}
    >
      {text}
    </Tag>
  )
}

export function NoteEditToolbar() {
  const { editing, savedCopy, beginEditing, finishEditing, resetCopy } = useEditableContent()
  const hasEdits = Object.keys(savedCopy).length > 0

  return (
    <div className="note-edit-toolbar" aria-label="笔记文本编辑工具">
      <div>
        <span className={editing ? 'edit-status is-editing' : 'edit-status'} />
        <p aria-live="polite">{editing ? '编辑模式：点击带虚线边框的正文即可修改' : hasEdits ? '自定义文本已保存在当前浏览器' : '正文支持站内编辑'}</p>
      </div>
      <div className="note-edit-actions">
        {editing ? (
          <button type="button" className="note-edit-primary" onClick={finishEditing}><Check size={14} />保存修改</button>
        ) : (
          <button type="button" className="note-edit-primary" onClick={beginEditing}><Pencil size={14} />编辑正文</button>
        )}
        <button type="button" onClick={resetCopy} disabled={!hasEdits}><RotateCcw size={14} />恢复默认</button>
      </div>
    </div>
  )
}
