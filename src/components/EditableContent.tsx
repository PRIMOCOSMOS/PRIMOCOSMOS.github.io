import { createContext, useContext, useEffect, useLayoutEffect, useRef, useState, type ElementType, type ReactNode } from 'react'
import { Check, Download, Pencil, RotateCcw, Upload } from 'lucide-react'

const STORAGE_KEY = 'primocosmos-note-copy-v1'

type StoredCopy = Record<string, string>

interface EditableContentContextValue {
  editing: boolean
  savedCopy: StoredCopy
  beginEditing: () => void
  finishEditing: () => void
  resetCopy: () => void
  updateCopy: (key: string, value: string) => void
  status: string
  exportCopy: () => void
  importCopy: (file: File) => Promise<void>
}

const EditableContentContext = createContext<EditableContentContextValue | null>(null)

function readStoredCopy(): StoredCopy {
  if (typeof window === 'undefined') return {}
  try {
    const saved = window.localStorage.getItem(STORAGE_KEY)
    const parsed = saved ? JSON.parse(saved) : {}
    return validCopy(parsed) ? parsed : {}
  } catch {
    return {}
  }
}

function validCopy(value: unknown): value is StoredCopy {
  return !!value && typeof value === 'object' && !Array.isArray(value)
    && Object.entries(value).every(([key, text]) => !['__proto__', 'constructor', 'prototype'].includes(key) && key.length < 200 && typeof text === 'string' && text.length <= 100000)
}

export function EditableContentProvider({ children }: { children: ReactNode }) {
  const [editing, setEditing] = useState(false)
  const [savedCopy, setSavedCopy] = useState<StoredCopy>(readStoredCopy)
  const currentCopy = useRef(savedCopy)
  const [status, setStatus] = useState('修改保存在当前浏览器；可导出备份')

  const persist = (next: StoredCopy) => {
    currentCopy.current = next
    setSavedCopy(next)
    try {
      window.localStorage.setItem(STORAGE_KEY, JSON.stringify(next))
      setStatus('已自动保存到当前浏览器')
    } catch {
      setStatus('浏览器存储不可用，修改暂存于本页；请导出备份')
    }
  }

  const updateCopy = (key: string, value: string) => {
    const normalized = value.replace(/\r\n?/g, '\n').replace(/\u00a0/g, ' ')
    if (currentCopy.current[key] !== normalized) persist({ ...currentCopy.current, [key]: normalized })
  }

  const resetCopy = () => {
    if (!window.confirm('恢复所有默认文本？当前浏览器中保存的修改将被清除。')) return
    persist({})
    setEditing(false)
  }

  useEffect(() => {
    const sync = (event: StorageEvent) => {
      if (event.key !== STORAGE_KEY) return
      const copy = readStoredCopy()
      currentCopy.current = copy
      setSavedCopy(copy)
      setStatus('已同步此浏览器其他页面的修改')
    }
    window.addEventListener('storage', sync)
    return () => window.removeEventListener('storage', sync)
  }, [])

  const exportCopy = () => {
    const blob = new Blob([JSON.stringify({ format: 'primocosmos-copy', version: 1, copy: currentCopy.current }, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const anchor = document.createElement('a')
    anchor.href = url
    anchor.download = 'primocosmos-text-backup.json'
    anchor.click()
    window.setTimeout(() => URL.revokeObjectURL(url), 1000)
  }
  const importCopy = async (file: File) => {
    try {
      if (file.size > 2_000_000) throw new Error('备份过大')
      const backup = JSON.parse(await file.text())
      if (backup.format !== 'primocosmos-copy' || backup.version !== 1 || !validCopy(backup.copy)) throw new Error('格式不匹配')
      persist({ ...currentCopy.current, ...backup.copy })
    } catch { setStatus('未导入：请选择有效的本站 JSON 文本备份') }
  }
  const value = {
    editing,
    savedCopy,
    beginEditing: () => setEditing(true),
    finishEditing: () => setEditing(false),
    resetCopy,
    updateCopy,
    status, exportCopy, importCopy,
  }

  return <EditableContentContext.Provider value={value}>{children}</EditableContentContext.Provider>
}

export function useEditableContent() {
  const context = useContext(EditableContentContext)
  if (!context) throw new Error('Editable content must be rendered inside EditableContentProvider')
  return context
}

interface EditableTextProps {
  textKey: string
  children: string
  as?: ElementType
  className?: string
  'aria-live'?: 'polite' | 'assertive' | 'off'
}

export function EditableText({ textKey, children, as: Tag = 'p', className, 'aria-live': ariaLive }: EditableTextProps) {
  const { editing, savedCopy, updateCopy } = useEditableContent()
  const text = savedCopy[textKey] ?? children
  const element = useRef<HTMLElement>(null)
  useLayoutEffect(() => {
    if (element.current && document.activeElement !== element.current && element.current.innerText !== text) element.current.textContent = text
  }, [text, textKey, editing])

  return (
    <Tag
      className={className}
      ref={element}
      data-copy-key={textKey}
      style={{ whiteSpace: 'pre-wrap' }}
      contentEditable={editing}
      role={editing ? 'textbox' : undefined}
      aria-label={editing ? `编辑正文 ${textKey}` : undefined}
      aria-multiline={editing ? true : undefined}
      aria-live={editing ? undefined : ariaLive}
      suppressContentEditableWarning
      data-editable={editing ? 'active' : undefined}
      spellCheck={editing}
      onInput={(event: React.FormEvent<HTMLElement>) => { if (editing) updateCopy(textKey, event.currentTarget.innerText) }}
      onPaste={(event: React.ClipboardEvent<HTMLElement>) => {
        if (!editing) return
        event.preventDefault()
        const selection = window.getSelection()
        if (!selection?.rangeCount) return
        const range = selection.getRangeAt(0)
        if (!event.currentTarget.contains(range.commonAncestorContainer)) return
        range.deleteContents()
        const pasted = document.createTextNode(event.clipboardData.getData('text/plain'))
        range.insertNode(pasted)
        range.setStartAfter(pasted); range.collapse(true)
        selection.removeAllRanges(); selection.addRange(range)
        updateCopy(textKey, event.currentTarget.innerText)
      }}
      onBlur={(event: React.FocusEvent<HTMLElement>) => { if (editing) updateCopy(textKey, event.currentTarget.innerText) }}
    >
      {children}
    </Tag>
  )
}

export function NoteEditToolbar({ compact = false }: { compact?: boolean }) {
  const { editing, savedCopy, beginEditing, finishEditing, resetCopy, status, exportCopy, importCopy } = useEditableContent()
  const fileInput = useRef<HTMLInputElement>(null)
  const hasEdits = Object.keys(savedCopy).length > 0

  return (
    <div className={`note-edit-toolbar${compact ? ' note-edit-toolbar--compact' : ''}`} aria-label={compact ? '全站文本编辑工具' : '笔记文本编辑工具'}>
      <div>
        <span className={editing ? 'edit-status is-editing' : 'edit-status'} />
        <p aria-live="polite">{editing ? `点击虚线正文编辑 · ${status}` : status}</p>
      </div>
      <div className="note-edit-actions">
        {editing ? (
          <button type="button" className="note-edit-primary" onClick={finishEditing}><Check size={14} />保存修改</button>
        ) : (
          <button type="button" className="note-edit-primary" onClick={beginEditing}><Pencil size={14} />编辑正文</button>
        )}
        <button type="button" onClick={resetCopy} disabled={!hasEdits}><RotateCcw size={14} />恢复默认</button>
        <button type="button" onClick={exportCopy}><Download size={14} />导出备份</button>
        <button type="button" onClick={() => fileInput.current?.click()}><Upload size={14} />导入备份</button>
        <input ref={fileInput} type="file" accept=".json,application/json" hidden aria-label="导入文本备份文件" onChange={event => { const file = event.target.files?.[0]; if (file) void importCopy(file); event.target.value = '' }} />
      </div>
    </div>
  )
}
