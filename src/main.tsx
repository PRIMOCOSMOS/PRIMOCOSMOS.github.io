import '@fontsource-variable/tektur'
import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App'
import { EditableContentProvider } from './components/EditableContent'
import './styles.css'

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <EditableContentProvider>
      <App />
    </EditableContentProvider>
  </React.StrictMode>,
)
