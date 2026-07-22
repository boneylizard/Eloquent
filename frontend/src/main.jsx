import React from 'react';

import ReactDOM from 'react-dom/client';

import App from './App';

import BackendGate from './components/BackendGate';
import DevDebugPanel from './components/DevDebugPanel';

function DevDebugPanelControlled() {
  const [open, setOpen] = React.useState(false);
  React.useEffect(() => {
    const onKey = (e) => {
      if (e.ctrlKey && e.shiftKey && (e.key === 'D' || e.key === 'd')) {
        e.preventDefault();
        setOpen((v) => !v);
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);
  return <DevDebugPanel open={open} onClose={() => setOpen(false)} />;
}



import { BrowserRouter } from 'react-router-dom'; // Import

import { setupFetchInterceptor } from './utils/auth-interceptor';
import { installExternalLinkHandler } from './utils/externalLinks';
import { installInterfaceZoom } from './utils/interfaceZoom';

import './tv-performance.css';

import { syncTvPerformanceFromUrlAndStorage } from './utils/tvPerformanceMode';

import { scheduleEloquentSplashDismiss } from './utils/eloquentSplash';



syncTvPerformanceFromUrlAndStorage();



// Initialize interceptor immediately

setupFetchInterceptor();
installExternalLinkHandler();
installInterfaceZoom();



function dismissGEMMASplash() {

  const splash = document.getElementById('eloquent-splash');

  if (!splash || splash.classList.contains('eloquent-splash--out')) return;

  splash.classList.add('eloquent-splash--out');

  const remove = () => splash.remove();

  splash.addEventListener('transitionend', remove, { once: true });

  const fadeMs =

    typeof window !== 'undefined' &&

    window.matchMedia?.('(prefers-reduced-motion: reduce)')?.matches

      ? 200

      : 500;

  window.setTimeout(remove, fadeMs);

}



const root = ReactDOM.createRoot(document.getElementById('root'));

root.render(

  <React.StrictMode>

    <BackendGate>
      <BrowserRouter>

        <App />

      </BrowserRouter>
    </BackendGate>

    <DevDebugPanelControlled />

  </React.StrictMode>

);



void scheduleEloquentSplashDismiss(() => {
  const splash = document.getElementById('eloquent-splash');
  if (splash) {
    splash.classList.add('eloquent-splash--out');
    setTimeout(() => splash.remove(), 450);
  }
});

