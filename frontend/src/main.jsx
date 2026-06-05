import React from 'react';

import ReactDOM from 'react-dom/client';

import App from './App';



import { BrowserRouter } from 'react-router-dom'; // Import

import { setupFetchInterceptor } from './utils/auth-interceptor';

import './tv-performance.css';

import { syncTvPerformanceFromUrlAndStorage } from './utils/tvPerformanceMode';

import { scheduleEloquentSplashDismiss } from './utils/eloquentSplash';



syncTvPerformanceFromUrlAndStorage();



// Initialize interceptor immediately

setupFetchInterceptor();



function dismissEloquentSplash() {

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

    <BrowserRouter>

      <App />

    </BrowserRouter>

  </React.StrictMode>

);



void scheduleEloquentSplashDismiss(dismissEloquentSplash);

