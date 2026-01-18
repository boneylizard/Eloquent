import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
import { AppProvider } from './contexts/AppContext'; // Named import!
import { ThemeProvider } from "./components/ThemeProvider"; // Named import!
import { BrowserRouter } from 'react-router-dom'; // Import
import { setupFetchInterceptor } from './utils/auth-interceptor';

// Initialize interceptor immediately
setupFetchInterceptor();


const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <BrowserRouter> {/* Wrap your App */}
      <AppProvider>
        <ThemeProvider>
          <App />
        </ThemeProvider>
      </AppProvider>
    </BrowserRouter>
  </React.StrictMode>
);