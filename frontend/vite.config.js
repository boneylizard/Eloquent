import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
const isSingleGpu = process.env.VITE_SINGLE_GPU_MODE === 'true';
const targetPort = isSingleGpu ? 8000 : 8001;
export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': '/src',
    },
  },
  server: {
    proxy: {
      '/static': {
        target: `http://127.0.0.1:${targetPort}`,
        changeOrigin: true,
        rewrite: (path) => path,
      },
    },
  },
});
