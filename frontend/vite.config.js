import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'node:path';

function releaseModuleInventory() {
  return {
    name: 'mirid-release-module-inventory',
    apply: 'build',
    generateBundle(_options, bundle) {
      const modules = new Set();
      for (const output of Object.values(bundle)) {
        if (output.type !== 'chunk') continue;
        for (const moduleId of Object.keys(output.modules)) {
          const cleanId = moduleId.split('?')[0];
          const relativeId = path.relative(process.cwd(), cleanId).replaceAll('\\', '/');
          if (!relativeId.startsWith('..') && relativeId.startsWith('src/')) modules.add(relativeId);
        }
      }
      this.emitFile({
        type: 'asset',
        fileName: '.vite/module-inventory.json',
        source: `${JSON.stringify({ modules: [...modules].sort() }, null, 2)}\n`,
      });
    },
  };
}

export default defineConfig(({ mode }) => {
  const env = { ...process.env, ...loadEnv(mode, process.cwd(), '') };
  const isSingleGpu = env.VITE_SINGLE_GPU_MODE === 'true';
  const targetPort = isSingleGpu ? 8000 : 8001;
  const includedModules = new Set(
    String(env.VITE_MIRID_INCLUDED_MODULES || '')
      .split(',')
      .map((moduleId) => moduleId.trim())
      .filter(Boolean),
  );

  return {
    plugins: [react(), releaseModuleInventory()],
    define: {
      __MIRID_INCLUDE_ELECTIONS__: JSON.stringify(includedModules.has('elections')),
    },
    resolve: {
      alias: {
        '@': '/src',
      },
    },
    build: {
      manifest: true,
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
  };
});
