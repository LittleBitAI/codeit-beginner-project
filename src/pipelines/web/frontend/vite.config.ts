import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// backend는 학습을 실행하므로 127.0.0.1에만 붙습니다. dev server도 같은 곳을 봅니다.
const BACKEND = 'http://127.0.0.1:8000';

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    strictPort: true,
    proxy: {
      '/api': { target: BACKEND, changeOrigin: false },
    },
  },
  build: {
    outDir: 'dist',
    sourcemap: false,
  },
  test: {
    environment: 'jsdom',
    globals: true,
    setupFiles: ['./src/test/setup.ts'],
    css: false,
  },
});
