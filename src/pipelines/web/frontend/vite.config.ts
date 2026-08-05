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
    rollupOptions: {
      output: {
        // Cognito/AppSync client는 크기가 커서 화면 code와 분리해 browser cache를 재사용합니다.
        manualChunks: { amplify: ['aws-amplify'] },
      },
    },
  },
  test: {
    environment: 'jsdom',
    globals: true,
    setupFiles: ['./src/test/setup.ts'],
    css: false,
  },
});
