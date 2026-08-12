import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { BrowserRouter } from 'react-router-dom';

import { App } from './App';
import { applyTheme, readTheme } from './design/tokens';
import './design/global.css';

// 색과 글꼴의 단일 출처는 tokens.ts입니다. CSS 변수도 여기서 주입합니다.
// 첫 그림 전에 씌워야 어두운 판으로 한 번 깜빡였다 밝아지는 일이 없습니다.
applyTheme(readTheme());

const container = document.getElementById('root');
if (!container) throw new Error('#root 를 찾지 못했습니다.');

createRoot(container).render(
  <StrictMode>
    <BrowserRouter>
      <App />
    </BrowserRouter>
  </StrictMode>,
);
