import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  // Relative base so the build works from any static host (GitHub Pages,
  // an HF static Space, a subpath) without extra configuration.
  base: './',
})
