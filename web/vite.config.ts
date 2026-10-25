/// <reference types="vitest" />
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

const api = process.env.FOUNDRY_API_URL ?? "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react()],
  server: { port: 5173, proxy: { "/api": api, "/healthz": api } },
  preview: { port: 4173, proxy: { "/api": api } },
  build: {
    rollupOptions: {
      output: {
        manualChunks: (id) =>
          /node_modules\/(react-markdown|remark-|mdast-|micromark|hast-|unified|vfile)/.test(id)
            ? "markdown"
            : /node_modules\/(react|react-dom|react-router|@remix-run|scheduler|@tanstack)\//.test(id)
              ? "react"
              : /node_modules\/framer-motion/.test(id)
                ? "motion"
                : undefined,
      },
    },
  },
  test: {
    environment: "jsdom",
    setupFiles: ["./src/test/setup.ts"],
    css: { modules: { classNameStrategy: "non-scoped" } },
  },
});
