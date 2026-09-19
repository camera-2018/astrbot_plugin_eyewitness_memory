import { defineConfig, devices } from "@playwright/test";
export default defineConfig({
  testDir: "./e2e",
  fullyParallel: false,
  workers: 1,
  use: { baseURL: "http://127.0.0.1:19879", trace: "retain-on-failure" },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
  webServer: {
    command: "cd .. && PYTHONPATH=. uv run python scripts/demo.py --port 19879",
    url: "http://127.0.0.1:19879",
    reuseExistingServer: false,
    timeout: 30000,
  },
});
