import { test, expect } from "@playwright/test";

test.beforeEach(async ({page}) => {
  await page.addInitScript(() => {
    window.AstrBotPluginPage = {
      ready: async () => ({}),
      apiPost: async (_path, body) => {
        const response = await fetch('/api/bridge', {
          method:'POST', headers:{'Content-Type':'application/json', Authorization:'Bearer demo-local-only-token-do-not-use-in-production'},
          body:JSON.stringify(body),
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error);
        return data;
      },
    };
  });
});

test("AstrBot bridge, source inspection, edit, preview and settings without token UI", async ({
  page,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.goto("/");
  await expect(page.locator('input[type=password]')).toHaveCount(0);
  await expect(
    page.getByRole("heading", { name: "群聊记忆", exact:true }),
  ).toBeVisible();
  await page
    .getByRole("button")
    .filter({ hasText: "十月参加学校的绘画比赛" })
    .click();
  await expect(page.getByRole("heading", { name: "原始消息与上下文" })).toBeVisible();
  await expect(page.getByText("你说的是十月的那场比赛吗？", { exact: true })).toBeVisible();
  await expect(page.getByText("平台发送时间", { exact: false }).first()).toBeVisible();
  await expect(page.getByText("采集时间（发送时间未知）", { exact: false }).first()).toBeVisible();
  await page
    .getByLabel("记忆摘要")
    .fill("小林自述计划在十月参加绘画比赛，当前尚在准备。");
  await page.getByRole("button", { name: "保存修改" }).click();
  await expect(page.getByText("版本 2", { exact: true })).toBeVisible();
  await page.keyboard.press("Escape");
  await page.getByRole("button", { name: "召回测试" }).click();
  await page.getByLabel("群作用域").selectOption("demo:GroupMessage:10001");
  await page.getByLabel("模拟提问").fill("小林之前说的绘画比赛计划是什么");
  await page.getByRole("button", { name: "开始试召回" }).click();
  await expect(page.getByRole("heading", { name: "拟注入内容" })).toBeVisible();
  await page.getByRole("button", { name: "召回记录", exact: true }).click();
  await expect(
    page.getByText("小林之前说的绘画比赛计划是什么", { exact: true }),
  ).toBeVisible();
  await page.getByRole("button", { name: "设置",exact:true }).click();
  const model = page.getByLabel("辅助模型 Provider ID", { exact: true });
  await expect(page.getByLabel("Qdrant API Key（可选）", { exact: true })).toHaveValue("");
  await expect(model).toHaveJSProperty("tagName", "SELECT");
  await expect(model.locator('option[value="demo-other"]')).toHaveCount(1);
  await model.selectOption("demo-other");
  const toggle = page.getByLabel("插件开关", { exact: true });
  await expect(toggle.locator("option")).toHaveCount(2);
  await expect(toggle.locator('option[value="shadow"]')).toHaveCount(0);
  await expect(toggle).toHaveValue("active");
  await toggle.selectOption("off");
  await page.getByRole("button", { name: "保存设置" }).click();
  await expect(page.getByRole("status")).toHaveText("设置已保存");
  await page.reload();
  await expect(page.locator("header").getByText("已停用", { exact: true })).toBeVisible();
  await page.getByRole("button", { name: "设置", exact: true }).click();
  await expect(model).toHaveValue("demo-other");
  await model.selectOption("demo-only");
  await expect(toggle).toHaveValue("off");
  await toggle.selectOption("active");
  await page.getByRole("button", { name: "保存设置" }).click();
  await expect(page.getByRole("status")).toHaveText("设置已保存");
  expect(errors).toEqual([]);
});

test("stale settings do not overwrite another editor", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("button", { name: "设置", exact: true }).click();
  const budget = page.getByLabel("每日辅助模型调用上限（UTC）", { exact: true });
  await expect(budget).toBeVisible();
  await budget.fill("250");
  const original = await page.evaluate(async () => {
    const bridge = window.AstrBotPluginPage!;
    const cfg = await bridge.apiPost("api", { path: "settings" }) as Record<string, unknown>;
    await bridge.apiPost("api", { path: "settings", method: "PUT", body: { ...cfg, daily_calls: 300 } });
    return cfg.daily_calls;
  });
  await page.getByRole("button", { name: "保存设置", exact: true }).click();
  await expect(page.getByText("请求无效或数据版本冲突，请检查并刷新", { exact: true })).toBeVisible();
  page.once("dialog", d => d.accept());
  await page.getByRole("button", { name: "重新读取配置", exact: true }).click();
  await expect(budget).toHaveValue("300");
  await budget.fill(String(original));
  await page.getByRole("button", { name: "保存设置", exact: true }).click();
  await expect(page.getByRole("status")).toHaveText("设置已保存");
});

test("mobile navigation and screenshot", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/");
  await expect(
    page.getByRole("heading", { name: "群聊记忆",exact:true }),
  ).toBeVisible();
  await expect(
    page.getByRole("button").filter({ hasText: "咖啡" }),
  ).toBeVisible();
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= window.innerWidth,
    ),
  ).toBe(true);
  await page.screenshot({ path: "test-results/mobile.png", fullPage: true });
  await page.setViewportSize({ width: 1440, height: 1000 });
  await page.screenshot({ path: "test-results/desktop.png", fullPage: true });
});

test("disable and delete memory with explicit confirmation", async ({
  page,
}) => {
  await page.goto("/");
  const card = page.getByRole("button").filter({ hasText: "摄影课" });
  await card.click();
  await page.getByLabel("状态", { exact: true }).selectOption("disabled");
  await page.getByRole("button", { name: "保存修改" }).click();
  await expect(page.getByText("版本 2", { exact: true })).toBeVisible();
  page.once("dialog", (d) => d.accept());
  await page.getByRole("button", { name: "永久删除", exact: true }).click();
  await expect(
    page.getByRole("heading", { name: "记忆与来源" }),
  ).not.toBeVisible();
  await expect(card).toHaveCount(0);
});
