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
  await expect(page.getByLabel("召回记录类型")).toHaveValue("active");
  await expect(page.getByText("小林之前说的绘画比赛计划是什么", { exact: true })).toHaveCount(0);
  await page.getByLabel("召回记录类型").selectOption("preview");
  await expect(
    page.getByText("小林之前说的绘画比赛计划是什么", { exact: true }),
  ).toBeVisible();
  await page.getByRole("button", { name: "记忆库", exact: true }).click();
  const stats = page.locator("section").filter({hasText:"实聊召回 · 最近 24 小时"});
  await expect(stats).toContainText("0 次检查 · 0 次注入");
  await expect(stats).toContainText("手动测试单列：1 次测试，1 次通过");
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
  const batchSize = page.getByLabel("后台每批送审新消息上限", { exact: true });
  await expect(batchSize).toBeVisible();
  await batchSize.fill("45");
  const original = await page.evaluate(async () => {
    const bridge = window.AstrBotPluginPage!;
    const cfg = await bridge.apiPost("api", { path: "settings" }) as Record<string, unknown>;
    await bridge.apiPost("api", { path: "settings", method: "PUT", body: { ...cfg, batch_size: 50 } });
    return cfg.batch_size;
  });
  await page.getByRole("button", { name: "保存设置", exact: true }).click();
  await expect(page.getByText("请求无效或数据版本冲突，请检查并刷新", { exact: true })).toBeVisible();
  page.once("dialog", d => d.accept());
  await page.getByRole("button", { name: "重新读取配置", exact: true }).click();
  await expect(batchSize).toHaveValue("50");
  await batchSize.fill(String(original));
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

test("failed extraction status remains visible after reload on mobile", async ({ page }) => {
  await page.route("**/api/bridge", async (route) => {
    if (route.request().postDataJSON()?.path !== "overview") return route.continue();
    const response = await route.fetch();
    const data = await response.json();
    await route.fulfill({
      response,
      json: {
        ...data,
        extraction_failed_messages: 80,
        last_extraction_failure: {
          scope: "demo:GroupMessage:10001",
          reason: "后台提取：模型返回空内容或非文本",
          created: 1791314400,
        },
      },
    });
  });
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/");
  const notice = page.getByText("提取失败原文：80 条", { exact: false });
  await expect(notice).toBeVisible();
  await expect(notice).toContainText("不会自动重试");
  await expect(notice).toContainText("后台提取：模型返回空内容或非文本");
  await page.reload();
  await expect(notice).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
  await page.screenshot({ path: "test-results/extraction-failure-mobile.png", fullPage: true });
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

test("manual failed-batch retry requires confirmation and cannot be duplicated", async ({page}) => {
  await page.goto("/");
  await page.getByRole("button", {name:"失败提取", exact:true}).click();
  await expect(page.getByRole("heading", {name:"失败提取批次"})).toBeVisible();
  await expect(page.getByText("后台提取：上游 HTTP 429：额度不足或已耗尽", {exact:true}).last()).toBeVisible();
  const batch = await page.evaluate(async () => {
    const result = await window.AstrBotPluginPage!.apiPost("api", {path:"failed-batches"}) as {items:{id:string;attempted_at:number}[]};
    return result.items[0];
  });
  page.once("dialog", d => d.dismiss());
  await page.getByRole("button", {name:"重新提取", exact:true}).click();
  await expect(page.getByText("1 条原文", {exact:true})).toBeVisible();
  await page.screenshot({path:"test-results/failed-batches.png", fullPage:true});
  page.once("dialog", d => d.accept());
  await page.getByRole("button", {name:"重新提取", exact:true}).click();
  await expect(page.getByRole("status")).toContainText("1 条原文已重新入队");
  await expect(page.getByText("没有失败批次。", {exact:true})).toBeVisible();
  const duplicate = await page.evaluate(async (batch) => {
    try {
      await window.AstrBotPluginPage!.apiPost("api", {path:`failed-batches/${batch.id}/retry`,method:"POST",body:{confirm:"RETRY",attempted_at:batch.attempted_at}});
      return "incorrectly accepted";
    } catch (e) { return (e as Error).message; }
  }, batch);
  expect(duplicate).toContain("状态已变化");
  await page.reload();
  await page.getByRole("button", {name:"失败提取", exact:true}).click();
  await expect(page.getByText("没有失败批次。", {exact:true})).toBeVisible();
});
