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

test("wide-recall settings persist through the same native configuration bridge", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("button", { name: "设置", exact: true }).click();
  const fields = [
    ["问题理解时间上限（秒）", "7"],
    ["补充检索线索上限（0 关闭）", "2"],
    ["送辅助模型选择的候选池上限", "30"],
    ["单轮原文核验候选上限", "8"],
  ];
  for (const [label, value] of fields) await page.getByLabel(label, { exact: true }).fill(value);
  await page.getByRole("button", { name: "保存设置", exact: true }).click();
  await expect(page.getByRole("status")).toHaveText("设置已保存");
  await page.reload();
  await page.getByRole("button", { name: "设置", exact: true }).click();
  for (const [label, value] of fields) await expect(page.getByLabel(label, { exact: true })).toHaveValue(value);
  const persisted = await page.evaluate(async () => window.AstrBotPluginPage!.apiPost("api", { path: "settings" }));
  expect(persisted).toMatchObject({ understanding_timeout: 7, query_expansion_limit: 2, candidate_pool_size: 30, source_review_limit: 8 });
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

test("maintenance settings and same-group aliases persist without dropping existing options", async ({ page }) => {
  await page.goto("/");
  const original = await page.evaluate(async () => window.AstrBotPluginPage!.apiPost("api", { path: "settings" })) as Record<string, unknown>;
  await page.getByRole("button", { name: "设置", exact: true }).click();
  await page.getByLabel("定时整理", { exact: true }).selectOption("true");
  await page.getByLabel("整理间隔（小时）", { exact: true }).fill("24");
  await page.getByLabel("语义复核", { exact: true }).selectOption("true");
  await page.getByLabel("每 24 小时语义复核最多条数", { exact: true }).fill("16");
  await page.getByLabel("单批语义复核超时（秒）", { exact: true }).fill("30");
  await page.getByLabel("同一群的会话映射", { exact: true }).fill("\n  official:GroupMessage:new=demo:GroupMessage:10001  \n\n");
  await expect(page.getByText("映射会复用群记忆，但不会映射用户身份", { exact: false })).toBeVisible();
  await page.getByRole("button", { name: "保存设置", exact: true }).click();
  await expect(page.getByRole("status")).toHaveText("设置已保存");
  await page.reload();
  await page.getByRole("button", { name: "设置", exact: true }).click();
  await expect(page.getByLabel("定时整理", { exact: true })).toHaveValue("true");
  await expect(page.getByLabel("语义复核", { exact: true })).toHaveValue("true");
  await expect(page.getByLabel("每 24 小时语义复核最多条数", { exact: true })).toHaveValue("16");
  await expect(page.getByLabel("同一群的会话映射", { exact: true })).toHaveValue("official:GroupMessage:new=demo:GroupMessage:10001");
  const saved = await page.evaluate(async () => window.AstrBotPluginPage!.apiPost("api", { path: "settings" })) as Record<string, unknown>;
  expect(saved).toMatchObject({ maintenance_enabled: true, maintenance_interval_hours: 24, maintenance_semantic_enabled: true, maintenance_review_limit: 16, maintenance_timeout: 30 });
  expect(saved.json_mode).toEqual(original.json_mode);
  expect(saved.systemone_check_support).toEqual(original.systemone_check_support);
  await page.evaluate(async original => {
    const bridge = window.AstrBotPluginPage!;
    const latest = await bridge.apiPost("api", { path: "settings" }) as Record<string, unknown>;
    await bridge.apiPost("api", { path: "settings", method: "PUT", body: { ...original, revision: latest.revision } });
  }, original);
});

test("maintenance preview is read-only and a started background run disables repeat action", async ({ page }) => {
  let started = false;
  let runCalls = 0;
  await page.route("**/api/bridge", async route => {
    const request = route.request().postDataJSON();
    if (request.path === "maintenance/preview") {
      expect(request.method).toBe("GET");
      return route.fulfill({ json: { scanned: 12, counts: { disabled: 2, demoted_to_pending: 1 }, decisions: [
        { id: "example", code: "expired", reason: "已超过原有有效期；停用且不延长有效期", to_status: "disabled" },
      ] } });
    }
    if (request.path === "maintenance/run") {
      expect(request.method).toBe("POST");
      runCalls += 1; started = true;
      return route.fulfill({ json: { started: true } });
    }
    if (request.path === "overview") {
      const response = await route.fetch();
      const body = await response.json();
      return route.fulfill({ response, json: { ...body, maintenance: {
        enabled: true, interval_hours: 24, next_due: 1893499200, last_success: 1893412800,
        running: started, last_run: { scanned: 20, counts: { disabled: 3, demoted_to_pending: 1 } },
        latest: { semantic: { reviewed: 8, forgotten: 1, uncertain: 2, calls: 1 } },
      } } });
    }
    return route.continue();
  });
  await page.goto("/");
  const card = page.getByRole("region", { name: "记忆定时整理状态" });
  await expect(card).toContainText("最近扫描 20 条 · 停用 3 条 · 退回核验 1 条");
  await expect(card).toContainText("歧义保留 2 条");
  await card.getByRole("button", { name: "预览规则整理" }).click();
  await expect(card).toContainText("预览扫描 12 条：拟停用 2 条，退回核验 1 条");
  await expect(card).toContainText("尚未修改记忆，也未调用模型");
  expect(runCalls).toBe(0);
  page.once("dialog", dialog => dialog.accept());
  await card.getByRole("button", { name: "立即整理", exact: true }).click();
  await expect(card.getByRole("button", { name: "立即整理", exact: true })).toBeDisabled();
  await expect(card).toContainText("整理已在后台开始");
  expect(runCalls).toBe(1);
});

test("maintenance history displays readable reasons instead of internal actor identifiers", async ({ page }) => {
  await page.route("**/api/bridge", async route => {
    const path = route.request().postDataJSON()?.path as string;
    if (!path?.startsWith("memories/")) return route.continue();
    const response = await route.fetch();
    const data = await response.json();
    await route.fulfill({ response, json: { ...data, versions: [
      { version: 3, actor: "scheduled-semantic:superseded:原作者已明确取消这个计划", snapshot: JSON.stringify(data), created: 1893412800 },
      { version: 2, actor: "maintenance:internal-run-id:expired", snapshot: JSON.stringify(data), created: 1893412700 },
      ...data.versions,
    ] } });
  });
  await page.goto("/");
  await page.getByRole("button").filter({ hasText: "咖啡" }).click();
  await expect(page.getByText("定时整理 · 已超过原有有效期", { exact: false })).toBeVisible();
  await expect(page.getByText("定时复核 · 原作者已明确取消这个计划", { exact: false })).toBeVisible();
  await expect(page.getByText("internal-run-id", { exact: false })).toHaveCount(0);
});
