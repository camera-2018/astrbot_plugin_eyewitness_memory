import {
  useCallback,
  useEffect,
  useState,
  type FormEvent,
  type ReactNode,
} from "react";
import {
  Archive,
  ArrowUpRight,
  BookOpen,
  Check,
  ChevronLeft,
  ChevronRight,
  Clock3,
  Database,
  FlaskConical,
  Loader2,
  Search,
  Settings2,
  ShieldCheck,
  Sparkles,
  Trash2,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
  SheetDescription,
} from "@/components/ui/sheet";
import {
  api,
  date,
  labels,
  type Memory,
  type Overview,
  type Settings,
  type Trace,
  type Recall,
} from "@/lib/api";

type Page = "memories" | "traces" | "preview" | "settings";
const navigation: { id: Page; label: string; icon: typeof BookOpen }[] = [
  { id: "memories", label: "记忆库", icon: BookOpen },
  { id: "traces", label: "召回记录", icon: Clock3 },
  { id: "preview", label: "召回测试", icon: FlaskConical },
  { id: "settings", label: "设置", icon: Settings2 },
];
function ErrorNotice({ error }: { error: string }) {
  return error ? (
    <div role="alert" className="error">
      {error}
    </div>
  ) : null;
}
function Empty({ children }: { children: ReactNode }) {
  return (
    <div className="empty">
      <Archive size={30} strokeWidth={1.3} />
      <div className="mt-4 leading-7">{children}</div>
    </div>
  );
}
function Tag({ value }: { value: string }) {
  return (
    <Badge variant="secondary" className="font-normal">
      {labels[value] || value}
    </Badge>
  );
}
function Field({
  title,
  children,
  hint,
}: {
  title: string;
  children: ReactNode;
  hint?: string;
}) {
  return (
    <label className="field">
      <span>{title}</span>
      {children}
      {hint && <small className="subtle">{hint}</small>}
    </label>
  );
}

export default function App() {
  const [page, setPage] = useState<Page>("memories");
  const [overview, setOverview] = useState<Overview | null>(null);
  const [scope, setScope] = useState("");
  const [error, setError] = useState("");
  const refresh = useCallback(async () => {
    try {
      setOverview(await api<Overview>("overview"));
      setError("");
    } catch (e) {
      setError((e as Error).message);
    }
  }, []);
  useEffect(() => {
    void refresh();
    const t = setInterval(() => void refresh(), 15000);
    return () => clearInterval(t);
  }, [refresh]);
  return (
    <div className="shell">
      <main className="main">
        <header className="flex justify-between items-center gap-4 mb-5">
          <h1 className="title">群聊记忆</h1>
          <Badge variant="outline">{overview ? (overview.mode === "active" ? "已启用" : "已停用") : "加载中"}</Badge>
        </header>
        <div className="navigation">
        <nav>
          {navigation.map((n) => (
            <button
              key={n.id}
              className={`nav-link ${page === n.id ? "current" : ""}`}
              onClick={() => setPage(n.id)}
            >
              <n.icon size={18} />
              {n.label}
            </button>
          ))}
        </nav>
        </div>
        <ErrorNotice error={error} />
        {overview && page === "memories" && (
          <div className="grid grid-cols-2 xl:grid-cols-4 gap-4 mt-7">
            {[
              ["记忆总数", overview.memories, BookOpen],
              ["生效记忆", overview.active, Check],
              ["待审核", overview.pending, Clock3],
              ["今日辅助调用", overview.calls_today, Sparkles],
            ].map(([label, value, Icon]) => {
              const I = Icon as typeof BookOpen;
              return (
                <div
                  className="bg-card border rounded-md p-4"
                  key={label as string}
                >
                  <div className="flex justify-between subtle">
                    <span>{label as string}</span>
                    <I size={16} />
                  </div>
                  <div className="text-xl font-medium mt-2 tabular-nums">
                    {value as number}
                    <span className="text-xs text-muted-foreground ml-2">
                      {label === "今日辅助调用" ? "次" : "条"}
                    </span>
                  </div>
                </div>
              );
            })}
          </div>
        )}
        {overview?.last_error && (
          <div className="notice mt-5">最近后台状态：{overview.last_error}</div>
        )}
        {page !== "settings" && (
          <div className="toolbar">
            <select
              aria-label="群作用域"
              className="scope-select"
              value={scope}
              onChange={(e) => setScope(e.target.value)}
            >
              <option value="">
                {page === "preview" ? "请选择试召回群" : "所有已记录的群"}
              </option>
              {overview?.scopes.map((s) => (
                <option key={s.scope} value={s.scope}>
                  {s.scope} · {s.messages} 条原文
                </option>
              ))}
            </select>
            <span className="subtle flex gap-1 items-center">
              <ShieldCheck size={13} />
              检索严格按群隔离
            </span>
          </div>
        )}
        {page === "memories" && <Memories scope={scope} refresh={refresh} />}
        {page === "traces" && <Traces scope={scope} />}
        {page === "preview" && <Preview scope={scope} />}
        {page === "settings" && <SettingsPage refresh={refresh} />}
        <div className="mt-10 pt-5 border-t subtle text-xs flex flex-wrap justify-between gap-3">
          <span className="flex items-center gap-2">
            <Database size={12} />
            SQLite 为权威数据 · Qdrant 可重建索引
          </span>
          {overview && (
            <span>
              {(overview.disk_bytes / 1048576).toFixed(1)} MiB 本地数据 ·{" "}
              {overview.index_queue} 项索引待同步
            </span>
          )}
        </div>
      </main>
    </div>
  );
}

function Memories({
  scope,
  refresh,
}: {
  scope: string;
  refresh: () => Promise<void>;
}) {
  const [items, setItems] = useState<Memory[]>([]),
    [total, setTotal] = useState(0),
    [offset, setOffset] = useState(0),
    [status, setStatus] = useState(""),
    [query, setQuery] = useState(""),
    [error, setError] = useState(""),
    [busy, setBusy] = useState(true),
    [selected, setSelected] = useState<Memory | null>(null);
  const [summary, setSummary] = useState(""),
    [editStatus, setEditStatus] = useState("pending"),
    [saving, setSaving] = useState(false);
  useEffect(() => setOffset(0), [scope, status, query]);
  const load = useCallback(async () => {
    setBusy(true);
    try {
      const data = await api<{ items: Memory[]; total: number }>(
        `memories?${new URLSearchParams({ scope, status, q: query, offset: String(offset) })}`,
      );
      setItems(data.items);
      setTotal(data.total);
      setError("");
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }, [scope, status, query, offset]);
  useEffect(() => {
    const t = setTimeout(() => void load(), 180);
    return () => clearTimeout(t);
  }, [load]);
  async function open(id: string) {
    try {
      const m = await api<Memory>(`memories/${id}`);
      setSelected(m);
      setSummary(m.summary);
      setEditStatus(m.status);
    } catch (e) {
      setError((e as Error).message);
    }
  }
  async function save() {
    if (!selected) return;
    setSaving(true);
    try {
      const m = await api<Memory>(`memories/${selected.id}`, {
        method: "PUT",
        body: JSON.stringify({
          summary,
          status: editStatus,
          version: selected.version,
        }),
      });
      setSelected(m);
      await load();
      await refresh();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setSaving(false);
    }
  }
  async function remove() {
    if (
      !selected ||
      !confirm("永久删除这条记忆及版本记录？对应原文仍按保留策略保存。")
    )
      return;
    setSaving(true);
    try {
      await api(`memories/${selected.id}`, { method: "DELETE" });
      setSelected(null);
      await load();
      await refresh();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setSaving(false);
    }
  }
  return (
    <>
      <div className="flex flex-wrap gap-3 mb-5">
        <div className="relative flex-1 min-w-48">
          <Search
            className="absolute left-3 top-3 text-muted-foreground"
            size={15}
          />
          <Input
            aria-label="搜索记忆"
            className="pl-9 bg-card"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="搜索记忆内容或人物 ID…"
          />
        </div>
        <select
          aria-label="记忆状态"
          className="scope-select"
          value={status}
          onChange={(e) => setStatus(e.target.value)}
        >
          <option value="">所有状态</option>
          <option value="active">生效中</option>
          <option value="pending">待审核</option>
          <option value="disabled">已停用</option>
        </select>
      </div>
      <ErrorNotice error={error} />
      {busy ? (
        <div className="empty">
          <Loader2 className="animate-spin" />
          正在加载记忆…
        </div>
      ) : items.length ? (
        <div className="grid gap-2">
          {items.map((m) => (
            <button
              className="memory-card"
              key={m.id}
              onClick={() => void open(m.id)}
            >
              <div className="flex justify-between">
                <div className="flex gap-2">
                  <Tag value={m.kind} />
                  <Tag value={m.status} />
                  {m.expires < Date.now() / 1000 && <Tag value="已过期" />}
                </div>
                <ArrowUpRight size={15} className="text-muted-foreground" />
              </div>
              <p className="memory-body">{m.summary}</p>
              <div className="subtle flex flex-wrap justify-between gap-2 text-xs">
                <span>
                  人物 {m.subject_id} · {labels[m.stance]}
                </span>
                <span>{date(m.updated)}</span>
              </div>
              <div className="mono text-muted-foreground border-t pt-3 mt-3 truncate">
                {m.scope}
              </div>
            </button>
          ))}
        </div>
      ) : (
        <Empty>
          这里还没有符合条件的记忆。
          <br />
          <span className="text-xs">
            请检查「设置」中的群白名单和辅助模型；新消息达到批次阈值后才会提取记忆。
          </span>
        </Empty>
      )}
      <div className="flex justify-between items-center mt-5 subtle">
        <span>
          共 {total} 条 · 第 {Math.floor(offset / 30) + 1} 页
        </span>
        <div className="flex gap-2">
          <Button
            variant="outline"
            size="sm"
            disabled={offset === 0}
            onClick={() => setOffset(Math.max(0, offset - 30))}
          >
            <ChevronLeft />
            上一页
          </Button>
          <Button
            variant="outline"
            size="sm"
            disabled={offset + 30 >= total}
            onClick={() => setOffset(offset + 30)}
          >
            下一页
            <ChevronRight />
          </Button>
        </div>
      </div>
      <Sheet
        open={!!selected}
        onOpenChange={(v) => {
          if (!v) setSelected(null);
        }}
      >
        <SheetContent className="sm:max-w-xl w-full overflow-y-auto p-6">
          <SheetHeader className="px-0">
            <SheetTitle>记忆与来源</SheetTitle>
            <SheetDescription>
              摘要不是事实本身。修改后请对照原文确认。
            </SheetDescription>
          </SheetHeader>
          {selected && (
            <div className="space-y-6 mt-6">
              <div className="flex gap-2">
                <Tag value={selected.kind} />
                <Tag value={selected.stance} />
                <Tag value={`版本 ${selected.version}`} />
              </div>
              <Field title="记忆摘要">
                <Textarea
                  rows={5}
                  value={summary}
                  maxLength={600}
                  onChange={(e) => setSummary(e.target.value)}
                />
              </Field>
              <Field title="状态">
                <select
                  aria-label="状态"
                  value={editStatus}
                  onChange={(e) => setEditStatus(e.target.value)}
                >
                  <option value="active">生效中（确认内容有来源支持）</option>
                  <option value="pending">待审核</option>
                  <option value="disabled">已停用</option>
                </select>
              </Field>
              <div className="subtle">
                人物：{selected.subject_id}
                <br />
                作用域：{selected.scope}
                <br />
                有效期至：{date(selected.expires)}
              </div>
              <ErrorNotice error={error} />
              <div className="flex justify-between">
                <Button onClick={() => void save()} disabled={saving}>
                  保存修改
                </Button>
                <Button
                  variant="ghost"
                  className="text-destructive"
                  onClick={() => void remove()}
                  disabled={saving}
                >
                  <Trash2 size={14} />
                  永久删除
                </Button>
              </div>
              <section className="border-t pt-5">
                <h3 className="font-medium">
                  原始消息与上下文{" "}
                  <span className="subtle">
                    / {selected.context?.length || 0} 条
                  </span>
                </h3>
                <p className="subtle text-xs mt-2">
                  每个来源前后各取最多 10 条，合并去重后最多 50 条、18,000 字符。
                  仅含本插件已采集且仍保留的消息，可能跨越较长时间；不代表完整群历史。
                </p>
                {selected.context?.length ? (
                  selected.context.map((s) => (
                    <div className="source" key={s.id}>
                      {s.is_source && <Badge variant="outline">直接来源</Badge>}
                      <div className="subtle text-xs">
                        {s.sender_name} · {s.sender_id}
                        <br />
                        {s.time} · {s.time_kind}
                      </div>
                      <p className="mt-2">{s.text}</p>
                      {s.quote && <div className="mt-2 bg-muted rounded p-2 text-xs">
                        支持引用：{s.quote}
                      </div>}
                    </div>
                  ))
                ) : (
                  <p className="subtle mt-4">
                    来源已不可用，不应继续作为已核验记忆。
                  </p>
                )}
              </section>
              <section className="border-t pt-5">
                <h3 className="font-medium mb-3">修改历史</h3>
                {selected.versions?.map((v) => (
                  <details className="py-2 border-b subtle" key={v.version}>
                    <summary>
                      v{v.version} · {v.actor} · {date(v.created)}
                    </summary>
                    <p className="mt-2 whitespace-pre-wrap">
                      {JSON.parse(v.snapshot).summary}
                    </p>
                  </details>
                ))}
              </section>
            </div>
          )}
        </SheetContent>
      </Sheet>
    </>
  );
}

function RecallResult({ result }: { result: Recall }) {
  return (
    <div className="space-y-4">
      <div className="notice">
        {result.reason} · {result.elapsed_ms ?? 0} ms · {result.selected.length}{" "}
        条通过
      </div>
      {result.injection && (
        <div className="memory-card whitespace-pre-wrap text-sm leading-7">
          <h3 className="font-medium mb-3">拟注入内容</h3>
          {result.injection}
        </div>
      )}
      {result.candidates.map((c) => (
        <div className="memory-card" key={c.id}>
          <div className="flex gap-2 items-center">
            <Tag value={c.action} />
            <span className="mono text-muted-foreground">
              {c.id.slice(0, 8)}
            </span>
          </div>
          <p className="subtle mt-3">{c.reason}</p>
          {c.verification && (
            <p className="subtle">原文核验：{c.verification}</p>
          )}
        </div>
      ))}
    </div>
  );
}
function Traces({ scope }: { scope: string }) {
  const [items, setItems] = useState<Trace[]>([]),
    [error, setError] = useState(""),
    [busy, setBusy] = useState(true);
  useEffect(() => {
    setBusy(true);
    api<Trace[]>(`traces?${new URLSearchParams({ scope })}`)
      .then(setItems)
      .catch((e) => setError(e.message))
      .finally(() => setBusy(false));
  }, [scope]);
  return (
    <>
      <ErrorNotice error={error} />
      {busy ? (
        <Empty>正在加载记录…</Empty>
      ) : !items.length ? (
        <Empty>还没有召回记录。自动召回和手动测试都会记录决策。</Empty>
      ) : (
        <div className="space-y-3">
          {items.map((t) => (
            <details className="memory-card" key={t.id}>
              <summary className="cursor-pointer">
                <div className="inline-flex gap-3 items-center">
                  <Tag value={t.mode} />
                  <span>{t.query || "空消息"}</span>
                </div>
                <p className="subtle mt-2">
                  {date(t.created)} · {t.scope} · {t.reason}
                </p>
              </summary>
              <div className="mt-5">
                <RecallResult result={t.data} />
              </div>
            </details>
          ))}
        </div>
      )}
    </>
  );
}
function Preview({ scope }: { scope: string }) {
  const [query, setQuery] = useState(""),
    [sender, setSender] = useState(""),
    [result, setResult] = useState<Recall | null>(null),
    [error, setError] = useState(""),
    [busy, setBusy] = useState(false);
  async function run(e: FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError("");
    try {
      setResult(
        await api<Recall>("preview", {
          method: "POST",
          body: JSON.stringify({ scope, query, sender_id: sender }),
        }),
      );
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }
  return (
    <div className="max-w-3xl space-y-5">
      <form className="memory-card space-y-5" onSubmit={run}>
        <Field title="模拟提问">
          <Textarea
            rows={4}
            placeholder="例如：他上次说的那个项目后来怎么样了？"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            required
            maxLength={1500}
          />
        </Field>
        <Field
          title="提问者 ID（可选）"
          hint="填写稳定的 QQ 用户 ID，不是昵称。"
        >
          <Input value={sender} onChange={(e) => setSender(e.target.value)} />
        </Field>
        <Button disabled={busy || !scope}>
          {busy ? (
            <Loader2 className="animate-spin" />
          ) : (
            <FlaskConical size={16} />
          )}
          开始试召回
        </Button>
      </form>
      <ErrorNotice error={error} />
      {result && <RecallResult result={result} />}
    </div>
  );
}

function SettingsPage({ refresh }: { refresh: () => Promise<void> }) {
  const [cfg, setCfg] = useState<Settings | null>(null),
    [error, setError] = useState(""),
    [saved, setSaved] = useState(false),
    [busy, setBusy] = useState(false);
  const [providers, setProviders] = useState<{
    chat: string[];
    embedding: string[];
  }>({ chat: [], embedding: [] });
  const [eraseScope, setEraseScope] = useState(""),
    [eraseSubject, setEraseSubject] = useState("");
  useEffect(() => {
    api<Settings>("settings")
      .then(setCfg)
      .catch((e) => setError(e.message));
    api<typeof providers>("providers")
      .then(setProviders)
      .catch(() => {});
  }, []);
  function patch<K extends keyof Settings>(k: K, v: Settings[K]) {
    setCfg((c) => (c ? { ...c, [k]: v } : c));
    setSaved(false);
  }
  async function save(e: FormEvent) {
    e.preventDefault();
    if (!cfg) return;
    setBusy(true);
    setError("");
    const cleaned = {
      ...cfg,
      allowed_scopes: cfg.allowed_scopes.map((s) => s.trim()).filter(Boolean),
      bot_ids: cfg.bot_ids.map((s) => s.trim()).filter(Boolean),
    };
    try {
      setCfg(
        await api<Settings>("settings", {
          method: "PUT",
          body: JSON.stringify(cleaned),
        }),
      );
      setSaved(true);
      await refresh();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }
  async function erase() {
    if (
      !eraseScope ||
      !eraseSubject ||
      prompt(
        "将永久删除这个人在该群的记忆、来源及相关记录。输入 DELETE 确认：",
      ) !== "DELETE"
    )
      return;
    try {
      await api("erase-subject", {
        method: "POST",
        body: JSON.stringify({
          scope: eraseScope,
          subject_id: eraseSubject,
          confirm: "DELETE",
        }),
      });
      await refresh();
      alert(
        "本地相关数据已删除；向量删除任务已排队，旧索引无法通过权威数据校验。",
      );
    } catch (e) {
      setError((e as Error).message);
    }
  }
  async function reindex() {
    if (
      !confirm(
        "为全部记忆重建当前向量索引？这会产生 Embedding 调用。请先保存模型设置。",
      )
    )
      return;
    setBusy(true);
    try {
      const result = await api<{ queued: number }>("reindex", {
        method: "POST",
        body: "{}",
      });
      alert(`已排队 ${result.queued} 项；后台会按预算逐步处理。`);
      await refresh();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }
  if (!cfg)
    return (
      <>
        <ErrorNotice error={error} />
        <Empty>正在加载设置…</Empty>
      </>
    );
  return (
    <div className="max-w-4xl mt-7">
      <form onSubmit={save} className="space-y-6">
        <section className="memory-card space-y-5">
          <h2 className="font-medium">运行与作用域</h2>
          <div className="notice">
            启用后自动提取记忆，通过相关性检查和原文核验后用于回复。会产生辅助模型费用；白名单为空时不采集任何群。
          </div>
          <Field title="插件开关">
            <select
              aria-label="插件开关"
              value={cfg.mode}
              onChange={(e) =>
                patch("mode", e.target.value as Settings["mode"])
              }
            >
              <option value="off">关闭：停止采集与召回</option>
              <option value="active">启用：通过核验后追加记忆片段</option>
            </select>
          </Field>
          <Field title="允许记录的群（每行一个完整作用域）">
            <Textarea
              rows={3}
              placeholder="default:GroupMessage:123456"
              value={cfg.allowed_scopes.join("\n")}
              onChange={(e) =>
                patch("allowed_scopes", e.target.value.split("\n"))
              }
            />
          </Field>
          <Field title="其他机器人 ID（每行一个，不采集这些账号）">
            <Textarea
              rows={2}
              value={cfg.bot_ids.join("\n")}
              onChange={(e) => patch("bot_ids", e.target.value.split("\n"))}
            />
          </Field>
        </section>
        <section className="memory-card space-y-5">
          <h2 className="font-medium">模型与索引</h2>
          <div className="grid sm:grid-cols-2 gap-5">
            <Field title="辅助模型 Provider ID">
              <Input
                list="chat-providers"
                value={cfg.provider_id}
                onChange={(e) => patch("provider_id", e.target.value)}
              />
              <datalist id="chat-providers">
                {providers.chat.map((p) => (
                  <option key={p} value={p} />
                ))}
              </datalist>
            </Field>
            <Field title="Embedding Provider ID（可选）">
              <Input
                list="embedding-providers"
                value={cfg.embedding_provider_id}
                onChange={(e) => patch("embedding_provider_id", e.target.value)}
              />
              <datalist id="embedding-providers">
                {providers.embedding.map((p) => (
                  <option key={p} value={p} />
                ))}
              </datalist>
            </Field>
            <Field title="Qdrant URL（留空仅关键词检索）">
              <Input
                placeholder="http://qdrant:6333"
                value={cfg.qdrant_url}
                onChange={(e) => patch("qdrant_url", e.target.value)}
              />
            </Field>
            <Field title="独立 Collection 前缀">
              <Input
                value={cfg.collection}
                onChange={(e) => patch("collection", e.target.value)}
              />
            </Field>
          </div>
          <p className="subtle">
            记忆片段只追加到本轮用户消息并随会话历史保存，不写 system prompt。
            删除插件记忆不会回改已保存的聊天记录；需要彻底清除时还需删除相应 AstrBot 会话。
          </p>
          <p className="subtle">
            凭据继续由 AstrBot
            管理，不在这里填写或显示。向量模型变更使用独立索引代际。
          </p>
          <Button
            type="button"
            variant="outline"
            disabled={busy}
            onClick={() => void reindex()}
          >
            重建向量索引
          </Button>
        </section>
        <section className="memory-card space-y-5">
          <h2 className="font-medium">预算与保留</h2>
          <div className="grid sm:grid-cols-2 gap-5">
            {(
              [
                ["online_timeout", "单次在线时间预算（秒）", 0.5, 30, 0.5],
                ["extraction_timeout", "后台提取超时（秒）", 5, 180, 1],
                ["daily_calls", "每日辅助模型调用上限（UTC）", 0, 10000, 1],
                ["batch_size", "后台每批消息上限", 5, 100, 1],
                ["batch_age_seconds", "未满批次最长等待（秒）", 60, 86400, 1],
                ["retention_days", "未引用原文保留（天）", 1, 365, 1],
                ["trace_days", "召回记录保留（天）", 1, 90, 1],
                ["max_db_mb", "SQLite＋WAL 采集保护线（MiB）", 16, 4096, 1],
                [
                  "injection_chars",
                  "注入字符上限（不等于 tokens）",
                  200,
                  16000,
                  1,
                ],
              ] as const
            ).map(([key, label, min, max, step]) => (
              <Field title={label} key={key}>
                <Input
                  type="number"
                  min={min}
                  max={max}
                  step={step}
                  required
                  value={cfg[key]}
                  onChange={(e) => patch(key, Number(e.target.value))}
                />
              </Field>
            ))}
          </div>
        </section>
        <ErrorNotice error={error} />
        <div className="flex gap-4 items-center">
          <Button disabled={busy}>
            {busy ? <Loader2 className="animate-spin" /> : <Check />}保存设置
          </Button>
          {saved && (
            <span role="status" className="text-sm text-primary">
              设置已保存
            </span>
          )}
        </div>
      </form>
      <section className="memory-card mt-8 space-y-4 border-red-200">
        <h2 className="font-medium text-destructive">删除人物数据</h2>
        <p className="subtle">
          删除指定群内该人物的原文、记忆及关联版本；同时清理可能包含旧内容的召回记录。不可撤销。
        </p>
        <div className="grid sm:grid-cols-2 gap-4">
          <Field title="群作用域">
            <Input
              value={eraseScope}
              onChange={(e) => setEraseScope(e.target.value)}
            />
          </Field>
          <Field title="人物 ID">
            <Input
              value={eraseSubject}
              onChange={(e) => setEraseSubject(e.target.value)}
            />
          </Field>
        </div>
        <Button
          variant="outline"
          type="button"
          className="text-destructive"
          onClick={() => void erase()}
        >
          永久删除此人物数据
        </Button>
      </section>
    </div>
  );
}
