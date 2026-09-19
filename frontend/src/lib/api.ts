declare global {
  interface Window {
    AstrBotPluginPage?: {
      ready: () => Promise<unknown>;
      apiPost: (path: string, body: unknown) => Promise<unknown>;
    };
  }
}
export async function api<T>(path: string, init: RequestInit = {}): Promise<T> {
  const bridge = window.AstrBotPluginPage;
  if (!bridge) throw new Error("请从 AstrBot 的插件页面打开群聊记忆");
  await bridge.ready();
  const result = await bridge.apiPost("api", {
    path, method: init.method || "GET", body: init.body ? JSON.parse(String(init.body)) : {},
  }) as {status?:string; data?:T; error?:string; message?:string};
  if (result?.error || result?.status === "error")
    throw new Error(result.error || result.message || "请求失败");
  return (result?.status === "ok" && "data" in result ? result.data : result) as T;
}

export type SourceMessage = {
  id: string;
  sender_id: string;
  sender_name: string;
  text: string;
  quote: string;
  created: number;
  sent_at?: number | null;
  time?: string;
  time_kind?: string;
  is_source?: boolean;
};
export type Memory = {
  id: string;
  scope: string;
  subject_id: string;
  summary: string;
  kind: string;
  stance: string;
  status: string;
  created: number;
  updated: number;
  expires: number;
  version: number;
  sources?: SourceMessage[];
  context?: SourceMessage[];
  versions?: {
    version: number;
    actor: string;
    snapshot: string;
    created: number;
  }[];
};
export type Scope = { scope: string; messages: number; pending: number };
export type Overview = {
  messages: number;
  memories: number;
  active: number;
  pending: number;
  index_queue: number;
  calls_today: number;
  disk_bytes: number;
  mode: string;
  last_error: string;
  last_cycle: number;
  scopes: Scope[];
  version: string;
};
export type Settings = {
  revision?: string;
  mode: "off" | "active";
  allowed_scopes: string[];
  bot_ids: string[];
  provider_id: string;
  embedding_provider_id: string;
  qdrant_url: string;
  collection: string;
  online_timeout: number;
  extraction_timeout: number;
  batch_size: number;
  batch_age_seconds: number;
  daily_calls: number;
  retention_days: number;
  trace_days: number;
  max_db_mb: number;
  injection_chars: number;
};
export type Recall = {
  reason: string;
  mode: string;
  injection: string;
  elapsed_ms?: number;
  selected: { id: string; text: string }[];
  candidates: {
    id: string;
    action: string;
    reason: string;
    verification?: string;
  }[];
};
export type Trace = {
  id: string;
  scope: string;
  query: string;
  mode: string;
  reason: string;
  data: Recall;
  elapsed_ms: number;
  created: number;
};
export const date = (value: number) =>
  new Date(value * 1000).toLocaleString("zh-CN", { hour12: false });
export const labels: Record<string, string> = {
  active: "生效中",
  pending: "待审核",
  disabled: "已停用",
  shadow: "历史试运行",
  off: "已关闭",
  event: "事件",
  preference: "偏好",
  goal: "目标",
  agreement: "约定",
  self_report: "本人陈述",
  hearsay: "他人转述",
  accept: "相关",
  reject: "不相关",
  needs_source: "需查原文",
};
