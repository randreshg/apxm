export type DependencyType = "Data" | "Control";

export interface GraphNode {
  id: number;
  name: string;
  op: "Ask" | "Think" | "Reason" | "QueryMemory" | "UpdateMemory" | "Invoke" | "Branch" | "Switch" | "WaitAll" | "Merge" | "Fence" | "Plan" | "Reflect" | "Verify" | "Const";
  attributes: Record<string, unknown>;
}

export interface GraphEdge {
  from: number;
  to: number;
  dependency: DependencyType;
}

export interface ApxmGraph {
  name: string;
  nodes: GraphNode[];
  edges: GraphEdge[];
  parameters: Array<{ name: string; type_name: string }>;
  metadata: Record<string, unknown>;
}

export interface ExecuteRequest {
  graph: ApxmGraph;
  args: string[];
  session_id: string;
}

export interface ExecuteResponse {
  content?: string;
  results: Record<string, unknown>;
  stats: {
    executed_nodes: number;
    failed_nodes: number;
    duration_ms: number;
  };
  llm_usage: {
    input_tokens: number;
    output_tokens: number;
    total_requests: number;
  };
}

export interface OpenClawMessageInput {
  sessionKey: string;
  userMessage: string;
  skillMarkdown?: string;
  systemPrompt?: string;
  model?: string;
  tools?: string[];
}

export interface OpenClawMessageOutput {
  content: string;
  raw: ExecuteResponse;
}

export interface StoreFactRequest {
  text: string;
  tags?: string[];
  source?: string;
  session_id?: string;
}

export interface StoreFactResponse {
  id: string;
}

export interface SearchFactsRequest {
  query: string;
  limit?: number;
}

export interface Fact {
  id: string;
  text: string;
  tags: string[];
  source: string;
  session_id?: string;
  created_at: string;
  updated_at: string;
}

export interface FactSearchResult {
  fact: Fact;
  score: number;
}

export interface CapabilityRegistration {
  name: string;
  description: string;
  parameters_schema?: Record<string, unknown>;
  static_response?: unknown;
}
