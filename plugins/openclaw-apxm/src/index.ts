import { z } from "zod";

import { registerCapabilities } from "./capabilityBridge.js";
import { transpileSkillToGraph } from "./transpile.js";
import type {
  CapabilityRegistration,
  ExecuteRequest,
  ExecuteResponse,
  FactSearchResult,
  OpenClawMessageInput,
  OpenClawMessageOutput,
  SearchFactsRequest,
  StoreFactRequest,
  StoreFactResponse
} from "./types.js";

const executeResponseSchema = z.object({
  content: z.string().optional(),
  results: z.record(z.unknown()),
  stats: z.object({
    executed_nodes: z.number(),
    failed_nodes: z.number(),
    duration_ms: z.number()
  }),
  llm_usage: z.object({
    input_tokens: z.number(),
    output_tokens: z.number(),
    total_requests: z.number()
  })
});

const storeFactResponseSchema = z.object({
  id: z.string()
});

const factSchema = z.object({
  id: z.string(),
  text: z.string(),
  tags: z.array(z.string()),
  source: z.string(),
  session_id: z.string().optional(),
  created_at: z.string(),
  updated_at: z.string()
});

const factSearchResultSchema = z.object({
  fact: factSchema,
  score: z.number()
});

export interface OpenClawApxmAdapterConfig {
  baseUrl: string;
  defaultModel: string;
  defaultSystemPrompt?: string;
  defaultTools?: string[];
}

export class OpenClawApxmAdapter {
  private readonly baseUrl: string;
  private readonly defaultModel: string;
  private readonly defaultSystemPrompt: string;
  private readonly defaultTools: string[];

  constructor(config: OpenClawApxmAdapterConfig) {
    this.baseUrl = config.baseUrl.replace(/\/$/, "");
    this.defaultModel = config.defaultModel;
    this.defaultSystemPrompt = config.defaultSystemPrompt ?? "You are a helpful assistant.";
    this.defaultTools = config.defaultTools ?? ["bash", "read", "write", "search_web"];
  }

  async executeMessage(input: OpenClawMessageInput): Promise<OpenClawMessageOutput> {
    const transpiled = transpileSkillToGraph({
      markdown: input.skillMarkdown,
      fallbackPrompt: input.userMessage,
      defaultModel: input.model ?? this.defaultModel,
      defaultSystemPrompt: input.systemPrompt ?? this.defaultSystemPrompt,
      defaultTools: input.tools ?? this.defaultTools
    });

    const request: ExecuteRequest = {
      graph: transpiled.graph,
      args: [input.userMessage],
      session_id: input.sessionKey
    };

    const response = await fetch(`${this.baseUrl}/v1/execute`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(request)
    });

    if (!response.ok) {
      const text = await response.text();
      throw new Error(`APXM execute failed: ${response.status} ${text}`);
    }

    const json = await response.json();
    const parsed: ExecuteResponse = executeResponseSchema.parse(json);

    return {
      content: parsed.content ?? this.extractBestContent(parsed),
      raw: parsed
    };
  }

  async registerOpenClawTools(capabilities: CapabilityRegistration[]): Promise<void> {
    await registerCapabilities(this.baseUrl, capabilities);
  }

  async storeFact(input: StoreFactRequest): Promise<StoreFactResponse> {
    const response = await fetch(`${this.baseUrl}/v1/memory/facts/store`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        text: input.text,
        tags: input.tags ?? [],
        source: input.source ?? "openclaw",
        session_id: input.session_id
      })
    });
    if (!response.ok) {
      const text = await response.text();
      throw new Error(`APXM store_fact failed: ${response.status} ${text}`);
    }
    return storeFactResponseSchema.parse(await response.json());
  }

  async searchFacts(input: SearchFactsRequest): Promise<FactSearchResult[]> {
    const response = await fetch(`${this.baseUrl}/v1/memory/facts/search`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        query: input.query,
        limit: input.limit ?? 5
      })
    });
    if (!response.ok) {
      const text = await response.text();
      throw new Error(`APXM search_facts failed: ${response.status} ${text}`);
    }
    return z.array(factSearchResultSchema).parse(await response.json());
  }

  async deleteFact(id: string): Promise<void> {
    const response = await fetch(`${this.baseUrl}/v1/memory/facts/delete`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ id })
    });
    if (!response.ok) {
      const text = await response.text();
      throw new Error(`APXM delete_fact failed: ${response.status} ${text}`);
    }
  }

  private extractBestContent(response: ExecuteResponse): string {
    const values = Object.values(response.results);
    for (const value of values) {
      if (typeof value === "string") return value;
    }
    return "";
  }
}

export function createOpenClawApxmAdapter(config: OpenClawApxmAdapterConfig): OpenClawApxmAdapter {
  return new OpenClawApxmAdapter(config);
}
