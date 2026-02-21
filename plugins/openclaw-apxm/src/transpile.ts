import type { ApxmGraph } from "./types.js";

interface Frontmatter {
  name?: string;
  model?: string;
  system_prompt?: string;
  tools?: string[];
}

function parseFrontmatter(markdown: string): { frontmatter: Frontmatter; body: string } {
  const trimmed = markdown.trimStart();
  if (!trimmed.startsWith("---\n")) {
    return { frontmatter: {}, body: markdown };
  }

  const rest = trimmed.slice(4);
  const end = rest.indexOf("\n---\n");
  if (end < 0) {
    return { frontmatter: {}, body: markdown };
  }

  const yaml = rest.slice(0, end);
  const body = rest.slice(end + 5);
  const frontmatter: Frontmatter = {};

  for (const rawLine of yaml.split("\n")) {
    const line = rawLine.trim();
    if (!line || line.startsWith("#")) continue;
    const sep = line.indexOf(":");
    if (sep <= 0) continue;
    const key = line.slice(0, sep).trim();
    const value = line.slice(sep + 1).trim();

    if (key === "tools") {
      const parsed = value
        .replace(/^\[/, "")
        .replace(/\]$/, "")
        .split(",")
        .map((v) => v.trim().replace(/^['\"]|['\"]$/g, ""))
        .filter(Boolean);
      frontmatter.tools = parsed;
    } else if (key === "name") {
      frontmatter.name = value.replace(/^['\"]|['\"]$/g, "");
    } else if (key === "model") {
      frontmatter.model = value.replace(/^['\"]|['\"]$/g, "");
    } else if (key === "system_prompt") {
      frontmatter.system_prompt = value.replace(/^['\"]|['\"]$/g, "");
    }
  }

  return { frontmatter, body };
}

export function transpileSkillToGraph(params: {
  markdown?: string;
  fallbackPrompt: string;
  defaultModel: string;
  defaultSystemPrompt: string;
  defaultTools: string[];
}): { graph: ApxmGraph; inferredTools: string[]; inferredModel: string; inferredSystemPrompt: string } {
  const markdown = params.markdown ?? "";
  const { frontmatter, body } = parseFrontmatter(markdown);

  const template = body.trim() || params.fallbackPrompt;
  const tools = frontmatter.tools?.length ? frontmatter.tools : params.defaultTools;
  const model = frontmatter.model ?? params.defaultModel;
  const systemPrompt = frontmatter.system_prompt ?? params.defaultSystemPrompt;
  const graphName = frontmatter.name ?? "openclaw_skill";

  const graph: ApxmGraph = {
    name: graphName,
    nodes: [
      {
        id: 1,
        name: "main",
        op: "Ask",
        attributes: {
          template_str: template,
          model,
          system_prompt: systemPrompt,
          tools_enabled: tools.length > 0,
          tools
        }
      }
    ],
    edges: [],
    parameters: [{ name: "input", type_name: "str" }],
    metadata: { is_entry: true }
  };

  return {
    graph,
    inferredTools: tools,
    inferredModel: model,
    inferredSystemPrompt: systemPrompt
  };
}
