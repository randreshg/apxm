import type { CapabilityRegistration } from "./types.js";

export async function registerCapabilities(baseUrl: string, capabilities: CapabilityRegistration[]): Promise<void> {
  for (const capability of capabilities) {
    const payload = {
      name: capability.name,
      description: capability.description,
      parameters_schema: capability.parameters_schema ?? { type: "object", properties: {} },
      static_response: capability.static_response ?? { ok: true, name: capability.name }
    };

    const response = await fetch(`${baseUrl}/v1/capabilities/register`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(payload)
    });

    if (!response.ok) {
      const text = await response.text();
      throw new Error(`Failed to register capability '${capability.name}': ${response.status} ${text}`);
    }
  }
}
