export const FRONTIER_PROVIDERS = [
  {
    id: 'openai',
    label: 'OpenAI',
    keySetting: 'openAiApiKey',
    keyUrl: 'https://platform.openai.com/api-keys',
    billingUrl: 'https://platform.openai.com/settings/organization/billing/overview',
    guidance: 'Best when you specifically want OpenAI models or platform controls.',
    placeholder: 'sk-…',
  },
  {
    id: 'anthropic',
    label: 'Anthropic',
    keySetting: 'anthropicApiKey',
    keyUrl: 'https://console.anthropic.com/settings/keys',
    billingUrl: 'https://console.anthropic.com/settings/billing',
    guidance: 'Best when you specifically want Claude through Anthropic’s developer API.',
    placeholder: 'sk-ant-…',
  },
  {
    id: 'gemini',
    label: 'Google Gemini',
    keySetting: 'geminiApiKey',
    keyUrl: 'https://aistudio.google.com/app/apikey',
    billingUrl: 'https://ai.google.dev/gemini-api/docs/billing',
    guidance: 'Google AI Studio can issue a key; paid Gemini API use is governed by Google Cloud billing.',
    placeholder: 'Paste your Gemini API key',
  },
  {
    id: 'mistral',
    label: 'Mistral',
    keySetting: 'mistralApiKey',
    keyUrl: 'https://console.mistral.ai/api-keys/',
    billingUrl: 'https://console.mistral.ai/billing/',
    guidance: 'Direct access to Mistral’s own model catalogue and account controls.',
    placeholder: 'Paste your Mistral API key',
  },
  {
    id: 'xai',
    label: 'xAI',
    keySetting: 'xAiApiKey',
    keyUrl: 'https://console.x.ai/',
    billingUrl: 'https://console.x.ai/',
    guidance: 'Use when you specifically need Grok models through the xAI API.',
    placeholder: 'xai-…',
  },
  {
    id: 'meta',
    label: 'Meta Model API',
    keySetting: 'metaApiKey',
    keyUrl: 'https://developer.meta.com/',
    billingUrl: 'https://developer.meta.com/',
    guidance: 'Preview integration. Availability and account access may remain limited.',
    placeholder: 'Paste your Meta Model API key',
    preview: true,
  },
];

export function getFrontierProvider(providerId) {
  return FRONTIER_PROVIDERS.find((provider) => provider.id === providerId) || FRONTIER_PROVIDERS[0];
}
