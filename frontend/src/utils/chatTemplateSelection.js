export const MODEL_DEFAULT_CHAT_TEMPLATE_ID = 'model-default';

const BUILTIN_CHAT_TEMPLATE_IDS = new Set([
  MODEL_DEFAULT_CHAT_TEMPLATE_ID,
  'generic',
  'chatml',
]);

export function normaliseChatTemplateId(value) {
  const id = typeof value === 'string' ? value.trim() : '';
  if (!id) return MODEL_DEFAULT_CHAT_TEMPLATE_ID;
  if (BUILTIN_CHAT_TEMPLATE_IDS.has(id) || id.startsWith('custom:')) return id;
  return MODEL_DEFAULT_CHAT_TEMPLATE_ID;
}

export function getConversationChatTemplateId(conversations, conversationId) {
  if (!conversationId) return MODEL_DEFAULT_CHAT_TEMPLATE_ID;
  const conversation = (conversations || []).find((item) => item?.id === conversationId);
  return normaliseChatTemplateId(conversation?.chatTemplateId);
}

export function buildChatTemplateMessages(history, systemMessage = '') {
  const result = [];
  const system = String(systemMessage || '').trim();
  if (system) result.push({ role: 'system', content: system });

  for (const message of history || []) {
    if (!message || typeof message.content !== 'string') continue;
    if (message.role === 'system' || message.role === 'developer') {
      if (!system && message.content.trim()) {
        result.push({ role: 'system', content: message.content });
      }
      continue;
    }
    if (message.role !== 'user' && message.role !== 'bot' && message.role !== 'assistant') continue;
    result.push({
      role: message.role === 'user' ? 'user' : 'assistant',
      content: message.content,
    });
  }

  return result;
}

export function getChatTemplateRequestFields({
  conversations,
  conversationId,
  history,
  systemMessage,
  isApi = false,
  customTemplates = {},
}) {
  if (isApi) return {};
  let chatTemplateId = getConversationChatTemplateId(conversations, conversationId);
  if (chatTemplateId.startsWith('custom:')) {
    const customId = chatTemplateId.slice('custom:'.length);
    const template = customTemplates?.[customId]?.template;
    if (!(typeof template === 'string' && template.trim())) {
      chatTemplateId = MODEL_DEFAULT_CHAT_TEMPLATE_ID;
    }
  }
  return {
    chat_template_id: chatTemplateId,
    chat_template_messages: buildChatTemplateMessages(history, systemMessage),
  };
}
