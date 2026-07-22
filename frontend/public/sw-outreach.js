/* global self */

self.addEventListener('push', (event) => {
  let payload = { title: 'Eloquent', body: '', url: '/' };
  try {
    if (event.data) {
      payload = { ...payload, ...event.data.json() };
    }
  } catch (_) {}
  const title = payload.title || 'Eloquent';
  const options = {
    body: payload.body || 'New outreach message',
    data: {
      url: payload.url || '/',
      conversationId: payload.conversationId || null,
      messageId: payload.messageId || null,
    },
    icon: payload.icon || undefined,
    badge: '/favicon.svg',
  };
  event.waitUntil(self.registration.showNotification(title, options));
});

self.addEventListener('notificationclick', (event) => {
  event.notification.close();
  const data = (event.notification && event.notification.data) || {};
  let targetUrl = data.url || '/';
  try {
    const u = new URL(targetUrl, self.registration.scope);
    if (data.conversationId) {
      u.searchParams.set('outreach', '1');
      u.searchParams.set('cid', String(data.conversationId));
      if (data.messageId) u.searchParams.set('mid', String(data.messageId));
      targetUrl = u.toString();
    } else {
      targetUrl = u.toString();
    }
  } catch (_) {}
  event.waitUntil(
    self.clients.matchAll({ type: 'window', includeUncontrolled: true }).then(async (clientList) => {
      if (clientList && clientList.length > 0) {
        const client = clientList.find(c => c && typeof c.url === 'string') || clientList[0];
        try {
          if (client && client.navigate) {
            await client.navigate(targetUrl);
          }
        } catch (_) {}
        try {
          if (client) await client.focus();
        } catch (_) {}
        if (client) return client;
      }
      if (self.clients.openWindow) {
        return self.clients.openWindow(targetUrl);
      }
      return undefined;
    })
  );
});
