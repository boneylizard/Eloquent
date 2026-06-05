/**
 * Web Push registration for server-side outreach (desktop backend → mobile).
 */

function urlBase64ToUint8Array(base64String) {
  const padding = '='.repeat((4 - (base64String.length % 4)) % 4);
  const base64 = (base64String + padding).replace(/-/g, '+').replace(/_/g, '/');
  const rawData = window.atob(base64);
  const outputArray = new Uint8Array(rawData.length);
  for (let i = 0; i < rawData.length; ++i) {
    outputArray[i] = rawData.charCodeAt(i);
  }
  return outputArray;
}

function applicationServerKeysMatch(existingKey, expectedBuf) {
  if (!existingKey || !expectedBuf) return false;
  const a = existingKey instanceof ArrayBuffer ? new Uint8Array(existingKey) : new Uint8Array(existingKey);
  const b = expectedBuf instanceof Uint8Array ? expectedBuf : new Uint8Array(expectedBuf);
  if (a.byteLength !== b.byteLength) return false;
  for (let i = 0; i < a.length; i++) {
    if (a[i] !== b[i]) return false;
  }
  return true;
}

export async function registerOutreachWebPush(primaryApiUrl) {
  if (typeof window === 'undefined' || !('serviceWorker' in navigator) || !('PushManager' in window)) {
    return;
  }
  if (!primaryApiUrl || Notification.permission !== 'granted') {
    return;
  }
  const reg = await navigator.serviceWorker.register('/sw-outreach.js?v=20260414c', { scope: '/' });
  await reg.update();
  const vapidRes = await fetch(`${primaryApiUrl}/outreach/v1/vapid-public-key`);
  const vapidJson = await vapidRes.json().catch(() => ({}));
  const publicKey = vapidJson.publicKey;
  if (!publicKey) {
    return;
  }
  const keyBuf = urlBase64ToUint8Array(publicKey);
  let sub = await reg.pushManager.getSubscription();
  const sameKey = sub && applicationServerKeysMatch(sub.options?.applicationServerKey, keyBuf);
  if (!sub || !sameKey) {
    if (sub) {
      try {
        await sub.unsubscribe();
      } catch (_) {}
    }
    sub = await reg.pushManager.subscribe({
      userVisibleOnly: true,
      applicationServerKey: keyBuf,
    });
  }
  await fetch(`${primaryApiUrl}/outreach/v1/push/subscribe`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      subscription: sub.toJSON(),
      publicOrigin: window.location.origin,
    }),
  });
}
