
// Custom event to trigger login overlay
export const TRIGGER_LOGIN_EVENT = 'eloquent:trigger-login';

export function setupFetchInterceptor() {
    const originalFetch = window.fetch;
    window.fetch = async (...args) => {
        let [resource, config] = args;

        // 1. Inject Authorization Header
        const savedSettings = localStorage.getItem('Eloquent-settings');
        let password = "";
        if (savedSettings) {
            try {
                const parsed = JSON.parse(savedSettings);
                if (parsed.admin_password) password = parsed.admin_password;
            } catch (e) { }
        }

        if (password) {
            config = config || {};
            config.headers = config.headers || {};
            // Use Headers object if present, else plain object
            if (config.headers instanceof Headers) {
                config.headers.set('Authorization', `Bearer ${password}`);
            } else {
                config.headers['Authorization'] = `Bearer ${password}`;
            }
        }

        // 2. Perform Request
        const response = await originalFetch(resource, config);

        // 3. Catch 401
        if (response.status === 401) {
            console.log("🔒 Authentication required - Triggering Login Overlay via Event");
            // Dispatch custom event that App.jsx listens for
            window.dispatchEvent(new CustomEvent(TRIGGER_LOGIN_EVENT));
        }

        return response;
    };

    console.log("🛡️ Auth Interceptor initialized");
}
