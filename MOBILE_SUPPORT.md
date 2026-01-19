# Mobile Support Architecture

This document outlines the technical implementation of mobile support in the application, covering both backend networking and frontend responsiveness.

## Backend Implementation
The backend enables mobile devices on the same local network (LAN) to connect to the server.

### 1. Universal Host Binding
**File:** `launch.py`

By default, servers often listen on `localhost` (`127.0.0.1`), which restricts access to the local machine only. To allow external connections from mobile devices, the server binds to all network interfaces.

- **Mechanism:** The uvicorn server is started with `host="0.0.0.0"`.
- **Effect:** The server listens for incoming requests from any device on the network (e.g., Wi-Fi), not just the host computer.

```python
# Launch configuration in launch.py
main_backend = start_backend(host="0.0.0.0", port=backend_port, ...)
```

### 2. Dynamic IP Discovery
**File:** `launch.py`

The application automatically identifies the correct Local LAN IP address to ensure the frontend can communicate with the backend from a separate device.

- **Mechanism:**  
  1. Uses `socket.gethostbyname_ex` to list available network interfaces.
  2. Pings a public DNS (like `8.8.8.8`) to determine the primary outbound interface (the one connected to the router).
  3. Writes this IP into `frontend/public/ports.json`.
- **Effect:** When you open the app on your phone, the frontend reads `ports.json` and directs API requests to `http://192.168.1.X:8000` instead of `localhost`.

### 3. Permissive CORS Policy
**File:** `backend/app/main.py`

Mobile browsers enforce strict security policies between different origins. Connecting from `192.168.1.X:5173` to `192.168.1.X:8000` is considered a Cross-Origin request.

- **Mechanism:** The `CORSMiddleware` is configured with `allow_origins=["*"]`.
- **Effect:** This explicitly permits the browser on your mobile phone to fetch data from the server, bypassing standard blocking protocols.

### 4. Remote Authentication
**File:** `backend/app/main.py`

Since the server is exposed to the local network, security is maintained via optional password protection.

- **Mechanism:** A custom middleware checks for an `admin_password` in `settings.json`.
- **Effect:** If a password is set, mobile users must authenticate via Basic Auth (or a Bearer token) before accessing the API.

---

## Frontend Implementation
The frontend is built with a "Mobile-First" responsive design philosophy using Tailwind CSS and React.

### 1. Adaptive Layouts & Navigation
**Files:** `App.jsx`, `Sidebar.jsx`, `Navbar.jsx`

The UI adapts to smaller screens by changing navigation patterns and hiding non-essential elements.

- **Sidebar:** 
  - **Desktop:** A persistent, relative column next to the chat.
  - **Mobile:** A fixed, off-canvas drawer that slides in from the left. controlled by `md:hidden` and `translate-x` classes.
- **Navbar:** The top navigation bar includes a hamburger menu button on mobile to access specialized tabs that don't fit on the screen.
- **Viewport Handling:** Flexbox layouts (`min-h-screen flex flex-col`) ensure the chat input remains accessible at the bottom of the screen, even when the virtual keyboard appears.

### 2. Mobile Audio Handling (Silent Unlocker)
**File:** `frontend/src/contexts/AppContext.jsx`

Mobile browsers (especially iOS Safari) strictly block "Autoplay" audio. Audio can *only* stem from a direct user interaction (like a tap).

- **Problem:** TTS (Text-to-Speech) responses arrive seconds *after* the user's input. The browser considers this "too late" to be part of the interaction and blocks the audio.
- **Solution (The Silent Unlock):** 
  - When the user taps "Send", the app immediately creates and resumes the `AudioContext`.
  - It plays a split-second silent buffer.
  - **Effect:** This "wakes up" the audio engine while the user is still touching the screen. Because the engine is now active/unlocked, the TTS audio that arrives later can play successfully without being blocked.

```javascript
// AppContext.jsx
const unlockAudioContext = useCallback(() => {
  if (ctx.state === 'suspended') {
    ctx.resume(); // Resumes Web Audio API on first touch
  }
  // Plays silent buffer to force active state
}, []);
```
