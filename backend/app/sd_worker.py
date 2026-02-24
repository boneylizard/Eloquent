import logging
import threading
import traceback
import multiprocessing as mp
from multiprocessing.connection import Listener, Client

from .sd_manager import SDManager

logger = logging.getLogger(__name__)


def _handle_request(sd_manager: SDManager, request: dict):
    cmd = request.get("cmd")
    args = request.get("args") or {}

    if cmd == "load_model":
        return sd_manager.load_model(**args)
    if cmd == "generate_image":
        return sd_manager.generate_image(**args)
    if cmd == "enhance_image_with_adetailer":
        return sd_manager.enhance_image_with_adetailer(**args)
    if cmd == "get_progress":
        return sd_manager.get_progress(args.get("task_id"))
    if cmd == "is_adetailer_available":
        return sd_manager.is_adetailer_available()
    if cmd == "get_status":
        return {"loaded_models": sd_manager.current_model_paths}
    if cmd == "get_adetailer_models":
        return list(sd_manager.adetailer.available_models)
    if cmd == "get_adetailer_directory":
        return str(sd_manager.adetailer.model_directory) if sd_manager.adetailer.model_directory else None
    if cmd == "set_adetailer_directory":
        directory = args.get("directory")
        if directory:
            sd_manager.adetailer.set_model_directory(directory)
        return True
    if cmd == "shutdown":
        raise SystemExit

    raise ValueError(f"Unknown SD worker command: {cmd}")


def _client_thread(conn, sd_manager: SDManager):
    try:
        request = conn.recv()
        result = _handle_request(sd_manager, request)
        conn.send({"ok": True, "result": result})
    except SystemExit:
        conn.send({"ok": True, "result": True})
        raise
    except Exception as exc:
        conn.send({"ok": False, "error": str(exc), "traceback": traceback.format_exc()})
    finally:
        conn.close()


def sd_worker_entry(pipe_conn, authkey: bytes):
    listener = Listener(("127.0.0.1", 0), authkey=authkey)
    pipe_conn.send(listener.address)
    pipe_conn.close()

    sd_manager = SDManager()
    logger.info(f"SD worker listening on {listener.address}")

    while True:
        try:
            conn = listener.accept()
        except Exception as exc:
            logger.error(f"SD worker listener failed: {exc}", exc_info=True)
            break

        thread = threading.Thread(target=_client_thread, args=(conn, sd_manager), daemon=True)
        thread.start()


class SDWorkerClient:
    def __init__(self):
        self._ctx = mp.get_context("spawn")
        self._authkey = mp.current_process().authkey or b"sd-worker"
        self._address = None
        self._process = None
        self._lock = threading.Lock()
        self._ensure_worker()

    def _restart_worker(self):
        """Kill the worker process so the next request gets a fresh one. Use after load_model fails."""
        with self._lock:
            if self._process is not None:
                try:
                    if self._process.is_alive():
                        self._process.terminate()
                        self._process.join(timeout=10)
                        if self._process.is_alive():
                            self._process.kill()
                            self._process.join(timeout=5)
                except Exception as e:
                    logger.warning(f"Error stopping SD worker: {e}")
                self._process = None
                self._address = None
                logger.info("SD worker stopped; next request will start a new process.")

    def _ensure_worker(self):
        with self._lock:
            if self._process and self._process.is_alive() and self._address:
                return

            parent_conn, child_conn = self._ctx.Pipe(duplex=False)
            process = self._ctx.Process(
                target=sd_worker_entry,
                args=(child_conn, self._authkey),
                daemon=True
            )
            process.start()
            address = parent_conn.recv()
            parent_conn.close()

            self._process = process
            self._address = address
            logger.info(f"SD worker started at {address}")

    def _request(self, cmd: str, args=None):
        self._ensure_worker()
        args = args or {}

        last_exc = None
        for attempt in range(2):
            try:
                conn = Client(self._address, authkey=self._authkey)
            except Exception as exc:
                last_exc = exc
                self._ensure_worker()
                if attempt == 1:
                    raise RuntimeError(f"SD worker unavailable: {exc}") from exc
                continue

            try:
                conn.send({"cmd": cmd, "args": args})
                response = conn.recv()
            except (ConnectionResetError, BrokenPipeError, EOFError) as exc:
                last_exc = exc
                try:
                    conn.close()
                except Exception:
                    pass
                try:
                    self._restart_worker()
                except Exception:
                    pass
                self._ensure_worker()
                if attempt == 1:
                    raise RuntimeError(f"SD worker request failed: {exc}") from exc
                continue
            except Exception as exc:
                last_exc = exc
                try:
                    conn.close()
                except Exception:
                    pass
                self._ensure_worker()
                raise RuntimeError(f"SD worker request failed: {exc}") from exc
            finally:
                try:
                    conn.close()
                except Exception:
                    pass

            if not response.get("ok"):
                error = response.get("error", "Unknown worker error")
                # After load_model or generate_image failure the worker's native state may be corrupted; restart so next request gets a fresh process
                if cmd in ("load_model", "generate_image"):
                    try:
                        self._restart_worker()
                    except Exception as restart_exc:
                        logger.warning(f"Failed to restart SD worker after {cmd} error: {restart_exc}")
                raise RuntimeError(error)

            return response.get("result")

        raise RuntimeError(f"SD worker request failed after retry: {last_exc}")

    def load_model(self, model_path: str, gpu_id: int = 0):
        return self._request("load_model", {"model_path": model_path, "gpu_id": gpu_id})

    def generate_image(self, **kwargs):
        return self._request("generate_image", kwargs)

    def enhance_image_with_adetailer(self, **kwargs):
        return self._request("enhance_image_with_adetailer", kwargs)

    def get_progress(self, task_id: str):
        return self._request("get_progress", {"task_id": task_id})

    def is_adetailer_available(self):
        return self._request("is_adetailer_available")

    def get_status(self):
        return self._request("get_status")

    def get_adetailer_models(self):
        return self._request("get_adetailer_models")

    def get_adetailer_directory(self):
        return self._request("get_adetailer_directory")

    def set_adetailer_directory(self, directory: str):
        return self._request("set_adetailer_directory", {"directory": directory})

    def shutdown(self):
        try:
            self._request("shutdown")
        except Exception:
            pass
