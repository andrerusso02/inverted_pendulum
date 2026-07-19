import subprocess
import threading
import time

class PingChecker:
    def __init__(self, ip_address, log_func, interval=1.0):
        self.ip_address = ip_address
        self.log_func = log_func
        self.interval = interval
        self._running = False
        self._thread = None
        
        self._ping_ok = False
        self._ping_lock = threading.Lock()

    def start(self):
        if self._thread is not None:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
    
    def block_until_ping_ok(self):
        while self._running:
            if self.ping_ok:
                break
            time.sleep(0.1)
    
    @property
    def ping_ok(self):
        with self._ping_lock:
            return self._ping_ok

    def _run(self):
        while self._running:
            result = subprocess.run(
                ['ping', '-c', '1', '-W', '1', self.ip_address],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            
            if result.returncode != 0:
                with self._ping_lock:
                    self._ping_ok = False
                self.log_func(f"Ping to {self.ip_address} failed.")
            else:
                with self._ping_lock:
                    self._ping_ok = True
            
            # Sleep in small increments to allow responsive shutdown
            for _ in range(int(self.interval * 10)):
                if not self._running:
                    break
                time.sleep(0.1)
