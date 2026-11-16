const video = document.getElementById("video");
const placeholder = document.getElementById("placeholder");

const startButton = document.getElementById("start");
const stopButton = document.getElementById("stop");


function showRunning(running) {
    if (running && !video.getAttribute("src")) {
        video.src = "/video/stream";
    } else if (!running) {
        video.removeAttribute("src");
    }
    video.style.display = running ? "block" : "none";
    placeholder.style.display = running ? "none" : "block";

    startButton.style.display = running ? "none" : "block";
    stopButton.style.display = running ? "block" : "none";
}


async function post(url) {
    startButton.disabled = true;
    stopButton.disabled = true;

    try {
        const response = await fetch(url, { method: "POST" });
        if (!response.ok) {
            const body = await response.json().catch(() => ({}));
            alert(body.detail || `Помилка ${response.status}`);
            return false;
        }
        return true;
    } catch (error) {
        alert("Server is unavailable");
        return false;
    } finally {
        startButton.disabled = false;
        stopButton.disabled = false;
    }
}


startButton.onclick = async () => {
    if (await post("/system/start")) {
        showRunning(true);
    }
};


stopButton.onclick = async () => {
    if (await post("/system/stop")) {
        showRunning(false);
    }
};


// Keep the page in sync after EOF or a capture error.
async function refreshStatus() {
    if (startButton.disabled) return;
    try {
        const response = await fetch("/status");
        if (!response.ok) throw new Error("Status failed");
        const status = await response.json();
        showRunning(status.running);
    } catch (error) {
        showRunning(false);
    }
}
refreshStatus();
setInterval(refreshStatus, 2000);
