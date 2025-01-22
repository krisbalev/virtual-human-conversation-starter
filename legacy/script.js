const socket = new WebSocket("ws://localhost:8000");

socket.onopen = function () {
    console.log("Connected to WebSocket server");
};

socket.onmessage = function (event) {
    const message = JSON.parse(event.data);
    console.log("Received from backend:", message);

    if (message.type === "result") {
        const videoUrl = message.url;
        console.log(`Playing video: ${videoUrl}`);

        const videoElement = document.getElementById("response-video");
        videoElement.src = videoUrl;
        videoElement.style.display = "block";
        videoElement.play();
    }
};

socket.onerror = function (error) {
    console.error("WebSocket Error:", error);
};

socket.onclose = function () {
    console.log("WebSocket connection closed");
};
