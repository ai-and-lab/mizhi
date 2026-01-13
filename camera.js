export async function startCamera(video) {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: { ideal: "environment" } },
    audio: false,
  });
  video.srcObject = stream;
  await video.play();
  return new Promise((resolve) => {
    if (video.readyState >= 2) {
      resolve(stream);
    } else {
      video.onloadedmetadata = () => resolve(stream);
    }
  });
}

export function captureFrame(video, canvas, ctx) {
  const w = video.videoWidth;
  const h = video.videoHeight;
  if (w === 0 || h === 0) {
    return;
  }
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w;
    canvas.height = h;
  }
  ctx.drawImage(video, 0, 0, w, h);
}
