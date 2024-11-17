const canvas = document.getElementById('drawingCanvas');
const ctx = canvas.getContext('2d');
let isDrawing = false;

canvas.addEventListener('mousedown', () => {
  isDrawing = true;
  ctx.beginPath();
});

canvas.addEventListener('mouseup', () => {
  isDrawing = false;
});

canvas.addEventListener('mousemove', (event) => {
  if (!isDrawing) return;
  const rect = canvas.getBoundingClientRect();
  const x = event.clientX - rect.left;
  const y = event.clientY - rect.top;
  
  ctx.lineWidth = 2;
  ctx.lineCap = 'round';
  ctx.strokeStyle = 'black';
  
  ctx.lineTo(x, y);
  ctx.stroke();
});

// Temizle butonu
document.getElementById('clearButton').addEventListener('click', () => {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
});

// Tanı butonu
document.getElementById('recognizeButton').addEventListener('click', () => {
  const dataUrl = canvas.toDataURL(); // Canvas'ı base64 formatında al

  // AJAX isteği ile Python modeline gönder
  fetch('http://127.0.0.1:5000/recognize', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ image: dataUrl })
  })
  .then(response => response.json())
  .then(data => {
    // Sonucu alıp, HTML'deki "result" alanına yaz
    document.getElementById('result').innerText = 'Tanınan Rakam: ' + data.prediction;
  })
  .catch(error => {
    console.error('Error:', error);
  });
});
