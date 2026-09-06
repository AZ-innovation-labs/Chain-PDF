// script.js
async function runOfflineOCR(imageElementOrFile) {
  const worker = await Tesseract.createWorker('eng', 1, {
    workerPath: './vendor/worker.min.js',
    corePath: './vendor/tesseract-core-simd-lstm.wasm.js',
    langPath: './tessdata',
    cacheMethod: 'none', // Prevents attempting network checks
    gzip: true            // Set to false if you extracted the .traineddata file
  });

  const ret = await worker.recognize(imageElementOrFile);
  console.log(ret.data.text);

  await worker.terminate();
  return ret.data.text;
}