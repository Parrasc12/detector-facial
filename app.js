import { FaceLandmarker, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/+esm";

const MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";
const WASM_URL = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";
const MAX_FACES = 4;
const MAX_CAPTURES = 4;
const RECOGNITION_BASE = "https://cdn.jsdelivr.net/npm/@vladmandic/face-api@1.7.15";
const RECOGNITION_MODEL_BASE = `${RECOGNITION_BASE}/model`;
const DB_NAME = "cyanotype-face-registry";
const DB_VERSION = 1;
const MATCH_DISTANCE_LIMIT = 0.48;
const PERFORMANCE_MODES = {
  economy: { width: 320, height: 240, interval: 500, recognitionInterval: 5000, label: "Economia" },
  balanced: { width: 640, height: 480, interval: 200, recognitionInterval: 3000, label: "Equilibrado" },
  quality: { width: 960, height: 720, interval: 120, recognitionInterval: 2200, label: "Qualidade" }
};
const CONNECTIONS = {
  tessellation: FaceLandmarker.FACE_LANDMARKS_TESSELATION,
  contours: FaceLandmarker.FACE_LANDMARKS_CONTOURS,
  irises: FaceLandmarker.FACE_LANDMARKS_LEFT_IRIS.concat(FaceLandmarker.FACE_LANDMARKS_RIGHT_IRIS)
};

const el = (id) => document.getElementById(id);
const video = el("camera-video");
const photo = el("uploaded-photo");
const viewport = el("stage-viewport");
const overlay = el("overlay-canvas");
const ctx = overlay.getContext("2d");
const state = document.querySelector(".stage-state");
const statusText = el("status-text");
const emptyState = el("empty-state");
const readingEmpty = el("reading-empty");
const readingResults = el("reading-results");
const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");

let landmarker;
let stream;
let lastVideoTime = -1;

let activeSource = null;
let mirrored = true;
let isCyanotype = true;
let photoUrl = null;
let captures = [];
let latestResult = null;
let database;
let people = [];
let storageReady = false;
let storagePersisted = false;
let faceApiPromise;
let faceModelsPromise;
let recognitionEnabled = false;
let recognitionBusy = false;
let registrationBusy = false;
let recognitionCanvas;
let recognitionContext;
let recognizedFaces = [];
let recognitionExpiresAt = 0;
let nextRecognitionAt = 0;
let cameraEpoch = 0;
let inferenceTimer = 0;
let isDetecting = false;
let performanceMode = "balanced";
let adaptiveInterval = PERFORMANCE_MODES.balanced.interval;
let fastAnalysisStreak = 0;
let latestResultVersion = 0;
let renderedResultVersion = 0;
let pendingRenderResult = null;
let lastPaintSignature = "";
let interfaceFrame = 0;
let uiFpsStartedAt = 0;
let videoFrameSample = 0;
let analysisSampleCount = 0;
let analysisSampleAt = 0;


function setStatus(message, kind = "idle") {
  state.dataset.state = kind;
  statusText.textContent = message;
}

function updateResolution(width, height) {
  el("resolution-label").textContent = width && height ? `${width} × ${height} PX` : "— × — PX";
}

function sizeOverlay(width, height) {
  if (!width || !height) return;
  const bounds = viewport.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  lastPaintSignature = "";
  overlay.width = Math.round(bounds.width * ratio);
  overlay.height = Math.round(bounds.height * ratio);
  const scale = Math.min(bounds.width / width, bounds.height / height);
  const offsetX = (bounds.width - width * scale) / 2;
  const offsetY = (bounds.height - height * scale) / 2;
  ctx.setTransform(scale * ratio, 0, 0, scale * ratio, offsetX * ratio, offsetY * ratio);
  updateResolution(width, height);
}

function clearOverlay() {
  ctx.save();
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  ctx.restore();
}

function updateSourceView() {
  const hasMedia = Boolean(activeSource);
  emptyState.hidden = hasMedia;
  video.hidden = activeSource !== "camera";
  photo.hidden = activeSource !== "photo";
  el("capture-button").disabled = !hasMedia || !landmarker || captures.length >= MAX_CAPTURES;
  el("mirror-option").hidden = activeSource !== "camera";
}

function revealStage() {
  if (reducedMotion.matches) return;
  const flash = el("reveal-flash");
  flash.classList.remove("is-active");
  void flash.offsetWidth;
  flash.classList.add("is-active");
}

function playCaptureFlash() {
  if (reducedMotion.matches) return;
  const flash = el("capture-flash");
  flash.classList.remove("is-active");
  void flash.offsetWidth;
  flash.classList.add("is-active");
}

async function createLandmarker(delegate) {
  const vision = await FilesetResolver.forVisionTasks(WASM_URL);
  return FaceLandmarker.createFromOptions(vision, {
    baseOptions: { modelAssetPath: MODEL_URL, delegate },
    runningMode: "VIDEO",
    numFaces: MAX_FACES,
    outputFaceBlendshapes: true,
    outputFacialTransformationMatrixes: false
  });
}

async function initializeDetector() {
  setStatus("Carregando o modelo facial…", "loading");
  try {
    try {
      landmarker = await createLandmarker("GPU");
    } catch (gpuError) {
      console.warn("Não foi possível inicializar o detector com GPU; tentando CPU.", gpuError);
      landmarker = await createLandmarker("CPU");
    }
    el("photo-button").disabled = false;
    el("camera-button").disabled = false;
    setStatus("Detector pronto · aguardando uma imagem", "ready");
  } catch (error) {
    console.error("Falha ao carregar o modelo do detector facial.", error);
    setStatus("Não foi possível carregar o modelo. Verifique sua conexão e recarregue a página.", "error");
    el("camera-button").disabled = true;
    el("photo-button").disabled = true;
  }
}

function openPeopleDatabase() {
  return new Promise((resolve, reject) => {
    if (!("indexedDB" in window)) {
      reject(new Error("IndexedDB não está disponível neste navegador."));
      return;
    }
    const request = indexedDB.open(DB_NAME, DB_VERSION);
    request.onupgradeneeded = () => {
      if (!request.result.objectStoreNames.contains("people")) request.result.createObjectStore("people", { keyPath: "id" });
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error || new Error("Não foi possível abrir o armazenamento local."));
    request.onblocked = () => reject(new Error("O armazenamento local está bloqueado por outra aba do site."));
  });
}

function databaseRequest(mode, action, value) {
  return new Promise((resolve, reject) => {
    if (!database) {
      reject(new Error("O armazenamento local não está disponível."));
      return;
    }
    const transaction = database.transaction("people", mode);
    const store = transaction.objectStore("people");
    const request = action === "getAll" ? store.getAll() : action === "put" ? store.put(value) : action === "delete" ? store.delete(value) : store.get(value);
    let result;
    request.onsuccess = () => { result = request.result; };
    request.onerror = () => reject(request.error || new Error("Falha no armazenamento local."));
    transaction.oncomplete = () => resolve(result);
    transaction.onerror = () => reject(transaction.error || new Error("Falha na transação do armazenamento local."));
    transaction.onabort = () => reject(transaction.error || new Error("A gravação local foi cancelada."));
  });
}

async function initializePeopleStorage() {
  const notice = el("storage-notice");
  try {
    database = await openPeopleDatabase();
    people = await databaseRequest("readonly", "getAll");
    storageReady = true;
    try {
      storagePersisted = Boolean(await navigator.storage?.persisted?.());
      if (!storagePersisted && navigator.storage?.persist) storagePersisted = await navigator.storage.persist();
    } catch (error) {
      console.warn("Não foi possível confirmar armazenamento persistente.", error);
    }
    notice.dataset.state = storagePersisted ? "ready" : "warning";
    notice.textContent = storagePersisted
      ? "Cadastro local persistente ativo. Nome e descritores biométricos ficam neste navegador; as fotos não são guardadas."
      : "O navegador não confirmou armazenamento persistente; os cadastros podem ser apagados automaticamente. Os descritores são dados biométricos sensíveis.";
    el("register-person").disabled = false;
    renderPeople();
  } catch (error) {
    console.error("O cadastro biométrico não pode ser salvo.", error);
    notice.dataset.state = "error";
    notice.textContent = `Armazenamento persistente indisponível: ${error.message} Nenhum cadastro será salvo.`;
    el("register-person").disabled = true;
    storageReady = false;
  }
}

function loadFaceApi() {
  if (window.faceapi) return Promise.resolve(window.faceapi);
  if (faceApiPromise) return faceApiPromise;
  faceApiPromise = new Promise((resolve, reject) => {
    const script = document.createElement("script");
    script.src = `${RECOGNITION_BASE}/dist/face-api.js`;
    script.async = true;
    script.onload = () => window.faceapi ? resolve(window.faceapi) : reject(new Error("A biblioteca de reconhecimento não foi inicializada."));
    script.onerror = () => reject(new Error("Não foi possível carregar a biblioteca de reconhecimento facial."));
    document.head.append(script);
  });
  return faceApiPromise;
}

async function loadFaceRecognitionModels() {
  if (faceModelsPromise) return faceModelsPromise;
  faceModelsPromise = (async () => {
    const faceapi = await loadFaceApi();
    const cpuReady = await faceapi.tf.setBackend("cpu");
    if (!cpuReady) throw new Error("O backend CPU do reconhecimento não pôde ser inicializado.");
    await faceapi.tf.ready();
    await Promise.all([
      faceapi.nets.tinyFaceDetector.loadFromUri(RECOGNITION_MODEL_BASE),
      faceapi.nets.faceLandmark68TinyNet.loadFromUri(RECOGNITION_MODEL_BASE),
      faceapi.nets.faceRecognitionNet.loadFromUri(RECOGNITION_MODEL_BASE)
    ]);
    return faceapi;
  })().catch((error) => {
    faceModelsPromise = null;
    throw error;
  });
  return faceModelsPromise;
}

async function extractFaceDescriptors(image) {
  const faceapi = await loadFaceRecognitionModels();
  return faceapi.detectAllFaces(image, new faceapi.TinyFaceDetectorOptions({ inputSize: 160, scoreThreshold: .35 }))
    .withFaceLandmarks(true)
    .withFaceDescriptors();
}

function setRecognitionStatus(message, kind = "idle") {
  const status = el("recognition-state");
  status.dataset.state = kind;
  status.textContent = message;
}

function updateRecognitionControls() {
  const button = el("recognition-toggle");
  button.disabled = !people.length || !storageReady;
  updateCameraEnrollmentControl();
  button.textContent = recognitionEnabled ? "Desativar reconhecimento" : "Ativar reconhecimento";
  button.setAttribute("aria-pressed", String(recognitionEnabled));
  if (!people.length) setRecognitionStatus("Cadastre uma pessoa para reconhecer", "idle");
  else if (!recognitionEnabled) setRecognitionStatus("Reconhecimento desligado", "idle");
}

function renderPeople() {
  const list = el("registry-list");
  const count = people.length;
  el("registry-count").textContent = `${count} ${count === 1 ? "pessoa" : "pessoas"}`;
  list.replaceChildren();
  if (!count) {
    const empty = document.createElement("p");
    empty.className = "registry-empty";
    empty.textContent = "Nenhuma pessoa cadastrada.";
    list.append(empty);
    updateRecognitionControls();
    return;
  }
  for (const person of people) {
    const card = document.createElement("article");
    card.className = "person-card";
    const avatar = document.createElement("span");
    avatar.className = "person-avatar";
    avatar.setAttribute("aria-hidden", "true");
    avatar.textContent = person.name.trim().slice(0, 1).toUpperCase();
    const details = document.createElement("div");
    details.className = "person-details";
    const name = document.createElement("div");
    name.className = "person-name";
    name.textContent = person.name;
    const meta = document.createElement("div");
    meta.className = "person-meta";
    meta.textContent = `${person.descriptors.length} ${person.descriptors.length === 1 ? "descritor facial" : "descritores faciais"} armazenados localmente`;
    details.append(name, meta);
    const actions = document.createElement("div");
    actions.className = "person-actions";
    const rename = document.createElement("button");
    rename.type = "button";
    rename.className = "button button-secondary";
    rename.textContent = "Renomear";
    rename.setAttribute("aria-label", `Renomear ${person.name}`);
    rename.addEventListener("click", () => renamePerson(person));
    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "button button-danger";
    remove.textContent = "Excluir";
    remove.setAttribute("aria-label", `Excluir cadastro de ${person.name}`);
    remove.addEventListener("click", () => deletePerson(person));
    actions.append(rename, remove);
    card.append(avatar, details, actions);
    list.append(card);
  }
  updateRecognitionControls();
}

async function renamePerson(person) {
  const updatedName = window.prompt("Novo nome para esta pessoa:", person.name)?.trim();
  if (!updatedName || updatedName === person.name) return;
  try {
    await databaseRequest("readwrite", "put", { ...person, name: updatedName });
    people = people.map((entry) => entry.id === person.id ? { ...entry, name: updatedName } : entry);
    renderPeople();
  } catch (error) {
    console.error("Não foi possível renomear o cadastro.", error);
    setRecognitionStatus(`Falha ao renomear: ${error.message}`, "error");
  }
}

async function deletePerson(person) {
  if (!window.confirm(`Excluir ${person.name} e seus descritores faciais deste navegador?`)) return;
  try {
    await databaseRequest("readwrite", "delete", person.id);
    people = people.filter((entry) => entry.id !== person.id);
    recognizedFaces = [];
    if (!people.length) {
      recognitionEnabled = false;
      recognitionExpiresAt = 0;
      nextRecognitionAt = 0;
    }
    renderPeople();
    if (activeSource === "photo" && recognitionEnabled) await recognizeStillPhoto();
    else if (latestResult) drawResult(latestResult, activeSource === "camera" ? video.videoWidth : photo.naturalWidth, activeSource === "camera" ? video.videoHeight : photo.naturalHeight);
  } catch (error) {
    console.error("Não foi possível excluir o cadastro.", error);
    setRecognitionStatus(`Falha ao excluir: ${error.message}`, "error");
  }
}

function updateCameraEnrollmentControl() {
  const button = el("camera-enroll-button");
  if (button) button.disabled = !storageReady || !stream || registrationBusy;
}

async function savePersonDescriptors(name, descriptors) {
  const person = { id: crypto.randomUUID(), name, descriptors, createdAt: new Date().toISOString() };
  await databaseRequest("readwrite", "put", person);
  people = [...people, person];
  renderPeople();
  el("person-form").reset();
  setRecognitionStatus(`${name} cadastrada localmente · ${descriptors.length} amostra(s) processada(s).`, "ready");
  if (recognitionEnabled && activeSource === "photo") await recognizeStillPhoto();
  else if (recognitionEnabled && stream) {
    recognizedFaces = [];
    recognitionExpiresAt = 0;
    nextRecognitionAt = 0;
  }
}

async function registerPerson(event) {
  event.preventDefault();
  const name = el("person-name").value.trim();
  const files = [...el("person-photos").files];
  if (!storageReady) {
    el("storage-notice").dataset.state = "error";
    el("storage-notice").textContent = "Armazenamento local indisponível; o cadastro não foi salvo.";
    return;
  }
  if (!name || !files.length || !el("person-consent").checked) {
    setRecognitionStatus("Informe o nome, adicione ao menos uma foto e confirme o consentimento.", "error");
    return;
  }
  if (registrationBusy || recognitionBusy) {
    setRecognitionStatus("Aguarde a análise facial atual terminar antes de cadastrar.", "loading");
    return;
  }
  registrationBusy = true;
  updateCameraEnrollmentControl();
  const button = el("register-person");
  button.disabled = true;
  button.textContent = "Analisando fotos…";
  setRecognitionStatus("Carregando reconhecimento facial local…", "loading");
  try {
    const descriptors = [];
    const invalidPhotos = [];
    for (const file of files) {
      if (!file.type.startsWith("image/")) {
        invalidPhotos.push(file.name || "arquivo inválido");
        continue;
      }
      const imageUrl = URL.createObjectURL(file);
      const image = new Image();
      try {
        image.src = imageUrl;
        await image.decode();
        const results = await extractFaceDescriptors(image);
        if (results.length !== 1 || results[0].descriptor.length !== 128) {
          invalidPhotos.push(file.name || "foto sem um único rosto válido");
        } else {
          descriptors.push(Array.from(results[0].descriptor));
        }
      } finally {
        URL.revokeObjectURL(imageUrl);
      }
    }
    if (invalidPhotos.length) throw new Error(`Cada foto deve conter exatamente um rosto válido. Confira: ${invalidPhotos.join(", ")}. Nenhum cadastro foi salvo.`);
    if (!descriptors.length) throw new Error("Nenhum descritor facial válido foi gerado.");
    await savePersonDescriptors(name, descriptors);
  } catch (error) {
    console.error("Não foi possível cadastrar a pessoa.", error);
    setRecognitionStatus(error.message || "Falha no cadastro facial.", "error");
  } finally {
    registrationBusy = false;
    button.textContent = "Cadastrar por foto";
    button.disabled = !storageReady;
    updateCameraEnrollmentControl();
    if (stream) scheduleVideoAnalysis(0);
  }
}

async function registerPersonFromCamera() {
  const name = el("person-name").value.trim();
  if (!storageReady) {
    setRecognitionStatus("Armazenamento local indisponível; o cadastro não foi salvo.", "error");
    return;
  }
  if (!stream || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
    setRecognitionStatus("Ative a câmera antes de cadastrar um rosto ao vivo.", "error");
    return;
  }
  if (!name || !el("person-consent").checked) {
    setRecognitionStatus("Informe o nome e confirme o consentimento para cadastrar pela câmera.", "error");
    return;
  }
  if (registrationBusy || recognitionBusy) {
    setRecognitionStatus("Aguarde a análise facial atual terminar antes de cadastrar.", "loading");
    return;
  }

  registrationBusy = true;
  updateCameraEnrollmentControl();
  const button = el("camera-enroll-button");
  button.textContent = "Analisando rosto…";
  setRecognitionStatus("Analisando um quadro da câmera localmente…", "loading");
  try {
    const faceapi = await loadFaceRecognitionModels();
    if (!stream || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) throw new Error("A câmera foi desligada antes de concluir o cadastro.");
    const width = Math.min(video.videoWidth, 640);
    const height = Math.round(video.videoHeight * width / video.videoWidth);
    const frame = document.createElement("canvas");
    frame.width = width;
    frame.height = height;
    const frameContext = frame.getContext("2d", { willReadFrequently: true });
    if (!frameContext) throw new Error("Não foi possível preparar o quadro da câmera.");
    frameContext.drawImage(video, 0, 0, width, height);
    const detections = await faceapi.detectAllFaces(frame, new faceapi.TinyFaceDetectorOptions({ inputSize: 160, scoreThreshold: .35 }))
      .withFaceLandmarks(true).withFaceDescriptors();
    if (detections.length !== 1 || detections[0].descriptor.length !== 128) {
      throw new Error(detections.length === 0
        ? "A câmera não encontrou um rosto válido. Ajuste o enquadramento e tente novamente."
        : "Enquadre apenas uma pessoa para cadastrar pela câmera.");
    }
    await savePersonDescriptors(name, [Array.from(detections[0].descriptor)]);
  } catch (error) {
    console.error("Não foi possível cadastrar pela câmera.", error);
    setRecognitionStatus(error.message || "Falha no cadastro pela câmera.", "error");
  } finally {
    registrationBusy = false;
    button.textContent = "Cadastrar com câmera ao vivo";
    updateCameraEnrollmentControl();
    if (stream) scheduleVideoAnalysis(0);
  }
}

function compareDescriptor(descriptor) {
  let bestPerson = null;
  let bestDistance = Infinity;
  for (const person of people) {
    for (const saved of person.descriptors) {
      if (saved.length !== descriptor.length) continue;
      let squaredDistance = 0;
      for (let i = 0; i < descriptor.length; i += 1) {
        const difference = descriptor[i] - saved[i];
        squaredDistance += difference * difference;
      }
      const distance = Math.sqrt(squaredDistance);
      if (distance < bestDistance) {
        bestDistance = distance;
        bestPerson = person;
      }
    }
  }
  if (bestPerson && bestDistance <= MATCH_DISTANCE_LIMIT) {
    return { name: bestPerson.name, matched: true, distance: bestDistance, proximity: Math.round((1 - bestDistance / MATCH_DISTANCE_LIMIT) * 100) };
  }
  return { name: "Desconhecido", matched: false, distance: Number.isFinite(bestDistance) ? bestDistance : null, proximity: 0 };
}

function normalizedBox(box, width, height) {
  return { x: (box.x + box.width / 2) / width, y: (box.y + box.height / 2) / height, width: box.width / width, height: box.height / height };
}

function findIdentityForFace(landmarks) {
  if (!recognitionEnabled) return null;
  if (performance.now() > recognitionExpiresAt) return { name: "Desconhecido", matched: false, distance: null, proximity: 0 };
  const points = landmarks;
  let left = 1, right = 0, top = 1, bottom = 0;
  for (const point of points) {
    left = Math.min(left, point.x); right = Math.max(right, point.x);
    top = Math.min(top, point.y); bottom = Math.max(bottom, point.y);
  }
  const rawCenterX = (left + right) / 2;
  const centerX = activeSource === "camera" && mirrored ? 1 - rawCenterX : rawCenterX;
  const centerY = (top + bottom) / 2;
  let nearest = null;
  let nearestDistance = Infinity;
  for (const identity of recognizedFaces) {
    if (identity.used) continue;
    const dx = centerX - identity.box.x;
    const dy = centerY - identity.box.y;
    const distance = dx * dx + dy * dy;
    if (distance < nearestDistance) { nearestDistance = distance; nearest = identity; }
  }
  if (nearest && nearestDistance < .12) { nearest.used = true; return nearest.identity; }
  return { name: "Desconhecido", matched: false, distance: null, proximity: 0 };
}

async function recognizeSource(source, width, height, epoch = cameraEpoch) {
  if (recognitionBusy || !people.length || !recognitionEnabled) return;
  recognitionBusy = true;
  setRecognitionStatus("Comparando descritores localmente…", "loading");
  try {
    const faceapi = await loadFaceRecognitionModels();
    let input = source;
    let sourceWidth = width;
    let sourceHeight = height;
    if (source === video) {
      if (!recognitionCanvas) {
        recognitionCanvas = document.createElement("canvas");
        recognitionContext = recognitionCanvas.getContext("2d", { willReadFrequently: true });
      }
      sourceWidth = Math.min(width, 640);
      sourceHeight = Math.round(height * sourceWidth / width);
      if (recognitionCanvas.width !== sourceWidth || recognitionCanvas.height !== sourceHeight) {
        recognitionCanvas.width = sourceWidth;
        recognitionCanvas.height = sourceHeight;
      }
      recognitionContext.drawImage(video, 0, 0, sourceWidth, sourceHeight);
      input = recognitionCanvas;
    }
    const detections = await faceapi.detectAllFaces(input, new faceapi.TinyFaceDetectorOptions({ inputSize: 160, scoreThreshold: .35 }))
      .withFaceLandmarks(true).withFaceDescriptors();
    if (epoch !== cameraEpoch || !recognitionEnabled) return;
    recognizedFaces = detections.map((detection) => {
      const box = normalizedBox(detection.detection.box, sourceWidth, sourceHeight);
      if (source === video && mirrored) box.x = 1 - box.x;
      return { box, identity: compareDescriptor(detection.descriptor), used: false };
    });
    recognitionExpiresAt = performance.now() + Math.max(2600, PERFORMANCE_MODES[performanceMode].recognitionInterval * 1.5);
    if (detections.length) {
      const matched = recognizedFaces.filter((entry) => entry.identity.matched).length;
      setRecognitionStatus(`${matched}/${detections.length} correspondência(s) provável(is) · índice não é probabilidade`, "ready");
    } else {
      setRecognitionStatus("Nenhum rosto válido para comparar", "ready");
    }
    if (latestResult) drawResult(latestResult, sourceWidth === video.videoWidth ? video.videoWidth : width, sourceHeight === video.videoHeight ? video.videoHeight : height);
  } catch (error) {
    console.error("Falha no reconhecimento facial local.", error);
    recognitionEnabled = false;
    recognizedFaces = [];
    updateRecognitionControls();
    if (latestResult) drawResult(latestResult, activeSource === "camera" ? video.videoWidth : photo.naturalWidth, activeSource === "camera" ? video.videoHeight : photo.naturalHeight);
    setRecognitionStatus(`Reconhecimento indisponível: ${error.message}. A detecção facial básica continua ativa.`, "error");
  } finally {
    recognitionBusy = false;
    if (stream) scheduleVideoAnalysis(0);
  }
}

async function recognizeStillPhoto() {
  if (activeSource !== "photo" || !photo.naturalWidth || !recognitionEnabled) return;
  await recognizeSource(photo, photo.naturalWidth, photo.naturalHeight);
}

async function toggleRecognition() {
  if (recognitionEnabled) {
    recognitionEnabled = false;
    recognizedFaces = [];
    recognitionExpiresAt = 0;
    updateRecognitionControls();
    if (latestResult) drawResult(latestResult, activeSource === "camera" ? video.videoWidth : photo.naturalWidth, activeSource === "camera" ? video.videoHeight : photo.naturalHeight);
    return;
  }
  if (!people.length) return;
  recognitionEnabled = true;
  updateRecognitionControls();
  setRecognitionStatus("Carregando modelos faciais sob demanda…", "loading");
  try {
    await loadFaceRecognitionModels();
    if (!recognitionEnabled) return;
    setRecognitionStatus("Reconhecimento local ativo · correspondência não é certeza", "ready");
    if (activeSource === "photo") await recognizeStillPhoto();
    else if (stream) scheduleVideoAnalysis(0);
  } catch (error) {
    recognitionEnabled = false;
    console.error("Não foi possível iniciar o reconhecimento facial.", error);
    setRecognitionStatus(`Reconhecimento não carregado: ${error.message}. A detecção facial básica continua ativa.`, "error");
    updateRecognitionControls();
  }
}

function setPerformanceMode(mode) {
  if (!PERFORMANCE_MODES[mode]) return;
  performanceMode = mode;
  const profile = PERFORMANCE_MODES[mode];
  adaptiveInterval = profile.interval;
  fastAnalysisStreak = 0;
  el("analysis-target").textContent = `${(1000 / profile.interval).toFixed(1)} / s`;
  if (stream) {
    const track = stream.getVideoTracks()[0];
    if (track?.applyConstraints) {
      track.applyConstraints({ width: { ideal: profile.width }, height: { ideal: profile.height } })
        .then(() => { if (video.videoWidth) sizeOverlay(video.videoWidth, video.videoHeight); })
        .catch((error) => {
          console.warn("Não foi possível ajustar a resolução da câmera.", error);
          setRecognitionStatus(`Modo ${profile.label}: resolução não ajustável; mantendo a câmera atual.`, "error");
        });
    }
    scheduleVideoAnalysis(0);
  }
  if (recognitionEnabled) {
    recognizedFaces = [];
    nextRecognitionAt = performance.now();
  }
}

function scheduleVideoAnalysis(delay = adaptiveInterval) {
  if (!stream || document.hidden) return;
  if (inferenceTimer) clearTimeout(inferenceTimer);
  inferenceTimer = window.setTimeout(runVideoAnalysis, Math.max(0, delay));
}

function runVideoAnalysis() {
  inferenceTimer = 0;
  if (!stream || document.hidden || isDetecting || recognitionBusy || registrationBusy || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
    scheduleVideoAnalysis(adaptiveInterval);
    return;
  }
  isDetecting = true;
  const startedAt = performance.now();
  try {
    const width = video.videoWidth;
    const height = video.videoHeight;
    const result = landmarker.detectForVideo(video, startedAt);
    drawResult(result, width, height);
    analysisSampleCount += 1;
    if (recognitionEnabled && people.length && !recognitionBusy && performance.now() >= nextRecognitionAt) {
      nextRecognitionAt = performance.now() + PERFORMANCE_MODES[performanceMode].recognitionInterval;
      recognizeSource(video, width, height, cameraEpoch);
    }
    const elapsed = performance.now() - startedAt;
    if (elapsed > adaptiveInterval * .8) {
      adaptiveInterval = Math.min(PERFORMANCE_MODES[performanceMode].interval * 3, adaptiveInterval * 1.2);
      fastAnalysisStreak = 0;
    } else if (elapsed < adaptiveInterval * .4) {
      fastAnalysisStreak += 1;
      if (fastAnalysisStreak >= 5) {
        adaptiveInterval = Math.max(PERFORMANCE_MODES[performanceMode].interval, adaptiveInterval * .9);
        fastAnalysisStreak = 0;
      }
    }
  } catch (error) {
    console.error("Falha na detecção facial.", error);
    setStatus("A detecção facial encontrou um erro. Tente reiniciar a câmera.", "error");
  } finally {
    isDetecting = false;
    scheduleVideoAnalysis(adaptiveInterval);
  }
}

function renderInterface(now) {
  interfaceFrame = 0;
  if (!stream) return;
  if (pendingRenderResult && renderedResultVersion !== latestResultVersion) {
    const pending = pendingRenderResult;
    pendingRenderResult = null;
    renderedResultVersion = latestResultVersion;
    paintResult(pending.result, pending.width, pending.height);
  }
  if (!uiFpsStartedAt) {
    uiFpsStartedAt = now;
    videoFrameSample = video.getVideoPlaybackQuality?.().totalVideoFrames ?? video.webkitDecodedFrameCount ?? null;
    analysisSampleAt = now;
  }
  if (now - uiFpsStartedAt >= 1000) {
    const quality = video.getVideoPlaybackQuality?.();
    const totalFrames = quality?.totalVideoFrames ?? video.webkitDecodedFrameCount;
    const fps = Number.isFinite(totalFrames) && Number.isFinite(videoFrameSample) ? totalFrames - videoFrameSample : null;
    const elapsed = now - analysisSampleAt;
    el("video-fps-value").textContent = fps === null ? "--" : String(fps).padStart(2, "0");
    el("analysis-rate-value").textContent = String(Math.round((analysisSampleCount * 1000) / elapsed)).padStart(2, "0");
    analysisSampleCount = 0;
    uiFpsStartedAt = now;
    videoFrameSample = Number.isFinite(totalFrames) ? totalFrames : null;
    analysisSampleAt = now;
  }
  interfaceFrame = requestAnimationFrame(renderInterface);
}



function project(point, width, height, mirror) {
  return { x: (mirror ? 1 - point.x : point.x) * width, y: point.y * height };
}

function drawConnections(landmarks, connections, width, height, mirror, color, lineWidth, alpha = 1) {
  if (!connections?.length) return;
  ctx.save();
  ctx.strokeStyle = color;
  ctx.globalAlpha = alpha;
  ctx.lineWidth = lineWidth;
  ctx.beginPath();
  for (const connection of connections) {
    const start = landmarks[connection.start];
    const end = landmarks[connection.end];
    if (!start || !end) continue;
    const a = project(start, width, height, mirror);
    const b = project(end, width, height, mirror);
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
  }
  ctx.stroke();
  ctx.restore();
}

function drawLabel(text, x, y, maxWidth) {
  ctx.save();
  ctx.font = "600 10px Archivo, sans-serif";
  let visibleText = text;
  while (visibleText.length > 1 && ctx.measureText(visibleText).width > maxWidth - 16) {
    visibleText = `${visibleText.slice(0, -2)}…`;
  }
  const labelWidth = ctx.measureText(visibleText).width + 12;
  const left = Math.max(0, Math.min(maxWidth - labelWidth, x));
  const top = Math.max(3, y - 20);
  ctx.fillStyle = "rgba(16, 17, 20, .96)";
  ctx.fillRect(left, top, labelWidth, 16);
  ctx.fillStyle = "#E7E9EF";
  ctx.fillText(visibleText, left + 6, top + 11);
  ctx.restore();
}

function drawFace(landmarks, faceIndex, width, height) {
  const mirror = activeSource === "camera" && mirrored;
  if (el("mesh-toggle").checked) {
    drawConnections(landmarks, CONNECTIONS.tessellation, width, height, mirror, "#C7CBD3", Math.max(.55, width / 1150), .42);
    drawConnections(landmarks, CONNECTIONS.contours, width, height, mirror, "#E7E9EF", Math.max(1, width / 850), .88);
    drawConnections(landmarks, CONNECTIONS.irises, width, height, mirror, "#7298D5", Math.max(1, width / 700), .95);
  }
  if (el("markers-toggle").checked) {
    for (const index of [1, 4, 6, 10, 13, 14, 33, 133, 159, 263, 362, 386, 468, 473]) {
      const point = landmarks[index];
      if (!point) continue;
      const { x, y } = project(point, width, height, mirror);
      ctx.beginPath();
      ctx.fillStyle = index === 468 || index === 473 ? "#F0869D" : "#7298D5";
      ctx.shadowColor = "#7298D5";
      ctx.shadowBlur = 5;
      ctx.arc(x, y, Math.max(1.5, width / 350), 0, Math.PI * 2);
      ctx.fill();
    }
  }
  if (el("labels-toggle").checked) {
    const forehead = project(landmarks[10], width, height, mirror);
    const identity = findIdentityForFace(landmarks);
    const label = recognitionEnabled
      ? identity?.matched ? `${identity.name} · prox. ${identity.proximity}%` : "Desconhecido"
      : `ROSTO 0${faceIndex + 1}`;
    drawLabel(label, forehead.x - 50, forehead.y, width);
  }
}

function categoryScore(categories, name) {
  const category = categories?.find((item) => item.categoryName === name);
  return category ? category.score : 0;
}

function average(...values) {
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function readExpressions(blendshapes) {
  const categories = blendshapes?.categories ?? [];
  const score = (...names) => average(...names.map((name) => categoryScore(categories, name)));
  const smile = score("mouthSmileLeft", "mouthSmileRight");
  const mouth = categoryScore(categories, "jawOpen");
  const blink = score("eyeBlinkLeft", "eyeBlinkRight");
  const brows = score("browInnerUp", "browOuterUpLeft", "browOuterUpRight");
  return {
    smile: { value: smile, caption: smile >= .48 ? "Sorriso identificado" : smile >= .22 ? "Leve sorriso" : "Neutro" },
    mouth: { value: mouth, caption: mouth >= .42 ? "Aberta" : mouth >= .2 ? "Entreaberta" : "Fechada" },
    eyes: { value: blink, caption: blink >= .48 ? "Piscada identificada" : blink >= .2 ? "Semicerrados" : "Olhos abertos" },
    brows: { value: brows, caption: brows >= .38 ? "Elevadas" : "Relaxadas" }
  };
}

function showReadings(faceCount, result) {
  const faceCountText = String(faceCount).padStart(2, "0");
  if (el("face-count").textContent !== faceCountText) el("face-count").textContent = faceCountText;
  if (!faceCount || !result) {
    readingEmpty.hidden = false;
    readingResults.hidden = true;
    return;
  }
  readingEmpty.hidden = true;
  readingResults.hidden = false;
  const readings = readExpressions(result.faceBlendshapes?.[0]);
  for (const [key, reading] of Object.entries(readings)) {
    const percentage = Math.round(Math.min(1, Math.max(0, reading.value)) * 100);
    const value = el(`${key}-value`);
    const meter = el(`${key}-meter`);
    const caption = el(`${key}-caption`);
    const valueText = `${percentage}%`;
    if (value.textContent !== valueText) value.textContent = valueText;
    if (meter.style.width !== valueText) meter.style.width = valueText;
    if (caption.textContent !== reading.caption) caption.textContent = reading.caption;
  }
}

function paintSignature(result, width, height) {
  let hash = 2166136261;
  const mix = (value) => {
    hash ^= value | 0;
    hash = Math.imul(hash, 16777619);
  };
  mix(width);
  mix(height);
  mix(activeSource === "camera" && mirrored ? 1 : 0);
  mix(el("mesh-toggle").checked ? 1 : 0);
  mix(el("markers-toggle").checked ? 1 : 0);
  mix(el("labels-toggle").checked ? 1 : 0);
  mix(recognitionEnabled ? 1 : 0);
  mix(recognitionExpiresAt > performance.now() ? 1 : 0);
  for (const landmarks of result.faceLandmarks ?? []) {
    mix(landmarks.length);
    for (const point of landmarks) {
      mix(Math.round(point.x * 512));
      mix(Math.round(point.y * 512));
    }
  }
  for (const blendshape of result.faceBlendshapes ?? []) {
    for (const category of blendshape.categories ?? []) mix(Math.round(category.score * 100));
  }
  for (const entry of recognizedFaces) {
    for (const character of entry.identity.name) mix(character.charCodeAt(0));
    mix(entry.identity.matched ? 1 : 0);
    mix(entry.identity.proximity);
    mix(Math.round(entry.box.x * 512));
    mix(Math.round(entry.box.y * 512));
  }
  return String(hash >>> 0);
}

function paintResult(result, width, height) {
  const signature = paintSignature(result, width, height);
  if (signature === lastPaintSignature) return;
  lastPaintSignature = signature;
  clearOverlay();
  recognizedFaces.forEach((entry) => { entry.used = false; });
  const faces = result.faceLandmarks ?? [];
  faces.forEach((landmarks, index) => drawFace(landmarks, index, width, height));
  showReadings(faces.length, result);
}

function drawResult(result, width, height) {
  latestResult = result;
  latestResultVersion += 1;
  pendingRenderResult = { result, width, height };
  if (!stream) {
    const pending = pendingRenderResult;
    pendingRenderResult = null;
    renderedResultVersion = latestResultVersion;
    paintResult(pending.result, pending.width, pending.height);
  }
}

async function stopCamera(updateStatus = false) {
  cameraEpoch += 1;

  recognizedFaces = [];
  recognitionExpiresAt = 0;
  nextRecognitionAt = performance.now();
  if (inferenceTimer) clearTimeout(inferenceTimer);
  inferenceTimer = 0;
  if (interfaceFrame) cancelAnimationFrame(interfaceFrame);
  interfaceFrame = 0;
  uiFpsStartedAt = 0;
  analysisSampleCount = 0;
  lastVideoTime = -1;
  if (stream) {
    stream.getTracks().forEach((track) => track.stop());
    stream = null;
  }
  video.srcObject = null;
  video.classList.remove("is-mirrored");
  if (activeSource === "camera") {
    activeSource = null;
    updateCameraEnrollmentControl();
    viewport.style.aspectRatio = "";
    clearOverlay();
    latestResult = null;
    showReadings(0, null);
    updateSourceView();
    updateResolution(0, 0);
  }
  el("camera-button-label").textContent = "Ativar câmera";
  el("mirror-option").hidden = true;
  el("video-fps-value").textContent = "--";
  el("analysis-rate-value").textContent = "--";
  if (updateStatus) setStatus("Câmera desligada · pronto para outra imagem", "ready");
}

async function setRunningMode(mode) {
  if (landmarker) await landmarker.setOptions({ runningMode: mode });
}

async function toggleCamera() {
  if (stream) {
    await stopCamera(true);
    return;
  }
  if (!navigator.mediaDevices?.getUserMedia) {
    setStatus("Câmera indisponível. Abra esta página em localhost ou HTTPS.", "error");
    return;
  }
  setStatus("Solicitando acesso à câmera…", "loading");
  try {
    await setRunningMode("VIDEO");

    recognizedFaces = [];
    recognitionExpiresAt = 0;
    nextRecognitionAt = 0;
    const profile = PERFORMANCE_MODES[performanceMode];
    stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: { facingMode: "user", width: { ideal: profile.width }, height: { ideal: profile.height } }
    });
    video.srcObject = stream;
    await video.play();
    activeSource = "camera";
    updateCameraEnrollmentControl();
    viewport.style.aspectRatio = `${video.videoWidth} / ${video.videoHeight}`;
    mirrored = el("mirror-toggle").checked;
    video.classList.toggle("is-mirrored", mirrored);
    el("camera-button-label").textContent = "Desligar câmera";
    updateSourceView();
    sizeOverlay(video.videoWidth, video.videoHeight);
    setStatus("Câmera ativa · analisando em tempo real", "active");
    scheduleVideoAnalysis(0);
    interfaceFrame = requestAnimationFrame(renderInterface);
    revealStage();
  } catch (error) {
    if (stream) stream.getTracks().forEach((track) => track.stop());
    stream = null;
    video.srcObject = null;
    console.error("Não foi possível iniciar a câmera.", error);
    const message = error.name === "NotAllowedError"
      ? "Permita o acesso à câmera nas configurações do navegador."
      : error.name === "NotFoundError"
        ? "Nenhuma câmera foi encontrada neste dispositivo."
        : "Não foi possível iniciar a câmera. Verifique as permissões e tente novamente.";
    setStatus(message, "error");
  }
}

async function loadPhoto(file) {
  if (!file) return;
  if (!landmarker) {
    setStatus("Aguarde o carregamento do detector antes de escolher uma fotografia.", "loading");
    return;
  }
  if (!file.type.startsWith("image/")) {
    setStatus("Escolha um arquivo de imagem válido.", "error");
    return;
  }
  await stopCamera();
  if (photoUrl) URL.revokeObjectURL(photoUrl);
  photoUrl = URL.createObjectURL(file);
  photo.onload = async () => {
    activeSource = "photo";
    viewport.style.aspectRatio = `${photo.naturalWidth} / ${photo.naturalHeight}`;
    updateSourceView();
    sizeOverlay(photo.naturalWidth, photo.naturalHeight);
    setStatus("Fotografia carregada · analisando…", "loading");
    revealStage();
    try {
      await landmarker.setOptions({ runningMode: "IMAGE" });
      const result = landmarker.detect(photo);
      drawResult(result, photo.naturalWidth, photo.naturalHeight);
      el("video-fps-value").textContent = "--";
      if (recognitionEnabled) await recognizeStillPhoto();
      setStatus(result.faceLandmarks?.length ? "Leitura concluída · fotografia analisada" : "Fotografia analisada · nenhum rosto encontrado", "ready");
    } catch (error) {
      console.error("Não foi possível analisar a fotografia.", error);
      setStatus("Não foi possível analisar esta fotografia.", "error");
    }
  };
  photo.onerror = () => setStatus("Não foi possível abrir esta imagem. Escolha outro arquivo.", "error");
  photo.src = photoUrl;
}

function applyCyanotype(imageData) {
  const pixels = imageData.data;
  for (let i = 0; i < pixels.length; i += 4) {
    const luminance = (pixels[i] * .2126 + pixels[i + 1] * .7152 + pixels[i + 2] * .0722) / 255;
    pixels[i] = Math.round(5 + luminance * 53);
    pixels[i + 1] = Math.round(15 + luminance * 152);
    pixels[i + 2] = Math.round(29 + luminance * 219);
  }
  return imageData;
}

function captureImage() {
  if (!activeSource || !landmarker || captures.length >= MAX_CAPTURES) return;
  const source = activeSource === "camera" ? video : photo;
  const width = activeSource === "camera" ? video.videoWidth : photo.naturalWidth;
  const height = activeSource === "camera" ? video.videoHeight : photo.naturalHeight;
  if (!width || !height) return;
  const output = document.createElement("canvas");
  output.width = width;
  output.height = height;
  const outputCtx = output.getContext("2d", { willReadFrequently: true });
  if (activeSource === "camera" && mirrored) {
    outputCtx.translate(width, 0);
    outputCtx.scale(-1, 1);
  }
  outputCtx.drawImage(source, 0, 0, width, height);
  if (isCyanotype) {
    const pixels = outputCtx.getImageData(0, 0, width, height);
    outputCtx.putImageData(applyCyanotype(pixels), 0, 0);
  }
  const bounds = viewport.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  const scale = Math.min(bounds.width / width, bounds.height / height);
  const offsetX = (bounds.width - width * scale) / 2;
  const offsetY = (bounds.height - height * scale) / 2;
  outputCtx.setTransform(1, 0, 0, 1, 0, 0);
  outputCtx.drawImage(overlay, offsetX * ratio, offsetY * ratio, width * scale * ratio, height * scale * ratio, 0, 0, width, height);
  const dataUrl = output.toDataURL("image/jpeg", .92);
  captures.push({ dataUrl, filename: `leitura-cianotipo-${String(captures.length + 1).padStart(2, "0")}.jpg` });
  renderCaptures();
  playCaptureFlash();
}

function renderCaptures() {
  const slotContainer = el("capture-slots");
  slotContainer.replaceChildren();
  for (let i = 0; i < MAX_CAPTURES; i += 1) {
    const slot = document.createElement("div");
    slot.className = `capture-slot${captures[i] ? " is-filled" : ""}`;
    if (captures[i]) {
      const image = document.createElement("img");
      image.src = captures[i].dataUrl;
      image.alt = `Captura ${i + 1}`;
      const download = document.createElement("a");
      download.href = captures[i].dataUrl;
      download.download = captures[i].filename;
      download.textContent = "Baixar JPG";
      download.setAttribute("aria-label", `Baixar captura ${i + 1} como JPG`);
      slot.append(image, download);
    } else {
      slot.setAttribute("aria-label", `Espaço vazio ${i + 1}`);
    }
    slotContainer.append(slot);
  }
  el("capture-count").textContent = `${String(captures.length).padStart(2, "0")} / 04`;
  el("clear-captures").disabled = captures.length === 0;
  el("capture-button").disabled = !activeSource || !landmarker || captures.length >= MAX_CAPTURES;
}

function setFilter(cyanotype) {
  isCyanotype = cyanotype;
  for (const [button, selected] of [[el("cyanotype-button"), cyanotype], [el("color-button"), !cyanotype]]) {
    button.classList.toggle("is-selected", selected);
    button.setAttribute("aria-pressed", String(selected));
  }
  video.classList.toggle("is-color", !cyanotype);
  photo.classList.toggle("is-color", !cyanotype);
}

el("camera-button").addEventListener("click", toggleCamera);
el("photo-button").addEventListener("click", () => el("photo-input").click());
el("photo-input").addEventListener("change", (event) => {
  const [file] = event.target.files ?? [];
  if (file) loadPhoto(file);
  event.target.value = "";
});
el("cyanotype-button").addEventListener("click", () => setFilter(true));
el("color-button").addEventListener("click", () => setFilter(false));
el("capture-button").addEventListener("click", captureImage);
el("person-form").addEventListener("submit", registerPerson);
el("camera-enroll-button").addEventListener("click", registerPersonFromCamera);
el("recognition-toggle").addEventListener("click", toggleRecognition);
el("performance-mode").addEventListener("change", (event) => setPerformanceMode(event.target.value));
el("clear-captures").addEventListener("click", () => {
  captures = [];
  renderCaptures();
});
el("mirror-toggle").addEventListener("change", (event) => {
  mirrored = event.target.checked;
  video.classList.toggle("is-mirrored", mirrored);
});
for (const id of ["mesh-toggle", "markers-toggle", "labels-toggle"]) {
  el(id).addEventListener("change", () => {
    if (activeSource === "photo" && latestResult) drawResult(latestResult, photo.naturalWidth, photo.naturalHeight);
  });
}
window.addEventListener("resize", () => {
  if (activeSource === "camera" && video.videoWidth) sizeOverlay(video.videoWidth, video.videoHeight);
  else if (activeSource === "photo" && photo.naturalWidth) {
    sizeOverlay(photo.naturalWidth, photo.naturalHeight);
    if (latestResult) drawResult(latestResult, photo.naturalWidth, photo.naturalHeight);
  }
});
window.addEventListener("beforeunload", () => {
  if (stream) stream.getTracks().forEach((track) => track.stop());
  if (photoUrl) URL.revokeObjectURL(photoUrl);
});

renderCaptures();
el("photo-button").disabled = true;
el("analysis-target").textContent = `${(1000 / PERFORMANCE_MODES[performanceMode].interval).toFixed(1)} / s`;
initializePeopleStorage();
initializeDetector();
