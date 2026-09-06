const apiUrl = () => document.querySelector("#api-url").value.replace(/\/+$/, "");
const history = [];
const documents = [];
const $ = (selector) => document.querySelector(selector);

function showError(error) {
  $("#session-status").textContent = `Error: ${error.message}`;
}

function addTurn(role, content) {
  history.push({ role, content });
  const item = document.createElement("p");
  item.className = role;
  item.textContent = `${role === "assistant" ? "MediKiosk" : "Patient"}: ${content}`;
  $("#conversation").appendChild(item);
}

async function request(path, options = {}) {
  const response = await fetch(`${apiUrl()}${path}`, options);
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(payload.detail || `Request failed (${response.status})`);
  }
  return payload;
}

$("#login").addEventListener("click", async () => {
  try {
    const patientId = $("#patient-id").value.trim();
    if (!patientId) throw new Error("Enter a patient ID.");
    const result = await request("/auth/patient/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ patient_id: patientId }),
    });
    $("#session-status").textContent = result.message;
    const first = await request("/converse", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ history: [] }),
    });
    addTurn("assistant", first.next_question);
  } catch (error) {
    showError(error);
  }
});

$("#send").addEventListener("click", async () => {
  try {
    const message = $("#message").value.trim();
    if (!message) throw new Error("Enter a patient answer.");
    addTurn("patient", message);
    $("#message").value = "";
    const result = await request("/converse", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ history }),
    });
    if (result.next_question) addTurn("assistant", result.next_question);
  } catch (error) {
    showError(error);
  }
});

$("#upload").addEventListener("click", async () => {
  try {
    const file = $("#document").files[0];
    if (!file) throw new Error("Choose a document first.");
    const form = new FormData();
    form.append("file", file);
    const result = await request("/upload-document", { method: "POST", body: form });
    documents.push({
      storage_path: result.storage_path,
      extracted_document: result.extracted_document,
    });
    $("#document-result").textContent = JSON.stringify(result, null, 2);
  } catch (error) {
    showError(error);
  }
});

$("#summary").addEventListener("click", async () => {
  try {
    const transcript = history
      .map((turn) => `${turn.role}: ${turn.content}`)
      .join("\n");
    const result = await request("/generate-summary", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        patient_id: $("#patient-id").value.trim() || null,
        transcript,
        documents,
      }),
    });
    $("#summary-result").textContent = JSON.stringify(result, null, 2);
  } catch (error) {
    showError(error);
  }
});
