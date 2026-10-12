const $ = (id) => document.getElementById(id);
let file = null;

async function health() {
  try {
    const r = await fetch("/api/health");
    const j = await r.json();
    const el = $("status");
    if (j.model_loaded) {
      el.textContent = `model ready · ${j.params_millions}M params · ${j.img_size}px`;
      el.className = "status ok";
    } else {
      el.textContent = j.error || "no trained model";
      el.className = "status err";
    }
  } catch (e) {
    $("status").textContent = "backend not reachable";
    $("status").className = "status err";
  }
}
health();

function bindSlider(id, out, digits = 2) {
  const s = $(id), o = $(out);
  const upd = () => (o.textContent = (+s.value).toFixed(digits));
  s.addEventListener("input", upd); upd();
}
bindSlider("fg_thr", "fg_out");
bindSlider("ctr_thr", "ctr_out");
bindSlider("bnd_weight", "bnd_out");
bindSlider("min_area", "area_out", 0);

const drop = $("drop"), input = $("file");
drop.addEventListener("click", () => input.click());
["dragenter", "dragover"].forEach((e) =>
  drop.addEventListener(e, (ev) => { ev.preventDefault(); drop.classList.add("hot"); }));
["dragleave", "drop"].forEach((e) =>
  drop.addEventListener(e, () => drop.classList.remove("hot")));
drop.addEventListener("drop", (ev) => {
  ev.preventDefault();
  if (ev.dataTransfer.files.length) setFile(ev.dataTransfer.files[0]);
});
input.addEventListener("change", () => input.files.length && setFile(input.files[0]));

function setFile(f) {
  file = f;
  $("go").disabled = false;
  const url = URL.createObjectURL(f);
  $("input_img").src = url;
  $("base_img").src = url;
  $("overlay_img").src = "";
  $("count").textContent = "";
}

$("go").addEventListener("click", async () => {
  if (!file) return;
  const btn = $("go");
  btn.disabled = true; btn.textContent = "Segmenting…";
  const fd = new FormData();
  fd.append("file", file);
  fd.append("fg_thr", $("fg_thr").value);
  fd.append("ctr_thr", $("ctr_thr").value);
  fd.append("bnd_weight", $("bnd_weight").value);
  fd.append("min_area", $("min_area").value);
  try {
    const r = await fetch("/api/segment", { method: "POST", body: fd });
    const j = await r.json();
    if (!r.ok) throw new Error(j.error || "error");
    $("overlay_img").src = j.overlay;
    $("count").textContent = `${j.count} instance${j.count === 1 ? "" : "s"}`;
    const tb = document.querySelector("#inst_table tbody");
    tb.innerHTML = "";
    j.instances.forEach((it) => {
      const tr = document.createElement("tr");
      tr.innerHTML = `<td>${it.id}</td>
        <td><span class="swatch" style="background:${it.color}"></span></td>
        <td>${it.area}</td><td>${it.bbox.join(", ")}</td><td>${it.score}</td>`;
      tb.appendChild(tr);
    });
  } catch (e) {
    $("count").textContent = "failed: " + e.message;
  } finally {
    btn.disabled = false; btn.textContent = "Segment";
  }
});
