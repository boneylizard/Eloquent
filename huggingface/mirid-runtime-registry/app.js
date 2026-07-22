const manifestUrl =
  "https://huggingface.co/datasets/boneylizardwizard/mirid-runtime-packs/resolve/main/runtime-packages.manifest.json";

const packageList = document.querySelector("#package-list");
const status = document.querySelector("#status");
const platformFilter = document.querySelector("#platform-filter");
const familyFilter = document.querySelector("#family-filter");
const acceleratorFilter = document.querySelector("#accelerator-filter");
let packages = [];

function addOptions(select, values) {
  [...new Set(values)].sort().forEach((value) => {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = value;
    select.append(option);
  });
}

function formatBytes(bytes) {
  const units = ["B", "KB", "MB", "GB"];
  let value = bytes;
  let unitIndex = 0;
  while (value >= 1024 && unitIndex < units.length - 1) {
    value /= 1024;
    unitIndex += 1;
  }
  return `${value.toFixed(unitIndex > 1 ? 1 : 0)} ${units[unitIndex]}`;
}

function render() {
  const visible = packages.filter(
    (item) => {
      const family = item.family || "binding";
      return (
        (!platformFilter.value || item.platform === platformFilter.value) &&
        (!familyFilter.value || family === familyFilter.value) &&
        (!acceleratorFilter.value || item.accelerator === acceleratorFilter.value)
      );
    },
  );
  packageList.replaceChildren();
  visible.forEach((item) => {
    const article = document.createElement("article");
    const family = item.family || "Python binding";
    article.innerHTML = `
      <div class="package-heading">
        <div>
          <p class="meta">${family} · ${item.platform}</p>
          <h2>${item.engine || item.package}</h2>
        </div>
        <span class="backend">${item.accelerator}</span>
      </div>
      <dl>
        <div><dt>Upstream</dt><dd>${item.source.repository || item.package}</dd></div>
        <div><dt>Release</dt><dd>${item.source.revision || item.version || "current"}</dd></div>
        <div><dt>Size</dt><dd>${formatBytes(item.source.size)}</dd></div>
        <div><dt>Validation</dt><dd>${item.validation.replaceAll("-", " ")}</dd></div>
        <div><dt>SHA-256</dt><dd><code>${item.source.sha256}</code></dd></div>
      </dl>
      <a class="download" href="${item.downloadUrl}">Download verified package</a>
    `;
    packageList.append(article);
  });
  status.textContent = `${visible.length} verified package${visible.length === 1 ? "" : "s"}`;
}

async function loadManifest() {
  try {
    const response = await fetch(manifestUrl, { cache: "no-store" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const manifest = await response.json();
    packages = manifest.packages;
    addOptions(platformFilter, packages.map((item) => item.platform));
    addOptions(acceleratorFilter, packages.map((item) => item.accelerator));
    render();
  } catch (error) {
    status.textContent = "The package catalogue could not be reached. The repository remains available below.";
    console.error(error);
  }
}

[platformFilter, familyFilter, acceleratorFilter].forEach((filter) =>
  filter.addEventListener("change", render),
);
loadManifest();
