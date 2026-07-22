const copyButtons = document.querySelectorAll('[data-copy]');

for (const button of copyButtons) {
  button.addEventListener('click', async () => {
    const value = button.getAttribute('data-copy');
    if (!value) return;

    try {
      await navigator.clipboard.writeText(value);
      const originalLabel = button.textContent;
      button.textContent = 'Copied';
      window.setTimeout(() => {
        button.textContent = originalLabel;
      }, 1600);
    } catch {
      button.textContent = 'Select and copy';
    }
  });
}

const yearTargets = document.querySelectorAll('[data-current-year]');
for (const target of yearTargets) {
  target.textContent = String(new Date().getFullYear());
}
