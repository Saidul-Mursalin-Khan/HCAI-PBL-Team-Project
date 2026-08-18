(() => {
  const form = document.querySelector("[data-timed-form]");
  if (!form) return;

  const startedAt = performance.now();
  const timingInput = form.querySelector('input[name="response_time_ms"]');
  form.addEventListener("submit", () => {
    timingInput.value = String(Math.max(0, Math.round(performance.now() - startedAt)));
    window.setTimeout(() => {
      form.querySelectorAll('button[type="submit"]').forEach((button) => {
        button.disabled = true;
      });
    }, 0);
  });
})();
