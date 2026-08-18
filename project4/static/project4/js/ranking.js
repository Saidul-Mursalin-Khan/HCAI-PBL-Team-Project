(() => {
  const form = document.querySelector("[data-ranking-form]");
  if (!form) return;

  const list = form.querySelector("[data-ranking-list]");
  const rankingInput = form.querySelector('input[name="ranking"]');
  const confirm = form.querySelector("[data-ranking-confirm]");
  const submit = form.querySelector("[data-ranking-submit]");
  const reset = form.querySelector("[data-ranking-reset]");
  const status = form.querySelector("[data-ranking-status]");
  const initialOrder = Array.from(list.children);
  let dragged = null;

  const sync = (announcement = "") => {
    const cards = Array.from(list.children);
    cards.forEach((card, index) => {
      card.querySelector(".p4-rank-number").textContent = String(index + 1);
      card.querySelector('[data-move="up"]').disabled = index === 0;
      card.querySelector('[data-move="down"]').disabled = index === cards.length - 1;
    });
    rankingInput.value = JSON.stringify(cards.map((card) => Number(card.dataset.movieId)));
    submit.disabled = !confirm.checked;
    if (announcement) status.textContent = announcement;
  };

  list.addEventListener("dragstart", (event) => {
    dragged = event.target.closest("[data-movie-id]");
    if (!dragged) return;
    dragged.classList.add("is-dragging");
    event.dataTransfer.effectAllowed = "move";
  });

  list.addEventListener("dragend", () => {
    if (dragged) dragged.classList.remove("is-dragging");
    dragged = null;
    sync("Ranking updated.");
  });

  list.addEventListener("dragover", (event) => {
    event.preventDefault();
    const target = event.target.closest("[data-movie-id]");
    if (!dragged || !target || dragged === target) return;
    const box = target.getBoundingClientRect();
    const insertAfter = event.clientY > box.top + box.height / 2;
    list.insertBefore(dragged, insertAfter ? target.nextSibling : target);
  });

  list.addEventListener("click", (event) => {
    const button = event.target.closest("[data-move]");
    if (!button) return;
    const card = button.closest("[data-movie-id]");
    const direction = button.dataset.move;
    if (direction === "up" && card.previousElementSibling) {
      list.insertBefore(card, card.previousElementSibling);
    }
    if (direction === "down" && card.nextElementSibling) {
      list.insertBefore(card.nextElementSibling, card);
    }
    card.focus({ preventScroll: true });
    sync(`${card.querySelector("strong").textContent} moved ${direction}.`);
  });

  confirm.addEventListener("change", () => sync());
  reset.addEventListener("click", () => {
    initialOrder.forEach((card) => list.appendChild(card));
    confirm.checked = false;
    sync("Original order restored.");
  });

  sync();
})();

